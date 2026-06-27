import numpy as np
from numba import njit, prange
import scipy.sparse as sp
import sknetwork as sn
from sknetwork.clustering import BaseClustering, Louvain, Leiden
from sknetwork.utils.check import check_format, check_random_state, get_probs
from sknetwork.topology import get_core_decomposition


@njit(cache=True, nogil=True, parallel=True)
def _internal_edge(indptr, indices, partition):
    is_internal_edge = np.empty(len(indices), dtype="bool")
    for n1 in prange(len(indptr) - 1):
        for offset, n2 in enumerate(indices[indptr[n1] : indptr[n1 + 1]]):
            is_internal_edge[indptr[n1] + offset] = partition[n1] == partition[n2]
    return is_internal_edge


def gam(g, u, v, method="rand", adjusted=True):
    """
    Compute one of 11 graph-aware measures to compare graph partitions.

    Parameters
    ----------
    g: sp.sparse
        adjaceny matrix of the graph on which the partitions are defined.

    u: array_like 
        Partiton of nodes. A numpy array of length n where u[i] = j means node i
        is in part j. Parts must be labeled 0-n_parts.

    v: array_like
        Partiton of nodes. A numpy array of length n where u[i] = j means node i
        is in part j. Parts must be labeled 0-n_parts.

    method: str
      one of 'rand', 'jaccard', 'mn', 'gmn', 'min' or 'max'

    adjusted: bool
      if True, return adjusted measure (preferred). All measures can be
      adjusted except 'jaccard'.

    Returns
    -------
    float: A graph-aware similarity measure between vertex partitions u and v.

    Examples
    --------
    >>> g = sn.data.karate_club()
    >>> part1 = sn.clustering.Louvain().fit_predict(g)
    >>> part2 = sn.clustering.PropagationClustering().fit_predict(g)
    >>> print(gam(g, part1, part2))

     Reference
    ---------
    Valérie Poulin and François Théberge, "Comparing Graph Clusterings:
    Set Partition Measures vs. Graph-aware Measures", IEEE Transactions
    on Pattern Analysis and Machine Intelligence 43, 6 (2021)
    https://doi.org/10.1109/TPAMI.2020.3009862
    """
    g = sp.triu(g).tocsr()
    bu = _internal_edge(g.indptr, g.indices, np.asarray(u))
    bv = _internal_edge(g.indptr, g.indices, np.asarray(v))
    su = np.sum(bu)
    sv = np.sum(bv)
    suv = np.sum(bu * bv)
    m = len(bu)
    ## all adjusted measures
    if adjusted:
        if method == "jaccard":
            raise ValueError("no adjusted jaccard measure, set adjusted=False")
        elif method == "rand" or method == "mn":
            return (suv - su * sv / m) / (np.average([su, sv]) - su * sv / m)
        elif method == "gmn":
            return (suv - su * sv / m) / (np.sqrt(su * sv) - su * sv / m)
        elif method == "min":
            return (suv - su * sv / m) / (np.min([su, sv]) - su * sv / m)
        elif method == "max":
            return (suv - su * sv / m) / (np.max([su, sv]) - su * sv / m)
        else:
            raise ValueError(
                f"Method not found. Should be one of ['jaccard', 'rand', 'gmn', 'min', 'max']. Got {method}"
            )
    ## all non-adjusted measures
    else:
        if method == "jaccard":
            union_b = np.sum((bu + bv) > 0)
            return suv / union_b
        elif method == "rand":
            return 1 - (su + sv) / m + 2 * suv / m
        elif method == "mn":
            return suv / np.average([su, sv])
        elif method == "gmn":
            return suv / np.sqrt(su * sv)
        elif method == "min":
            return suv / np.min([su, sv])
        elif method == "max":
            return suv / np.max([su, sv])
        else:
            raise ValueError(
                f"Method not found. Should be one of ['jaccard', 'rand', 'gmn', 'min', 'max']. Got {method}"
            )


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _ecg_weights(indptr, indices, data, partitions):
    for n1 in prange(len(indptr) - 1):
        for offset, n2 in enumerate(indices[indptr[n1] : indptr[n1 + 1]]):
            data[indptr[n1] + offset] = np.sum(partitions[n1, :] == partitions[n2, :])
    return data


@njit(cache=True, nogil=True, parallel=True)
def _min_weight_outside_core(indptr, data, core, min_weight):
    for node in prange(len(indptr) - 1):
        if core[node] < 2:
            data[indptr[node] : indptr[node + 1]] = min_weight


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _compute_refuse_scores(indptr, indices, data, labels):
    overall_refuse = np.empty(len(indptr) - 1)
    community_refuse = np.empty(len(indptr) - 1)
    for node in prange(len(indptr) - 1):
        degree = indptr[node + 1] - indptr[node]
        edge_weights = data[indptr[node] : indptr[node + 1]]
        weighted_degree = np.sum(edge_weights)
        neighbor_is_in_community = (
            labels[indices[indptr[node] : indptr[node + 1]]] == labels[node]
        )
        in_community_weighted_degree = np.sum(edge_weights[neighbor_is_in_community])
        overall_refuse[node] = (degree - weighted_degree) / degree
        community_refuse[node] = in_community_weighted_degree / weighted_degree
    return overall_refuse, community_refuse


class ECG(BaseClustering):
    """
    Stable ensemble-based graph clustering. The ensemble consists of single-level randomized Louvain;
    each member of the ensemble gets a "vote" to determine if the edges are intra-community or not;
    the votes are aggregated into ECG edge-weights in range [0,1]; a final (full depth) Leiden is run
    using those edge weights;

    Parameters
    ----------
    resolution :
        Resolution parameter.
    ens_size :
        Number of Louvain runs in the ensemble
    min_weight :
        Minimum edge weight
    sort_clusters :
        If ``True``, sort labels in decreasing order of cluster size.
    return_probs :
        If ``True``, return the probability distribution over clusters (soft clustering).
    return_aggregate :
        If ``True``, return the adjacency matrix of the graph between clusters.
    refuse_score :
        Flag to compute refuse to cluster scores.
    random_state :
        Random number generator or random seed. If None, numpy.random is used.
    rng :
        numpy Generator object to use. If passed random_state is not used

    Attributes
    ----------
    labels_ : np.ndarray, shape (n,)
        Label of each node.
    CSI_ : float
        Overall score on how confident the nodes were during the ensemble step.
        High score if most edges are close to weight 1 or 0. A high score is
        indicative of clear community structure.
    refuse_overall_ : np.ndarray, shape (n,)
        Score on how confident the node was during the ensemble step, computed
        as its degree in the weighted ensemble graph. Only set if refuse_score is True.
    refuse_community_ : np.ndarray, shape (n,)
        Score that indicates how tight a node is attatched to its community.
        Low score indicates weak attatchment. Only set if refuse_score is True.
    probs_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distribution over labels.
    labels_row_, labels_col_ : np.ndarray
        Labels of rows and columns, for bipartite graphs.
    probs_row_, probs_col_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distributions over labels for rows and columns (for bipartite graphs).
    aggregate_ : sparse.csr_matrix
        Aggregate adjacency matrix or biadjacency matrix between clusters.

    Notes
    -----
    The ECG edge weight function is defined as:
        min_weight + ( 1 - min_weight ) x (#votes_in_ensemble) / ens_size
    Edges outside the 2-core are assigned 'min_weight'.

    Example
    --------
    >>> g = sn.data.karate_club()
    >>> part = ECG().fit_predict(g)

    Reference
    ---------
    Valérie Poulin and François Théberge, "Ensemble clustering for graphs: comparisons and applications",
    Appl Netw Sci 4, 51 (2019). https://doi.org/10.1007/s41109-019-0162-z
    """

    def __init__(
        self,
        ens_size: int = 16,
        min_weight: float = 0.05,
        final: str = "leiden",
        resolution: float = 1.0,
        sort_clusters: bool = True,
        return_probs: bool = False,
        refuse_score: bool = False,
        random_state=None,
        rng=None,
        return_aggregate: bool = False,
    ):
        super(ECG, self).__init__(
            sort_clusters=sort_clusters,
            return_probs=return_probs,
            return_aggregate=return_aggregate,
        )
        if ens_size <= 0 or not float(ens_size).is_integer():
            raise ValueError(f"ens_size must be a positive integer. Got {ens_size}")
        self.ens_size = ens_size
        if min_weight < 0:
            raise ValueError(f"min_weight must be non-negative. Got {min_weight}")
        self.min_weight = min_weight
        if final not in ["louvain", "leiden"]:
            raise ValueError(f"final must be one of 'louvain' or 'leiden'. Got {final}")
        self.final = final
        if resolution < 0:
            raise ValueError(f"resolution must be non-negative. Got {resolution}")
        self.resolution = resolution
        self.refuse_score = refuse_score
        if rng is not None:
            self.rng = rng
        self.rng = np.random.default_rng(random_state)

    def fit(self, g):
        g = sp.csr_matrix(g)
        g = check_format(g)
        # Stage one, compute weights
        self.weights = g.copy().astype("float64")
        partitions = np.empty((self.ens_size, g.shape[0], ), dtype="int32")
        for i in range(self.ens_size):
            louvain = Louvain(
                resolution=self.resolution,
                n_aggregations=0,
                shuffle_nodes=True,
                random_state=self.rng.choice(2**32),
            )
            partitions[i, :] = louvain.fit_predict(g)
        _ecg_weights(
            self.weights.indptr, self.weights.indices, self.weights.data, partitions.transpose()
        )
        self.weights.data = self.weights.data / self.ens_size
        self.weights.data = self.min_weight + (1 - self.min_weight) * self.weights.data
        
        # Force min_weight outside 2-core
        core = get_core_decomposition(g)
        _min_weight_outside_core(
            self.weights.indptr, self.weights.data, core, self.min_weight
        )

        # Stage two, cluster weighted graph
        if self.final == "louvain":
            clusterer = Louvain(
                resolution=self.resolution,
                shuffle_nodes=True,
                sort_clusters=self.sort_clusters,
                random_state=self.rng.choice(2**32),
            )
        else:
            clusterer = Leiden(
                resolution=self.resolution,
                shuffle_nodes=True,
                sort_clusters=self.sort_clusters,
                random_state=self.rng.choice(2**32),
            )

        self.labels_ = clusterer.fit_predict(self.weights)
        self.CSI_ = 1 - np.mean(np.minimum(self.weights.data, 1 - self.weights.data))
        self._secondary_outputs(g)

        if self.refuse_score:
            self.refuse_overall_, self.refuse_community_ = _compute_refuse_scores(
                self.weights.indptr,
                self.weights.indices,
                self.weights.data,
                self.labels_,
            )

        return self
