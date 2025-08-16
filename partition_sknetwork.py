import numpy as np
import scipy.sparse as sp
import sknetwork as sn
import numba


@numba.njit
def _internal_edge(g_indptr, g_indices, partition):
    is_internal_edge = np.empty(len(g_indices), dtype="bool")
    for n1 in range(len(g_indptr)-1):
        for data_offset, n2 in enumerate(g_indices[g_indptr[n1]:g_indptr[n1+1]]):
            is_internal_edge[g_indptr[n1]+data_offset] = partition[n1] == partition[n2]
    return is_internal_edge


def gam(g, u, v, method="rand", adjusted=True):
    """
    Compute one of 11 graph-aware measures to compare graph partitions.
    
    Parameters
    ----------
    g: adjaceny matrix of the graph on which the partitions are defined.

    u: Partiton of nodes. A numpy array of length n where u[i] = j means node i is in part j. Parts must be labeled 0-n_parts.

    v: Partiton of nodes. A numpy array of length n where u[i] = j means node i is in part j. Parts must be labeled 0-n_parts.

    method: 'str'
      one of 'rand', 'jaccard', 'mn', 'gmn', 'min' or 'max'

    adjusted: 'bool'
      if True, return adjusted measure (preferred). All measures can be adjusted except 'jaccard'.
      
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
    Valérie Poulin and François Théberge, "Comparing Graph Clusterings: Set Partition Measures vs. Graph-aware Measures",
    IEEE Transactions on Pattern Analysis and Machine Intelligence 43, 6 (2021) https://doi.org/10.1109/TPAMI.2020.3009862
    """
    g = sp.triu(g).tocsr()
    bu = _internal_edge(g.indptr, g.indices, u)
    bv = _internal_edge(g.indptr, g.indices, v)
    su = np.sum(bu)
    sv = np.sum(bv)
    suv = np.sum(bu*bv)
    m = len(bu)
    ## all adjusted measures
    if adjusted:
        if method=="jaccard":
            raise ValueError("no adjusted jaccard measure, set adjusted=False")
        elif method=="rand" or method=="mn":
            return((suv-su*sv/m)/(np.average([su,sv])-su*sv/m))  
        elif method=="gmn":
            return((suv-su*sv/m)/(np.sqrt(su*sv)-su*sv/m))            
        elif method=="min":
             return((suv-su*sv/m)/(np.min([su,sv])-su*sv/m))  
        elif method=="max":
             return((suv-su*sv/m)/(np.max([su,sv])-su*sv/m))              
        else:
            raise ValueError(f"Method not found. Should be one of ['jaccard', 'rand', 'gmn', 'min', 'max']. Got {method}")
    ## all non-adjusted measures
    else:
        if method=="jaccard":
            union_b = np.sum((bu+bv)>0)
            return(suv/union_b)
        elif method=="rand":
            return(1-(su+sv)/m+2*suv/m)
        elif method=="mn":
            return(suv/np.average([su,sv]))
        elif method=="gmn":
            return(suv/np.sqrt(su*sv))
        elif method=="min":
            return(suv/np.min([su,sv]))
        elif method=="max":
            return(suv/np.max([su,sv]))
        else:
            raise ValueError(f"Method not found. Should be one of ['jaccard', 'rand', 'gmn', 'min', 'max']. Got {method}")


@numba.njit
def _ecg_weights(g_indptr, g_indices, g_data, partitions):
    for n1 in range(len(g_indptr)-1):
        for data_offset, n2 in enumerate(g_indices[g_indptr[n1]:g_indptr[n1+1]]):
            g_data[g_indptr[n1]+data_offset] = np.sum(partitions[n1, :] == partitions[n2, :])
    return g_data


class ECG:
    """
    Stable ensemble-based graph clustering;
    the ensemble consists of single-level randomized Louvain; 
    each member of the ensemble gets a "vote" to determine if the edges 
    are intra-community or not;
    the votes are aggregated into ECG edge-weights in range [0,1]; 
    a final (full depth) Leiden is run using those edge weights;
    
    Examples
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
            ens_size:int=16,
            min_weight:float=0.05,
            final:str="leiden",
            resolution:float=1.0,
            refuse_score:bool=False,
            seed = None,
            rng = None
    ):
        if ens_size <= 0 or not ens_size.is_integer():
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
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()


    def fit(self, g):
        # Stage one, compute weights
        self.weights = g.copy().astype("float64")
        partitions = np.empty((g.shape[0], self.ens_size), dtype="int32")
        for i in range(self.ens_size):
            louvain = sn.clustering.Louvain(resolution=self.resolution, n_aggregations=0, shuffle_nodes=True, random_state=self.rng.choice(100000))
            partitions[:, i] = louvain.fit_predict(g)
        _ecg_weights(self.weights.indptr, self.weights.indices, self.weights.data, partitions)
        self.weights.data = self.weights.data/self.ens_size
        self.weights.data = self.min_weight + (1-self.min_weight)*self.weights.data
        # Force min_weight outside 2-core
        core = sn.topology.get_core_decomposition(g)
        for i, core_num in enumerate(core):
            if core_num < 2:
                self.weights.data[self.weights.indptr[i]:self.weights.indptr[i+1]] = self.min_weight

        # Stage two, cluster weighted graph
        if self.final == "louvain":
            clusterer = sn.clustering.Louvain(resolution=self.resolution, shuffle_nodes=True, random_state=self.rng.choice(100000))
        else:
            clusterer = sn.clustering.Leiden(resolution=self.resolution, shuffle_nodes=True, random_state=self.rng.choice(100000))
    
        self.labels = clusterer.fit_predict(self.weights)


    def fit_predict(self, g):
        self.fit(g)
        return self.labels
