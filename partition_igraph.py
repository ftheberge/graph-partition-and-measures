# coding=utf-8
import numpy as np
import igraph

## Graph-aware measures (igraph version)
def gam(self, u, v, method="rand", adjusted=True):
    """
    Compute one of 11 graph-aware measures to compare graph partitions.
    
    Parameters
    ----------
    self: Graph of type 'igraph.Graph' on which the partitions are defined.

    u: Partition of type 'igraph.clustering.VertexClustering' on 'self', or a dictionary of node:community.

    v: Partition of type 'igraph.clustering.VertexClustering' on 'self', or a dictionary of node:community.

    method: 'str'
      one of 'rand', 'jaccard', 'mn', 'gmn', 'min' or 'max'

    adjusted: 'bool'
      if True, return adjusted measure (preferred). All measures can be adjusted except 'jaccard'.
      
    Returns
    -------
    A graph-aware similarity measure between vertex partitions u and v.
    
    Examples
    --------
    >>> g = ig.Graph.Famous('Zachary')
    >>> part1 = g.community_multilevel()
    >>> part2 = g.community_label_propagation()
    >>> print(g.GAM(part1, part2))
    
     Reference
    ---------
    Valérie Poulin and François Théberge, "Comparing Graph Clusterings: Set partition measures vs. Graph-aware measures", https://arxiv.org/abs/1806.11494.
    """
    if(type(u) is dict):
        d1 = u
    else:
        d1 = {val:idx for idx,part in enumerate(u) for val in part}
    if(type(v) is dict):
        d2 = v
    else:
        d2 = {val:idx for idx,part in enumerate(v) for val in part}    
    bu = np.array([(d1[x.tuple[0]]==d1[x.tuple[1]]) for x in self.es])
    bv = np.array([(d2[x.tuple[0]]==d2[x.tuple[1]]) for x in self.es])
    su = np.sum(bu)
    sv = np.sum(bv)
    suv = np.sum(bu*bv)
    m = len(bu)
    ## all adjusted measures
    if adjusted:
        if method=="jaccard":
            print("no adjusted jaccard measure, set adjusted=False")
            return None
        if method=="rand" or method=="mn":
            return((suv-su*sv/m)/(np.average([su,sv])-su*sv/m))  
        if method=="gmn":
            return((suv-su*sv/m)/(np.sqrt(su*sv)-su*sv/m))            
        if method=="min":
             return((suv-su*sv/m)/(np.min([su,sv])-su*sv/m))  
        if method=="max":
             return((suv-su*sv/m)/(np.max([su,sv])-su*sv/m))              
        else:
            print('Wrong method!')

    ## all non-adjusted measures
    else:
        if method=="jaccard":
            union_b = sum((bu+bv)>0)
            return(suv/union_b)
        if method=="rand":
            return(1-(su+sv)/m+2*suv/m)
        if method=="mn":
            return(suv/np.average([su,sv]))
        if method=="gmn":
            return(suv/np.sqrt(su*sv))
        if method=="min":
            return(suv/np.min([su,sv]))
        if method=="max":
            return(suv/np.max([su,sv]))
        else:
            print('Wrong method!')
        
    return None

igraph.Graph.gam = gam

import igraph
import numpy as np

def community_ecg(self, weights=None, ens_size = 16, min_weight = 0.05, final='louvain'):
    """
    Stable ensemble-based graph clustering;
    the ensemble consists of single-level randomized Louvain; 
    each member of the ensemble gets a "vote" to determine if the edges 
    are intra-community or not;
    the votes are aggregated into ECG edge-weights in range [0,1]; 
    a final (full depth) Louvain is run using those edge weights;
    
    Parameters
    ----------
    self: graph of type 'igraph.Graph'
      Graph to define the partition on.
    weights: list of double, optional 
      the edge weights
    ens_size: int 
      the size of the ensemble of single-level Louvain
    min_weight: double in range [0,1] 
      the ECG edge weight for edges with zero votes from the ensemble
    final: 'louvain' or 'leiden'
      the algorithm to run on the final re-weighted graph
      
    Returns
    -------
    partition
      The final partition, of type 'igraph.clustering.VertexClustering'
    partition.W
      The ECG edge weights
    partition.CSI
      The community strength index
    partition.original_modularity
      The modularity with respect to the original edge weights

    Notes
    -----
    The ECG edge weight function is defined as:
      
      min_weight + ( 1 - min_weight ) x (#votes_in_ensemble) / ens_size
      
    Edges outside the 2-core are assigned 'min_weight'.
    
    Examples
    --------
    >>> g = igraph.Graph.Famous('Zachary')
    >>> part = g.community_ecg(ens_size=25, min_weight = .1)
    >>> print(part.CSI)
    
    Reference
    ---------
    Valérie Poulin and François Théberge, "Ensemble clustering for graphs: comparisons and applications", Appl Netw Sci 4, 51 (2019). 
    https://doi.org/10.1007/s41109-019-0162-z
    """
    W = [0]*self.ecount()
    ## Ensemble of level-1 Louvain
    for i in range(ens_size):
        p = np.random.permutation(self.vcount()).tolist()
        g = self.permute_vertices(p)
        l1 = g.community_multilevel(weights=weights, return_levels=True)[0].membership
        b = [l1[p[x.tuple[0]]]==l1[p[x.tuple[1]]] for x in self.es]
        W = [W[i]+b[i] for i in range(len(W))]
    W = [min_weight + (1-min_weight)*W[i]/ens_size for i in range(len(W))]
    ## Force min_weight outside 2-core
    core = self.shell_index()
    ecore = [min(core[x.tuple[0]],core[x.tuple[1]]) for x in self.es]
    w = [W[i] if ecore[i]>1 else min_weight for i in range(len(ecore))]
    if final=='leiden':
        part = self.community_leiden(weights=w, objective_function='modularity')
    else:
        part = self.community_multilevel(weights=w)
    part.W = w
    part.CSI = 1-2*np.sum([min(1-i,i) for i in w])/len(w)
    part._modularity_params['weights'] = weights
    part.recalculate_modularity()
    return part

igraph.Graph.community_ecg = community_ecg
