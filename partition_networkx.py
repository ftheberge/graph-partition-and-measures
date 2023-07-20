# coding=utf-8
import numpy as np
import networkx

## Graph-aware measures (igraph version)
def gam(self, u, v, method="rand", adjusted=True):
    """
    Compute one of 11 graph-aware measures to compare graph partitions.
    
    Parameters
    ----------
    self: object of type 'networkx.classes.graph.Graph'
      Graph on which the partitions are defined.

    u: iterable of sets of nodes in 'self' where each set of node is a community, or a dictionary of node:community

    v: iterable of sets of nodes in 'self' where each set of node is a community, or a dictionary of node:community

    method: str
      one of 'rand', 'jaccard', 'mn', 'gmn', 'min' or 'max'

    adjusted: bool
      if True, return adjusted measure (preferred). All measures can be adjusted except 'jaccard'.

    Returns
    -------
    A graph-aware similarity measure between vertex partitions u and v.
    
    Examples
    --------
    >>> import networkx as nx
    >>> import GAM_networkx
    >>> ## two complete graphs connectedby a path
    >>> g = nx.barbell_graph(10,3)
    >>> ## Girvan-Newman returns a sequence of partitions
    >>> gn = list(nx.algorithms.community.girvan_newman(g))
    >>> ## compare the partitions with 2 or 3 parts
    >>> g.GAM(gn[0], gn[1], method='rand', adjusted=True)
    
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
    bu = np.array([d1[e[0]] == d1[e[1]] for e in self.edges()]) 
    bv = np.array([d2[e[0]] == d2[e[1]] for e in self.edges()]) 
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

networkx.classes.graph.Graph.gam = gam

from networkx.algorithms.core import core_number
import community
from collections import namedtuple

def community_ecg(self, weight='weight', ens_size = 16, min_weight = 0.05, resolution=1.0):
    """
    Stable ensemble-based graph clustering;
    the ensemble consists of single-level randomized Louvain; 
    each member of the ensemble gets a "vote" to determine if the edges 
    are intra-community or not;
    the votes are aggregated into ECG edge-weights in range [0,1]; 
    a final (full depth) Louvain is run using those edge weights;
    
    Parameters
    ----------
    self: graph of type 'networkx.classes.graph.Graph'
      Graph to define the partition on.
    weight : str, optional
      the key in graph to use as weight. Default to 'weight'
    ens_size: int, optional
      the size of the ensemble of single-level Louvain
    min_weight: float in range [0,1], optional
      the ECG edge weight for edges with zero votes from the ensemble
    resolution: positive float, optional
      resolution parameter; larger values favors smaller communities

    Returns
    -------
    an object of type 'partition_networkx.Partition' with:
    
    Partition.partition:
        The final partition as a dictionary on the vertices
    Partition.W
      The ECG edge weights s a dictionary on the edges
    Partition.CSI
      The community strength index (float)

    Notes
    -----
    The ECG edge weight function is defined as:
      
      min_weight + ( 1 - min_weight ) x (#votes_in_ensemble) / ens_size
      
    Edges outside the 2-core are assigned 'min_weight'.
    
    Examples
    --------
    >>> g = nx.karate_club_graph()
    >>> P = community.ecg(g)
    >>> part = P.partition
    >>> print(P.CSI)
    
    Reference
    ---------
    Valérie Poulin and François Théberge, "Ensemble clustering for graphs: comparisons and applications", Appl Netw Sci 4, 51 (2019). 
    https://doi.org/10.1007/s41109-019-0162-z
    """
    W = {k:0 for k in self.edges()}
    ## Ensemble of level-1 Louvain 
    for i in range(ens_size):
        d = community.generate_dendrogram(self, weight=weight, randomize=True)
        l = community.partition_at_level(d,0)
        for e in self.edges():
            W[e] += int(l[e[0]] == l[e[1]])
    ## vertex core numbers
    core = core_number(self)
    ## set edge weights
    for e in self.edges():
        m = min(core[e[0]],core[e[1]])
        if m > 1:
            W[e] = min_weight + (1-min_weight)*W[e]/ens_size
        else:
            W[e] = min_weight

    networkx.set_edge_attributes(self, W, 'W')
    part = community.best_partition(self, weight='W', resolution=resolution)
    P = namedtuple('Partition', ['partition', 'W', 'CSI'])
    w = list(W.values())
    CSI = 1-2*np.sum([min(1-i,i) for i in w])/len(w)
    p = P(part,W,CSI)
    return p

community.ecg = community_ecg

