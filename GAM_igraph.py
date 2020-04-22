# coding=utf-8
import numpy as np
import igraph

## Graph-aware measures (igraph version)
def GAM(self, u, v, method="rand", adjusted=True):
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

igraph.Graph.GAM = GAM

