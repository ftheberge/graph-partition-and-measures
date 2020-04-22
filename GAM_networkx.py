import numpy as np
import networkx

## Graph-aware measures (igraph version)
def GAM(self, u, v, method="rand", adjusted=True):
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
    A graph-aware similarity measure between partitions u and v.
    
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

networkx.classes.graph.Graph.GAM = GAM


