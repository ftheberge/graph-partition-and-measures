# Graph_Aware_Measures
Python3 code implementing 11 graph-aware measures for comparing graph partitions.

The measures are respectively:
* 'rand': the RAND index
* 'jaccard': the Jaccard index
* 'mn': pairwise similarity normalized with the mean function
* 'gmn': pairwise similarity normalized with the geometric mean function
* 'min': pairwise similarity normalized with the minimum function
* 'max': pairwise similarity normalized with the maximum function

Each measure can be adjusted (recommended) or not, except for 'jaccard'.
Details can be found in: 

Valérie Poulin and François Théberge, "Comparing Graph Clusterings: Set partition measures vs. Graph-aware measures", https://arxiv.org/abs/1806.11494.

# Included files

* GAM_igraph.py: implements the graph-aware measures (GAM) for igraph objects
* GAM_networkx.py: implements the graph-aware measures (GAM) for networkx objects
* ecg.py: ensemble clustering for graphs, a good stable igraph-based partitionning algorithm, see https://github.com/ftheberge/Ensemble-Clustering-for-Graphs
* GAM.ipynb: a Jupyter notebook illustrating the use of GAM with igraph and networkx.

# Example with igraph

First, we need to import the supplied Python file GAM_igraph; we also import the ensemble clustering code from ecg.py.

```pyhon
import igraph as ig
import ecg
import GAM_igraph
import numpy as np
```

Next, let's build a graph with communities (dense subgraphs):

```python
## build graph with n=1000 vertices and n_comm=10 communities
## p_in : edge probability within a community
## p_out: edge probability between communities
n_comm = 10
p_in = .1
p_out = 0.025
P = np.full((n_comm,n_comm),p_out)
np.fill_diagonal(P,p_in)
## ground truth communities is stored in 'class' attribute
g = ig.Graph.Preference(n=1000, type_dist=[1.0/n_comm]*n_comm, pref_matrix=P.tolist(), attribute='class')
gt = ig.clustering.VertexClustering.FromAttribute(g,'class')
```

Finally, we show a few examples of measures we can compute with GAM:

```python
## run Louvain and ECG:
ml = g.community_multilevel()
ec = g.community_ecg(ens_size=32)

## compare a few Graph-Aware Measures (GAM)
print('Adjusted Graph-Aware Rand Index for Louvain:',g.GAM(ml,gt))
print('Adjusted Graph-Aware Rand Index for ECG:',g.GAM(ec,gt))
print('\nJaccard Graph-Aware for Louvain:',g.GAM(ml,gt,method="jaccard",adjusted=False))
print('Jaccard Graph-Aware for ECG:',g.GAM(ec,gt,method="jaccard",adjusted=False))
```

# Example with networkx

First, we need to import the supplied Python file GAM_networkx.

```pyhon
import networkx as nx
import GAM_networkx
import numpy as np
```

Next, let's build a graph with communities (dense subgraphs):

```python
# Graph generation with 10 communities of size 10
commSize = 10
numComm = 10
G = nx.generators.planted_partition_graph(l=numComm,k=commSize,p_in=0.3,p_out=0.025)
true_comm = [set(list(range(commSize*i, commSize*(i+1)))) for i in range(numComm)]
```

We show a few examples of measures we can compute with GAM:

```python
# Not many partitionning algorithms in networkx ...
# Here using Girvan_Newman with the right number of communities:
algo_comm = list(nx.algorithms.community.girvan_newman(G))[numComm-2]
print("Adjusted Graph-Aware Rand Index for Girvan-Newman:",G.GAM(true_comm, algo_comm))
print("Jaccard Graph-Aware for Girvan-Newman:",G.GAM(true_comm, algo_comm, method="jaccard",adjusted=False))
```

Next, we compare with some non graph-aware measure (the adjusted Rand index); note that a different format is required for this function, so we build a dictionary for the partitions.

```python
## adjusted RAND index requires a different format (label for each vertex)
from sklearn.metrics import adjusted_rand_score as ARI

## build dictionaries for the communities
tc = {v:idx for idx, part in enumerate(true_comm) for v in part}
ac = {v:idx for idx, part in enumerate(algo_comm) for v in part}

## compute ARI
print("Adjusted non-Graph-Aware Rand Index for Girvan-Newman:",ARI(list(tc.values()), list(ac.values())))

## dictionaries can also be used with GAM
print("Adjusted Graph-Aware Rand Index for Girvan-Newman:",G.GAM(tc, ac))
```

