# Graph Partition and Measures

Python3 code implementing 11 graph-aware measures (gam) for comparing graph partitions as well as a stable ensemble-based graph partition algorithm (ecg).
This code is pip installable for both igraph and networkx:

* PyPI (igraph): https://pypi.org/project/partition-igraph/
* PyPI (networkx): https://pypi.org/project/partition-networkx/

## Graph aware measures (gam)

The measures are respectively:
* 'rand': the RAND index
* 'jaccard': the Jaccard index
* 'mn': pairwise similarity normalized with the mean function
* 'gmn': pairwise similarity normalized with the geometric mean function
* 'min': pairwise similarity normalized with the minimum function
* 'max': pairwise similarity normalized with the maximum function

Each measure can be adjusted (recommended) or not, except for 'jaccard'.
Details can be found in: 

* V. Poulin and F. Theberge, "Comparing Graph Clusterings: Set partition measures vs. Graph-aware measures," in IEEE Transactions on Pattern Analysis and Machine Intelligence, https://doi.org/10.1109/TPAMI.2020.3009862 and https://ieeexplore.ieee.org/document/9142444

* Pre-print: https://arxiv.org/abs/1806.11494

## Ensemble clustering for graphs (ecg)

This is a good, stable graph partitioning algorithm previously released on its own for igraph in:
https://github.com/ftheberge/Ensemble-Clustering-for-Graphs

The added networkx version relies on the python-louvain package: https://pypi.org/project/python-louvain/

```
pip install python-louvain
```

will install the 'community' library.

Details for ecg can be found in: 

* Valérie Poulin and François Théberge, Ensemble Clustering for Graphs. in: Aiello L., Cherifi C., Cherifi H., Lambiotte R., Lió P., Rocha L. (eds) Complex Networks and Their Applications VII. COMPLEX NETWORKS 2018. Studies in Computational Intelligence, vol 812. Springer (2019), https://doi.org/10.1007/978-3-030-05411-3_19 or https://link.springer.com/chapter/10.1007/978-3-030-05411-3_19 

* Pre-print: https://arxiv.org/abs/1809.05578

* V. Poulin and F. Théberge, Ensemble clustering for graphs: comparisons and applications, Network Science (2019) 4:51 https://doi.org/10.1007/s41109-019-0162-z or https://rdcu.be/bLn9i

* Pre-print: https://arxiv.org/abs/1903.08012

# Included files

* partition_igraph.py: implements the graph-aware measures (gam) and ensemble clustering (ecg) for igraph objects
* partition_networkx.py: implements the graph-aware measures (gam) and ensemble clustering(ecg) for networkx objects
* partition.ipynb: a Jupyter notebook illustrating the use of gam and ecg with igraph and networkx.

# Example with networkx

First, we need to import the supplied Python file partition_networkx.

```python
import networkx as nx
import community
import partition_networkx
import numpy as np
```

Next, let's build a graph with communities (dense subgraphs):

```python
# Graph generation with 10 communities of size 100
commSize = 100
numComm = 10
G = nx.generators.planted_partition_graph(l=numComm, k=commSize, p_in=0.1, p_out=0.02)
## store groud truth communities as 'iterables of sets of vertices'
true_comm = [set(list(range(commSize*i, commSize*(i+1)))) for i in range(numComm)]
```

run Louvain and ecg:

```python
ml = community.best_partition(G)
ec = community.ecg(G, ens_size=32)
```

We show a few examples of measures we can compute with gam:

```python
# for 'gam' partition are either iterables of sets of vertices or 'dict'
print("Adjusted Graph-Aware Rand Index for Louvain:",G.gam(true_comm, ml))
print("Adjusted Graph-Aware Rand Index for ecg:",G.gam(true_comm, ec.partition))

print("\nJaccard Graph-Aware for Louvain:",G.gam(true_comm, ml, method="jaccard",adjusted=False))
print("Jaccard Graph-Aware for ecg:",G.gam(true_comm, ec.partition, method="jaccard",adjusted=False))
```

Next, we compare with some non graph-aware measure (the adjusted Rand index); note that a different format is required for this function, so we build a dictionary for the partitions.

```python
## adjusted RAND index requires iterables over the vertices:
from sklearn.metrics import adjusted_rand_score as ARI
tc = {val:idx for idx,part in enumerate(true_comm) for val in part}

## compute ARI
print("Adjusted non-Graph-Aware Rand Index for Louvain:",ARI(list(tc.values()), list(ml.values())))
print("Adjusted non-Graph-Aware Rand Index for ecg:",ARI(list(tc.values()), list(ec.partition.values())))
```

# Example with igraph

We need to import the supplied Python file partition_igraph.

```pyhon
import igraph as ig
import partition_igraph
```

Next, let's build an igraph version of the previous networkx graph:

```python
## previous graph 'G' in igraph format:
g = ig.Graph(directed=False)
g.add_vertices(G.nodes())
g.add_edges(G.edges())
```

run Louvain and ecg:

```python
ml = g.community_multilevel()
ec = g.community_ecg(ens_size=32)
```

Finally, we show a few examples of measures we can compute with gam:

```python
## for 'gam' partition are either 'igraph.clustering.VertexClustering' or 'dict'
print('Adjusted Graph-Aware Rand Index for Louvain:',g.gam(ml,tc))
print('Adjusted Graph-Aware Rand Index for ECG:',g.gam(ec,tc))
print('\nJaccard Graph-Aware for Louvain:',g.gam(ml,tc,method="jaccard",adjusted=False))
print('Jaccard Graph-Aware for ECG:',g.gam(ec,tc,method="jaccard",adjusted=False))
```


