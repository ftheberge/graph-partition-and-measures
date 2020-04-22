# Graph_Aware_Measures
Python code implementing 11 graph-aware measures for comparing graph partitions.

Compute one of 11 graph-aware similarity measures to compare graph partitions.
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
