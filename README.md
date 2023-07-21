# Graph Partition and Measures

Python3 code implementing 11 graph-aware measures (**gam**) for comparing graph partitions as well as a stable ensemble-based graph partition algorithm (**ECG**).
This code is pip installable for both **igraph** and **networkx**:

* PyPI (igraph): https://pypi.org/project/partition-igraph/
* PyPI (networkx): https://pypi.org/project/partition-networkx/

Illustrative examples can be found in the following supplied notebooks:
* [example_with_igraph](https://github.com/ftheberge/graph-partition-and-measures/blob/master/example_with_igraph.ipynb)
* [example_with_networkx](https://github.com/ftheberge/graph-partition-and-measures/blob/master/example_with_networkx.ipynb)
  
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

* V. Poulin and F. Theberge, "Comparing Graph Clusterings: Set partition measures vs. Graph-aware measures," in IEEE Transactions on Pattern Analysis and Machine Intelligence, https://doi.org/10.1109/TPAMI.2020.3009862. Pre-print: https://arxiv.org/abs/1806.11494

## Ensemble clustering for graphs (ECG)

This is a good, stable graph partitioning algorithm. Description and applications of ECG can be found in: 

* Valérie Poulin and François Théberge, Ensemble clustering for graphs: comparisons and applications, Network Science (2019) 4:51 https://doi.org/10.1007/s41109-019-0162-z or https://rdcu.be/bLn9i. Pre-print: https://arxiv.org/abs/1903.08012
* Valérie Poulin and François Théberge, Ensemble Clustering for Graphs. in: Aiello L., Cherifi C., Cherifi H., Lambiotte R., Lió P., Rocha L. (eds) Complex Networks and Their Applications VII. COMPLEX NETWORKS 2018. Studies in Computational Intelligence, vol 812. Springer (2019), https://doi.org/10.1007/978-3-030-05411-3_19. Pre-print: https://arxiv.org/abs/1809.05578

## ECG Extras

Beside providing a good, stable graph clustering method, ECG can be useful for a few other tasks:

* We can define some **refuse to cluster scores** to rank the nodes in decreasing order, from the most likely to be an outlier (i.e. not part of a community) to the least likely. This can be useful for tasks such as **outlier detection**, but also to improve the **robustness** of the clustering results by leaving out nodes that are not stongly memeber of any community.
* Another use for the derived ECG edge weights is to obtain better **layouts** for community graphs.
* ECG also returns a Community strength indidcator (CSI), where values close to 1 are indicative of strong communities in the graph.

Those *extra* features are illustrated in the supplied notebook: [ECG_extras](https://github.com/ftheberge/graph-partition-and-measures/blob/master/ECG_extras.ipynb),
as well as in this [wiki](https://github.com/ftheberge/graph-partition-and-measures/wiki/ECG-Extras).


