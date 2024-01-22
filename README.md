[![bioRxiv](https://img.shields.io/badge/bioRxiv-202023.09.05-b31b1b.svg?style=flat-square)](https://www.biorxiv.org/content/10.1101/2023.09.05.556135v1)

# geneclusters

Here we implement a graph partitioning algorithm to summarize redundant gene-pathway information as a limited number of non-redundant gene-pathway groups. In this context, we show that the Kernighan-Lin (K/L) algorithm performs similarly to METIS and produces more interpretable clusters compared to other algorithms:

![Main figure](main.png)

*Left: Average loss (total cut size; see Methods) associated with applying each algorithm (spectral clustering (SC), METIS, Kernighan-Lin (K/L), spectral bisection (SB), or random permutation) to G (with 379 vertices; see Methods) over 1000 initiations (SC, random permutation) or 5x10e5 initiations (METIS, K/L). The SB implementation is deterministic and was run only once. Error bars indicate the standard deviation. Right: Unweighted adjacency matrix for G sorted by labels assigned by indicated algorithm. Red indicates the presence of an edge between two vertices. For each algorithm, labels corresponding to the best initiation (lowest loss) over 1000 initiations (SC, random permutation) or 5x10e5 initiations (METIS, K/L) are shown.* 

In our [paper](https://www.biorxiv.org/content/10.1101/2023.09.05.556135v1) we show how applications of K/L clustering to a biological graph help us summarize an omics dataset in simple non-redundant terms:

![Main figure](main2.png)


## Usage
see `examples.ipynb` notebook for K/L, METIS, spectral clustering, and spectral bisection implementations, benchmarking, and visualization 

```python
# list all genesets
gene_sets = gseapy.get_library_name()
print(gene_sets)
```

['ARCHS4_Cell-lines', 'ARCHS4_IDG_Coexp', 'ARCHS4_Kinases_Coexp',...]

```python
# run with internet
# list all genesets
gene_sets = gseapy.get_library_name()
```

```python
# assign the clusters
mat = get_gene_pathway_matrix(path)
frame, loss_temp = get_kernighan_lin_clusters(path=None, threshold=30, C=0, KL_modified=True, random_labels=True, unweighted=True, seed=5, no_progress=False, mat=mat)
frame.columns = ['cluster', 'description', 'is_gene']
frame.head()
```

	cluster	description	is_gene
5	0.0	ABAT	True
6	1.0	ABHD12	True
8	1.0	ABHD6	True
9	1.0	ACAA1	True
11	0.0	ACAD8	True

```python
# plot graph
s=10000
graph, pos, cur_labels, unique_clusters, colors, layout = get_layout(frame, mat.T, s, 15)
out_path = './'
plot_graph(layout, pos, graph, cur_labels, unique_clusters, colors, out_path)
```

## Notes
- It is worth trying a range of algorithms, as the best-performing one might differ based on the specific graph of interest.
- In [our case](https://www.biorxiv.org/content/10.1101/2023.09.05.556135v1), K/L and METIS perform very similarly, but we end up using the K/L algorithm because we found it to perform better on larger graphs.
- KL_modified: we add a slight modification 

## Some info on the K/L heuristic
[[CMU]](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kernlin.pdf)
[[Wikipedia]](https://en.wikipedia.org/wiki/Kernighan%E2%80%93Lin_algorithm)
![Main figure](main3.png)

## Citation
If you use this code in your work, please cite using the following BibTeX entry:
```
@article{vonMaydell2023,
  doi = {10.1101/2023.09.05.556135},
  url = {https://doi.org/10.1101/2023.09.05.556135},
  year = {2023},
  month = sep,
  publisher = {Cold Spring Harbor Laboratory},
  author = {Djuna von Maydell and Shannon Wright and Julia Maeve Bonner and Ping-Chieh Pao and Gloria Suella Menchaca and Gwyneth Welch and Carles A. Boix and Hansruedi Mathys and Guillaume Leclerc and Noelle Leary and George Samaan and Manolis Kellis and Li-Huei Tsai},
  title = {A single-cell atlas of {ABCA}7 loss-of-function reveals lipid disruptions,  mitochondrial dysfunction and {DNA} damage in neurons}
}
```
## Installation 
`git clone https://github.com/djunamay/geneclusters.git`
and follow the `examples.ipynb`

## Questions?
Please email us: djuna@mit.edu


## More examples:


1. load dependencies


```python
import numpy as np
import gseapy
from scipy.sparse import csr_matrix
from tqdm.contrib.concurrent import process_map
from functools import partial
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
import seaborn as sns

from geneclusters.geneclusters import get_scores, get_kernighan_lin_clusters, get_gene_pathway_matrix, get_full_matrix_from_bipartite
```

2. Download pathway database(s)


```python
# run with internet
gseapy.get_library_name() # to show library names
x = gseapy.get_library('HumanCyc_2016')
np.save('HumanCyc_2016.npy', x)
```

3. Iterate over multiple initiations


```python
help(get_kernighan_lin_clusters)
```

    Help on function get_kernighan_lin_clusters in module geneclusters.cluster_genes_new:
    
    get_kernighan_lin_clusters(path, threshold, C, KL_modified=True, random_labels=True, unweighted=True, seed=5)
        returns pandas dataframe annotating each gene and pathway to a cluster, based on pathway-gene dictionary and args
        Args:
            path str
                path to pathway-gene dictionary as ndarray
            threshold int
                if random_labels is True, gives number of genes per cluster. Min = 1, Max = total number of genes and pathways
                if random_labels is False, gives number of desired clusters. Min = 1, Max = total number of genes and pathways
            C float
                probability of false negative pathway-gene association (0<=c<= 1)
            KL_modified bool
                whether to run the modified KL algorithm 
            random_labels
                whether to randomize initiating labels or assign based on node centrality
            unweighted bool
                whether to consider weights when computing the shortest path between nodes, only considiered if random_labels is False
    



```python
path = './examples/HumanCyc_2016.npy'
repeats = 500

thresh = np.repeat(30,repeats)
seed = np.arange(repeats)
C = 0
KL_modified = True
random_labels = True
unweighted = True

results = process_map(partial(get_scores,path,C,KL_modified,random_labels,unweighted), seed, thresh)

```


      0%|          | 0/500 [00:00<?, ?it/s]


    3it [00:06,  2.06s/it, tot_impr=1444, last_impr=0, loss=410] ] 
    3it [00:06,  2.17s/it, tot_impr=1499, last_impr=0, loss=361]   
    3it [00:06,  2.19s/it, tot_impr=1455, last_impr=0, loss=388] 
    3it [00:06,  2.19s/it, tot_impr=1441, last_impr=0, loss=410] 
    3it [00:06,  2.21s/it, tot_impr=1443, last_impr=0, loss=395] 
    3it [00:06,  2.21s/it, tot_impr=1467, last_impr=0, loss=390]  
    4it [00:06,  1.67s/it, tot_impr=1414, last_impr=0, loss=425]
    4it [00:06,  1.67s/it, tot_impr=1435, last_impr=0, loss=423]
    4it [00:06,  1.69s/it, tot_impr=1454, last_impr=0, loss=393]
    ...

4. Select labels from the best loss


```python
loss = np.hstack([x[1] for x in results])
labs = np.vstack([x[0] for x in results])
labels = labs[np.argmin(loss)]
```


```python
# assign the clusters
frame, loss_temp = get_kernighan_lin_clusters(path, 30, C, KL_modified, random_labels, unweighted, seed=seed[np.argmin(loss)])
frame.columns = ['cluster', 'description', 'is_gene']

```

    6it [00:06,  1.15s/it, tot_impr=1509, last_impr=0, loss=349]   



```python
frame[frame['cluster']==23]['description']
```




    11                                                 ACAD8
    13                                                ACADSB
    90                                               ALDH6A1
    135                                                  AUH
    147                                                BCAT1
    148                                                BCAT2
    149                                               BCKDHA
    150                                               BCKDHB
    243                                                  DBT
    261                                                  DLD
    277                                                ECHS1
    426                                                HADHA
    434                                               HIBADH
    435                                                HIBCH
    442                                              HMGCLL1
    461                                             HSD17B10
    509                                                  IVD
    543                                                MCCC1
    544                                                MCCC2
    624                                                PDHA1
    625                                                PDHA2
    796                                                  SDS
    797                                                 SDSL
    972    pyruvate decarboxylation to acetyl CoA Homo sa...
    974     2-oxobutanoate degradation Homo sapiens PWY-5130
    975         threonine degradation Homo sapiens PWY66-428
    976        leucine degradation Homo sapiens LEU-DEG2-PWY
    977      isoleucine degradation Homo sapiens ILEUDEG-PWY
    978           valine degradation Homo sapiens VALDEG-PWY
    Name: description, dtype: object


    

