import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import random 
from nltk import flatten
from functools import partial
from tqdm.contrib.concurrent import process_map

def compute_permuted_t(all_correlations_indexed, partner1_cluster_anno, partner2_cluster_anno, u, index):
    np.random.seed()
    temp = np.random.permutation(all_correlations_indexed)
    x = compute_t(partner1_cluster_anno, partner2_cluster_anno, temp, u)
    return x

def compute_t(partner1_cluster_anno, partner2_cluster_anno, all_correlations_indexed, u):
    observed_ts = list()
    in_genes = (partner1_cluster_anno==partner2_cluster_anno)
    out_genes = np.invert(in_genes)
    observed_ts.append(np.mean(np.abs(all_correlations_indexed[in_genes]))-np.mean(np.abs(all_correlations_indexed[out_genes])))
    return observed_ts

def return_correlation_matrix(path, frame):
    data = pd.read_csv(path, index_col = 0)
    data = data.loc[np.invert(np.sum(data==0, axis = 1)>(.5*data.shape[1]))]
    correlations = data.T.corr()
    n = set(frame['description'])
    ind = [x in n for x in correlations.index]
    frame_gene = frame.loc[frame['is_gene']]
    correlations = correlations.iloc[ind,ind]
    names = correlations.index
    correlations = np.array(correlations)
    return correlations, names, frame_gene

def return_permutation_inputs(correlations, names, frame_gene):
    index = np.tril(correlations, k = -1).flatten()!=0
    all_correlations = correlations.flatten()
    partner1 = np.repeat(names, correlations.shape[1])
    partner2 = np.tile(names, correlations.shape[0])
    partner1_indexed = partner1[index]
    partner2_indexed = partner2[index]
    all_correlations_indexed = all_correlations[index]
    clus_unique = np.unique(frame_gene['cluster'])
    dictionary = dict(zip(frame_gene['description'], frame_gene['cluster']))
    partner1_cluster_anno = np.array([dictionary[x] for x in partner1_indexed])
    partner2_cluster_anno = np.array([dictionary[x] for x in partner2_indexed])
    return all_correlations_indexed, partner1_cluster_anno, partner2_cluster_anno, clus_unique

def run_permutations(path, frame, Nperm):
    correlations, names, frame_gene = return_correlation_matrix(path, frame)
    all_correlations_indexed, partner1_cluster_anno, partner2_cluster_anno, clus_unique = return_permutation_inputs(correlations, names, frame_gene)
    observed_t = compute_t(partner1_cluster_anno, partner2_cluster_anno, all_correlations_indexed, clus_unique)
    func = partial(compute_permuted_t, all_correlations_indexed, partner1_cluster_anno, partner2_cluster_anno, clus_unique)
    permuted_t = process_map(func, list(range(Nperm)))
    return permuted_t, observed_t

def compute_permutation_p(permuted_t, observed_t):
    f = ECDF(np.array(permuted_t).reshape(-1))
    p = 1-f(observed_t)
    return np.round(p[0], 4)