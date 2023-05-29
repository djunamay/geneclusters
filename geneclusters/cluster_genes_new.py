import numpy as np
import pandas as pd
from nltk import flatten

def get_gene_pathway_matrix(path_to_dict):
    '''
    returns pandas dataframe of pathways x genes indicating which pathway-gene pairs are key-value pairs in the input dictionary 
    Args:
        path_to_dict
            ndarray dictionary mapping pathway names to genes (e.g. as downloaded from GSEA)
    '''
    paths = np.load(path_to_dict, allow_pickle=True).item()
    genes = np.unique(flatten(list(paths.values())))
    path_names = list(paths.keys())   
    output = np.empty(shape=[len(path_names), len(genes)], dtype = int)
    sets = [set(paths[x]) for x in path_names]
    df = pd.DataFrame([[x in y for x in genes] for y in sets])+0
    df.columns = genes
    df.index = path_names
    return df

def create_random_labeling(matrix, threshold):
    '''
    returns ndarray grouping pathways and genes into clusters based on threshold
    Args:
        matrix ndarray
            gene x pathway matrix
        threshold int
            number of genes per cluster. Min = 1, Max = total number of genes and pathways
    '''
    N = np.sum(matrix.shape)
    num_clusters = np.ceil(N / threshold)
    new_threshold = N / num_clusters
    labeling = (np.arange(N) / new_threshold).astype('int')
    np.random.shuffle(labeling)
    return labeling

def compute_costs(i, j, c, matrix):
    '''
    Compute cost between node a and b
    '''
    x, y = matrix.shape
    if (i<x) & (j>=x):
        return np.max((matrix[i,j-x], c))
    if (j<x) & (i>=x):
        return np.max((matrix[j,i-x], c))
    if ((j<x) & (i<x)) | ((j>=x) & (i>=x)):
        return 0

def get_cross_costs(labeling, partition1, partition2, matrix, c):
    '''
    Compute pairwise costs of nodes in partition A and nodes in partition B
    ''' 
    partition1_indices = np.arange(len(labeling))[labeling==partition1]
    partition2_indices = np.arange(len(labeling))[labeling==partition2]
    L1 = len(partition1_indices)
    L2 = len(partition2_indices)
    cross_costs = np.empty(shape=(L1, L2))
    for i in range(L1):
        for j in range(L2):
            cross_costs[i, j] = compute_costs(partition1_indices[i], partition2_indices[j], c, matrix)
    return cross_costs, partition1_indices, partition2_indices

def compute_internal_cost(partition_indices, labeling, c, matrix, Ic):
    '''
    Compute internal cost for each node in partition A
    '''
    for i in partition_indices:
        for j in partition_indices:
            if i!=j:
                Ic[i] += compute_costs(i, j, c, matrix)
            else:
                continue
                
def compute_external_cost(partition1_indices, partition2_indices, cross_costs, Ec):
    '''
    Compute external costs for each node in partitions A and B
    '''
    Ec[partition1_indices] = np.sum(cross_costs, axis = 1)
    Ec[partition2_indices] = np.sum(cross_costs, axis = 0)
    
### Adapt tests to the new functions
### Finish annotating these functions
### Is the external cost computed between A and B or between A and other?