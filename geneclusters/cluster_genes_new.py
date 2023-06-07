import numpy as np
import pandas as pd
from nltk import flatten
from tqdm import tqdm
import numpy.ma as ma
from ipdb import set_trace
import numba as nb

def get_gene_pathway_matrix(path_to_dict):
    '''
    returns pandas dataframe of pathways x genes indicating which pathway-gene pairs are key-value pairs in the input dictionary 
    Args:
        path_to_dict
            ndarray dictionary mapping pathway names to genes (e.g. as downloaded from GSEA)
    Returns:
        matrix ndarray
            gene x pathway matrix of edge weights
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

def get_full_matrix_from_bipartite(matrix):
    m1 = np.concatenate((np.zeros(shape=(matrix.shape[0],matrix.shape[0])), matrix), axis = 1)
    m2 = np.concatenate((matrix.T, (np.zeros(shape=(matrix.shape[1],matrix.shape[1])))), axis = 1)
    return np.concatenate((m2, m1), axis = 0)

@nb.njit()
def invert_weights(matrix):
    matrix_out = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i,j]
            if val!=0:
                matrix_out[i,j] = 1/val
            elif val==0:
                matrix_out[i,j] = val
    return matrix_out

def create_nonrandom_labeling(matrix, centrality, unweighted):
    index_genes = (np.sum(matrix, axis = 1)>centrality)
    full_matrix = get_full_matrix_from_bipartite(matrix)
    if not unweighted:
        full_matrix = invert_weights(full_matrix)
    graph = csr_matrix(full_matrix)
    dist_matrix = shortest_path(csgraph=graph,directed=False, indices=np.where(index_genes)[0], return_predecessors=False, unweighted = unweighted)
    proba = distance_to_probability(dist_matrix)
    assignment = assign_cluster_based_on_proba(proba)
    return assignment

def distance_to_probability(dist_matrix):
    inv_df = 1/(dist_matrix+10e-10)
    df = inv_df/np.sum(inv_df, axis = 0)
    return df

def assign_cluster_based_on_proba(probas):
    assignment = []
    for node in range(probas.shape[1]):
        np.random.seed(5)
        x = np.random.choice(range(probas.shape[0]), size = 1, p = probas[:,node])[0]
        assignment.append(x)
    return assignment

def create_random_labeling(matrix, threshold):
    '''
    returns ndarray grouping pathways and genes into clusters based on threshold
    Args:
        matrix ndarray
            gene x pathway matrix
        threshold int
            number of genes per cluster. Min = 1, Max = total number of genes and pathways
    Returns:
        labeling 1D ndarray
            random cluster partition labels of length N-genes + N-pathways
    '''
    N = np.sum(matrix.shape)
    num_clusters = np.ceil(N / threshold)
    new_threshold = N / num_clusters
    labeling = (np.arange(N) / new_threshold).astype('int')
    np.random.shuffle(labeling)
    return labeling

@nb.njit()
def compute_costs(i, j, c, matrix):
    '''
    Compute cost between node i and j
    Args:
        i int
            index of first node (in labeling vector)
        j int
            index of second node (in labeling vector)
        c float 
            probability of false negative pathway-gene association (0<=c<= 1)
        matrix ndarray
            gene x pathway matrix
    Returns:
        cost float
            edge weight between node i and j
    '''
    x, y = matrix.shape
    if (i<x) & (j>=x):
        return max((matrix[i,j-x], c))
    if (j<x) & (i>=x):
        return max((matrix[j,i-x], c))
    if ((j<x) & (i<x)) | ((j>=x) & (i>=x)):
        return 0

@nb.njit()
def get_cross_costs(labeling, partition1_indices, partition2_indices, matrix, c):
    '''
    Compute pairwise costs of nodes in partition A and nodes in partition B
    Args:
        labeling 1D ndarray
            output vector from create_random_labeling()
        partition1 int
            cluster 1 label
        partition2 int
            cluster 2 label
        matrix ndarray
            gene x pathway matrix
        c float 
            probability of false negative pathway-gene association (0<=c<= 1)
    Returns:
        cost-matrix ndarray
            len(partition1) x len(partition2) matrix, where entries represent edge weights for pairwise nodes in partition1 and partition2
        
    ''' 

    L1 = len(partition1_indices)
    L2 = len(partition2_indices)
    cross_costs = np.empty(shape=(L1, L2))
    for i in range(L1):
        for j in range(L2):
            cross_costs[i, j] = compute_costs(partition1_indices[i], partition2_indices[j], c, matrix)
    return cross_costs

@nb.njit()
def compute_internal_cost(partition_indices, labeling, c, matrix, Ic):
    '''
    Compute internal cost for each node in partition A and save it to corresponding index in the Ic vector
    Args:
        partition_indices 1D ndarray (int) 
            index values for partition of interest in labeling vector
        labeling 1D ndarray
            output vector from create_random_labeling()
        c float 
            probability of false negative pathway-gene association (0<=c<= 1)
        matrix ndarray
            gene x pathway matrix
        Ic 1D ndarray
            empty vector of length = len(labeling)        
    '''
    for i in partition_indices:
        for j in partition_indices:
            if i!=j:
                Ic[i] += compute_costs(i, j, c, matrix)
            else:
                continue
@nb.njit()                
def compute_external_cost(partition1_indices, partition2_indices, cross_costs, Ec):
    '''
    Compute external costs for each node in partitions A and B and save it to corresponding index in the Ec vector
    Args:
        partition_indices 1D ndarray (int) 
            index values for partitions of interest in labeling vector
        labeling 1D ndarray
            output vector from create_random_labeling()
        c float 
            probability of false negative pathway-gene association (0<=c<= 1)
        matrix ndarray
            gene x pathway matrix
        Ec 1D ndarray
            empty vector of length = len(labeling)        
    '''
    Ec[partition1_indices] = np.sum(cross_costs, axis = 1)
    Ec[partition2_indices] = np.sum(cross_costs, axis = 0)
    
@nb.njit()
def compute_cost_metrics(labeling, matrix, partition1_indices, partition2_indices, c):
    '''
    Compute the cost metrics for KL clustering
    Args:
        labeling 1D ndarray
            output vector from create_random_labeling()
        matrix ndarray
            gene x pathway matrix
        partition1 int
            cluster 1 label
        partition2 int
            cluster 2 label
        c float 
            probability of false negative pathway-gene association (0<=c<= 1)
    Returns:
        - pairwise costs between nodes in partition1 and partition 2
        - per-node externval vs internal cost (D)
    '''
    cross_costs = get_cross_costs(labeling, partition1_indices, partition2_indices, matrix, c)
    Ic = np.zeros_like(labeling)
    compute_internal_cost(partition1_indices, labeling, c, matrix, Ic)
    compute_internal_cost(partition2_indices, labeling, c, matrix, Ic)

    Ec = np.zeros_like(labeling)
    compute_external_cost(partition1_indices, partition2_indices, cross_costs, Ec)

    D = Ec-Ic
    return cross_costs, D

@nb.njit()
def add_outer(cross_costs, D, partition1_indices, partition2_indices):
    out = np.zeros_like(cross_costs)
    A = D[partition1_indices]
    B = D[partition2_indices]
    for i in range(len(A)):
        for j in range(len(B)):
            out[i,j] = A[i]+B[j]
    return out
            
@nb.njit()
def discard_done_swaps(all_improvements, done_a, done_b):
    for a in done_a:
        all_improvements[a, :] = -np.inf
    for b in done_b:
        all_improvements[:, b] = -np.inf

@nb.njit()
def kernighan_lin_step(labeling, matrix, partition1, partition2, c, KL_modified):
    '''
    Reassign labels between two partitions based on kernighan-lin algorithm
    Args:
        labeling 1D ndarray
            output vector from create_random_labeling()
        matrix ndarray
            gene x pathway matrix
        partition1 int
            cluster 1 label
        partition2 int
            cluster 2 label
        c float 
            probability of false negative pathway-gene association (0<=c<= 1)
    '''
    A = np.where(labeling == partition1)[0]
    B = np.where(labeling == partition2)[0]
    if len(A)<=len(B):
        L = A
    else:
        L = B
    
    a_out = np.zeros_like(L)
    b_out = np.zeros_like(L)
    g_out = np.zeros_like(L)
    iteration = len(L)

    labeling_mask = np.zeros_like(labeling)
    labeling_temp = labeling.copy()

    
    if KL_modified:
        done_i = []
        done_j = []
    
    for it in range(iteration):

        cross_costs, D = compute_cost_metrics(labeling_temp, matrix, A, B, c)
        pairwise_d_sums = add_outer(cross_costs, D, A, B)
        g = pairwise_d_sums-2*cross_costs
        
        if KL_modified and it!=0:
            start = g.shape[0]-it
            end = g.shape[0]-it+1
            g[start:end, :] = -np.inf
            g[:, start:end] = -np.inf
            
        x, y = g.shape
        g_max_temp = np.argmax(g)
        i = g_max_temp // y
        j = g_max_temp % y
        index1 = A[i]

        index2 = B[j]
        
        a_out[it] = index1
        b_out[it] = index2
        g_out[it] = g[i,j]   
        if KL_modified:
            done_i.append(i)
            done_j.append(j)

            A = A[A!=index1]
            A = np.append(A, index2)
            B = B[B!=index2]
            B = np.append(B, index1)
        else:
            A = A[A!=index1]
            B = B[B!=index2]
        
    cumulative_sum = np.cumsum(g_out)
    k = np.argmax(cumulative_sum)
    gmax = cumulative_sum[k]
    if gmax > 0:
        for i in range(k+1):
            ra = a_out[i]
            rb = b_out[i]
            labeling[ra], labeling[rb] = labeling[rb], labeling[ra]
        return gmax
    else:
        return 0
    
def full_kl_step(labeling, matrix, c, KL_modified):
    '''
    Apply kernighan-lin algorithm to all partition pairs
        labeling 1D ndarray
            output vector from create_random_labeling()
        matrix ndarray
            gene x pathway matrix
        c float 
            probability of false negative pathway-gene association (0<=c<= 1)
    '''
    num_clusters = len(set(labeling))
    np.random.seed(5)
    order = np.random.permutation(num_clusters ** 2)
    impr = 0
    for o in order:
        cluster_1, cluster_2 = o // num_clusters, o % num_clusters
        impr+=kernighan_lin_step(labeling, matrix, cluster_1, cluster_2, c, KL_modified)
    return impr
    

def evaluate_cut(matrix, labeling, c):
    '''
    Compute loss based on specified partitioning.
    Args:
        labeling 1D ndarray
            output vector from create_random_labeling()
        matrix ndarray
            gene x pathway matrix
        c float 
            probability of false negative pathway-gene association (0<=c<= 1)
    '''
    value = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if labeling[i] != labeling[j + matrix.shape[0]]:
                if matrix[i, j]:
                    value += 1
                else:
                    value += c
    return value

def run_KL(labeling, matrix, c, KL_modified):
    '''
    Run kernighan-lin algorithm to cluster gene-pathway matrix into equally-sized partitions
    Args:
        labeling 1D ndarray
            output vector from create_random_labeling()
        matrix ndarray
            gene x pathway matrix
        c float 
            probability of false negative pathway-gene association (0<=c<= 1)
    '''
    tot = 0
    with tqdm() as p:
        while True:
            impr = full_kl_step(labeling, matrix, c, KL_modified)
            tot += impr
            p.set_postfix({
                    'tot_impr': tot,
                    'last_impr': impr,
                    'loss': evaluate_cut(matrix, labeling, c)
            })
            p.update()
            if impr==0:
                break
                
def get_kernighan_lin_clusters(path, threshold, C, KL_modified=True):
    '''
    returns pandas dataframe annotating each gene and pathway to a cluster, based on pathway-gene dictionary and args
    Args:
        path str
            path to pathway-gene dictionary as ndarray
        threshold int
            number of genes per cluster. Min = 1, Max = total number of genes and pathways
        C float
    '''
    mat = get_gene_pathway_matrix(path)
    pathway_names = mat.index
    gene_names = mat.columns
    matrix = np.ascontiguousarray(mat.values.T)
    labeling = create_random_labeling(matrix, threshold)
    run_KL(labeling, matrix, 0, KL_modified)
    frame = pd.DataFrame(labeling)
    frame['description'] = np.concatenate([gene_names, pathway_names])
    frame['is_gene'] = np.arange(frame.shape[0]) < matrix.shape[0]
    return frame