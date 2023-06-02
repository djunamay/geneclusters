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
    np.random.seed(5)
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
    #partition1_indices = np.arange(len(labeling))[labeling==partition1]
    #partition2_indices = np.arange(len(labeling))[labeling==partition2]
    L1 = len(partition1_indices)
    L2 = len(partition2_indices)
    cross_costs = np.empty(shape=(L1, L2))
    for i in range(L1):
        for j in range(L2):
            cross_costs[i, j] = compute_costs(partition1_indices[i], partition2_indices[j], c, matrix)
    return cross_costs#, partition1_indices, partition2_indices

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
    return cross_costs, D#, partition1_indices, partition2_indices, D

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
        #print('**')
        #print(A)
        #print(B)
        cross_costs, D = compute_cost_metrics(labeling_temp, matrix, A, B, c)
        pairwise_d_sums = add_outer(cross_costs, D, A, B)
        g = pairwise_d_sums-2*cross_costs
        #print(g.shape)
        if KL_modified & it!=0:
            #discard_done_swaps(g, done_i, done_j)
            g[-1, :] = -np.inf
            g[:, -1] = -np.inf
            
        x, y = g.shape
        g_max_temp = np.argmax(g)
        i = g_max_temp // y
        j = g_max_temp % y

        index1 = A[i]
        #if index1 in set(a_out):
        #    print('oops')
        #    break
        index2 = B[j]
        #if index2 in set(b_out):
        #    print('oops')
        #    break
        
        a_out[it] = index1
        b_out[it] = index2
        g_out[it] = g[i,j]   
        if KL_modified:
            done_i.append(i)
            done_j.append(j)
            #print(i)
            #print(index1)
            #print(j)
            #print(index2)
            #labeling_temp[index1], labeling_temp[index2] = labeling_temp[index2], labeling_temp[index1]
            #A = np.where(labeling_temp == partition1)[0]
            #print(A)
            #B = np.where(labeling_temp == partition2)[0]
            #print(B)
            A = A[A!=index1]
            A = np.append(A, index2)
            B = B[B!=index2]
            B = np.append(B, index1)
        else:
            A = A[A!=index1]
            B = B[B!=index2]
        
        #labeling_temp[index1] = max(labeling)+1
        #labeling_temp[index2] = max(labeling)+1
        
        #labeling_mask[index1] = 1
        #labeling_mask[index2] = 1
        #labeling_temp = ma.masked_array(labeling_temp, mask = labeling_mask)
        # I remove the best nodes from the next iteration right away (which is how the algorithm works no?)
        # G exchanges the labels of the best nodes prior to computing the next cost metric, but then excludes those nodes before computing the argmax
        #labeling_temp[index1], labeling_temp[index2] = labeling_temp[index2], labeling_temp[index1]
        
    cumulative_sum = np.cumsum(g_out)
    k = np.argmax(cumulative_sum)
    gmax = cumulative_sum[k]
    if gmax > 0:
        #set_trace()
        for i in range(k+1):
            ra = a_out[i]#.astype(int)
            rb = b_out[i]#.astype(int)
            labeling[ra], labeling[rb] = labeling[rb], labeling[ra]
        return gmax
    else:
        return 0
    
#@nb.njit()
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
    print(order)
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

#@nb.njit()
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
    print('test15')
    pathway_names = mat.index
    gene_names = mat.columns
    matrix = np.ascontiguousarray(mat.values.T)
    labeling = create_random_labeling(matrix, threshold)
    print(labeling)
    run_KL(labeling, matrix, 0, KL_modified)
    frame = pd.DataFrame(labeling)
    frame['description'] = np.concatenate([gene_names, pathway_names])
    frame['is_gene'] = np.arange(frame.shape[0]) < matrix.shape[0]
    return frame