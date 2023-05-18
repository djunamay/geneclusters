# Credit: Guillaume Leclerc & Wikipedia 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
import numba as nb
from .prepare_inputs import get_gene_pathway_matrix

@nb.njit()
def evaluate_cut(matrix, labeling, c):
    value = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if labeling[i] != labeling[j + matrix.shape[0]]:
                if matrix[i, j]:
                    value += 1
                else:
                    value += c
    return value
            
def create_random_labeling(matrix, threshold):
    '''
    returns pandas dataframe annotating each gene and pathway to a cluster, based on pathway-gene dictionary and args
    Args:
        matrix ndarray
            gene x pathway matrix
        threshold float
            clustering resolution; inversely proportional to number of clusters to uniformly partition genes and pathways into
    '''
    N = np.sum(matrix.shape)
    num_clusters = np.ceil(N / threshold)
    new_threshold = N / num_clusters
    labeling = (np.arange(N) / new_threshold).astype('int')
    np.random.shuffle(labeling)
    return labeling

@nb.njit()
def get_cost(matrix, i, j, c):
    if j < i:
        i, j = j, i
    if i >= matrix.shape[0]:
        return 0
    if j < matrix.shape[0]:
        return 0
    
    j -= matrix.shape[0]
    
    return max(matrix[i, j], c)

@nb.njit()
def cost_to_other(matrix, source, others, c):
    costs = np.zeros_like(source)
    for i, a in enumerate(source):
        for b in others:
            costs[i] += get_cost(matrix, a, b, c)
    return costs

@nb.njit()
def get_cross_costs(matrix, A, B, c):
    result = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            result[i, j] = get_cost(matrix, a, b, c)
    return result

@nb.njit()
def get_pairwise_improvements(matrix, labeling, a, b, c):
    nodes_a = np.where(labeling == a)[0]
    nodes_b = np.where(labeling == b)[0]
    internal_a = cost_to_other(matrix, nodes_a, nodes_a, c)
    internal_b = cost_to_other(matrix, nodes_b, nodes_b, c)
    ext_a_to_b = cost_to_other(matrix, nodes_a, nodes_b, c)
    ext_b_to_a = cost_to_other(matrix, nodes_b, nodes_a, c)
    D_a = ext_a_to_b - internal_a
    D_b = ext_b_to_a - internal_b
    cross_costs = get_cross_costs(matrix, nodes_a, nodes_b, c)
    all_improvements = np.zeros_like(cross_costs)
    for i in nb.prange(all_improvements.shape[0]):
        for j in range(all_improvements.shape[1]):
            all_improvements[i, j] = D_a[i] + D_b[j] - 2 * cross_costs[i , j]
    return all_improvements, nodes_a, nodes_b

@nb.njit()
def create_numba_int_list():
    l = List()
    l.append(32)
    l.pop()
    return l

@nb.njit()
def discard_done_swaps(all_improvements, done_a, done_b):
    for a in done_a:
        all_improvements[a, :] = -np.inf
    for b in done_b:
        all_improvements[:, b] = -np.inf

@nb.njit()
def kernighan_lin_step(matrix, labeling, cluster_1, cluster_2, c):
    '''
    returns 
    Args:
        matrix ndarray
            gene x pathway matrix
        labeling 1D ndarray
        cluster_1 1D ndarray
            new labels for cluster_1
        cluster_2 1D ndarray
            new labels for cluster_2
        c float (0<= c <= 1)
            probability of false negative pathway-gene association
    '''
    temp_labeling = labeling.copy()
    
    A = np.where(labeling == cluster_1)[0]
    B = np.where(labeling == cluster_2)[0]
    
    done_a = []
    done_b = []
    
    g = 0
    gs = []
    swaps = []
    
    for _ in range(min(len(A), len(B))):
        all_improvements, A, B = get_pairwise_improvements(matrix, temp_labeling, cluster_1, cluster_2, c)
        discard_done_swaps(all_improvements, done_a, done_b)
        
        ix = np.argmax(all_improvements)
        a, b = ix // all_improvements.shape[1], ix % all_improvements.shape[1]
        done_a.append(a)
        done_b.append(b)
        ra = A[a]
        rb = B[b]
        
        swaps.append((ra, rb))
        g += all_improvements[a, b]
        gs.append(g)
        temp_labeling[ra], temp_labeling[rb] = temp_labeling[rb], temp_labeling[ra]
            
    num_steps = np.argmax(np.array(gs))
    if gs[num_steps] > 0:
        for i in range(num_steps + 1):
            ra, rb = swaps[i]
            labeling[ra], labeling[rb] = labeling[rb], labeling[ra]

        return gs[num_steps]
    else:
        return 0
    
@nb.njit()
def full_kl_step(matrix, labeling, c):
    '''
    returns pandas dataframe annotating each gene and pathway to a cluster, based on pathway-gene dictionary and args
    Args:
        matrix ndarray
            gene x pathway matrix
        threshold float
            clustering resolution; inversely proportional to number of clusters to uniformly partition genes and pathways into
    '''
    num_clusters = len(set(labeling))
    order = np.random.permutation(num_clusters ** 2)
    
    impr = 0
    for o in order:
        cluster_1, cluster_2 = o // num_clusters, o % num_clusters
        impr += kernighan_lin_step(matrix, labeling, cluster_1, cluster_2, c)
    
    return impr
        
def kernighan_lin(matrix, labeling, c):
    '''
    returns pandas dataframe annotating each gene and pathway to a cluster, based on pathway-gene dictionary and args
    Args:
        matrix ndarray
            gene x pathway matrix
        threshold float
            clustering resolution; inversely proportional to number of clusters to uniformly partition genes and pathways into
    '''
    tot = 0
    with tqdm() as p:
        while True:
            impr = full_kl_step(matrix, labeling, c)
            tot += impr
            p.set_postfix({
                'tot_impr': tot,
                'last_impr': impr,
                'loss': evaluate_cut(matrix, labeling, c)
            })
            p.update()
            if impr == 0:
                break
    return tot
    
def score_for_thres(matrix, thres, c):
    labeling = create_random_labeling(matrix, thres)
    kernighan_lin(matrix, labeling, c)
    return evaluate_cut(matrix, labeling, c), labeling

def get_kernighan_lin_clusters(path, threshold, C):
    '''
    returns pandas dataframe annotating each gene and pathway to a cluster, based on pathway-gene dictionary and args
    Args:
        path str
            path to pathway-gene dictionary as ndarray
        threshold float
        C float
    '''
    mat = get_gene_pathway_matrix(path)
    pathway_names = mat.index
    gene_names = mat.columns
    matrix = np.ascontiguousarray(mat.values.T)
    results = score_for_thres(matrix, threshold, C)
    frame = pd.DataFrame(results[1])
    frame['description'] = np.concatenate([gene_names, pathway_names])
    frame['is_gene'] = np.arange(frame.shape[0]) < matrix.shape[0]
    return frame
