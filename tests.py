import numpy as np
import pandas as pd
from assertpy import assert_that
from geneclusters.cluster_genes_new import *

'''
Run tests
'''

def test_get_gene_pathway_matrix():
    """
    Testing function that converts ndarray dictionary of pathway-genes to matrix
    """
    path = './examples/test_dict.npy'
    my_dict = np.load(path, allow_pickle=True).item()
    test_mat = get_gene_pathway_matrix(path)
    counts = np.unique(sum(list(my_dict.values()), []), return_counts=True)[1]
    assert_that(np.unique(np.sum(test_mat)==counts)[0]).is_true()

def test_create_random_labeling():
    """
    Testing function that returns initial random clustering on pathways and genes
    """
    path = './examples/test_dict.npy'

    mat = get_gene_pathway_matrix(path)
    matrix = np.ascontiguousarray(mat.values.T)
    N = np.sum(matrix.shape)
    out = np.empty(shape = (N-1, N), dtype = int)

    # get random labels for allowed thresholds
    for x in range(1, N):
        out[x-1] = create_random_labeling(matrix, x)

    # check expected number of clusters
    for i in range(out.shape[0]):
        uniques = np.unique(out[i], return_counts = True)
        nclust = np.ceil(N/(i+1))
        assert_that(len(uniques[0])==nclust).is_true()
        # check expected number of genes per cluster
        if (N/nclust)%1 >0:
            # if N is not divisble by nclust, two distinct cluster sizes should exist
            assert_that(len(np.unique(uniques[1]))==2).is_true()
        else:
            # if N is divisble by nclust, a single cluster size should exist
            assert_that(len(np.unique(uniques[1]))==1).is_true()
            

def test_compute_internal_cost():
    """
    Test function that computes internal cost for all nodes in a given partition
    """
    path = './examples/test_dict.npy'
    mat = get_gene_pathway_matrix(path)
    pathway_names = mat.index
    gene_names = mat.columns
    matrix = np.ascontiguousarray(mat.values.T)
    
    labeling = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    A = np.where(labeling == 0)[0]
    B = np.where(labeling == 1)[0]
    cross_costs = get_cross_costs(labeling, A, B, matrix, 0)
    Ic = np.zeros(len(labeling), dtype = int)
    compute_internal_cost(A, labeling, 0, matrix, Ic)
    compute_internal_cost(B, labeling, 0, matrix, Ic)
    
    ground_truth_Ic = np.array([0, 1, 0, 1, 1, 0, 2, 1])
    assert_that(np.array_equal(Ic, ground_truth_Ic)).is_true()
    
def test_compute_external_cost():
    """
    Test function that computes external cost for all nodes in a given partition
    """
    path = './examples/test_dict.npy'
    mat = get_gene_pathway_matrix(path)
    pathway_names = mat.index
    gene_names = mat.columns
    matrix = np.ascontiguousarray(mat.values.T)
    
    labeling = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    A = np.where(labeling == 0)[0]
    B = np.where(labeling == 1)[0]
    cross_costs = get_cross_costs(labeling, A, B, matrix, 0)

    Ec = np.zeros(len(labeling), dtype = int)
    compute_external_cost(A, B, cross_costs, Ec)
    ground_truth_Ec = np.array([2, 1, 1, 1, 0, 3, 1, 1])
    assert_that(np.array_equal(Ec, ground_truth_Ec)).is_true()

def test_compute_cost_metrics():
    '''
    Test function that computes all cost metrics
    '''
    path = './examples/test_dict.npy'
    mat = get_gene_pathway_matrix(path)
    pathway_names = mat.index
    gene_names = mat.columns
    matrix = np.ascontiguousarray(mat.values.T)  
    labeling = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    A = np.where(labeling == 0)[0]
    B = np.where(labeling == 1)[0]

    cross_costs, D = compute_cost_metrics(labeling, matrix, A, B, 0)
    cross_costs_ground = np.array([[0,0,1,1],[0,0,1,0],[0,0,1,0],[1,0,0,0]])
    partition1_indices_ground = np.array([0,1,2,7])
    partition2_indices_ground = np.array([3,4,5,6])
    D_ground = np.array([2,0,1,0,-1,3,-1,0])
    
    assert_that(np.array_equal(cross_costs, cross_costs_ground)).is_true()
    assert_that(np.array_equal(A, partition1_indices_ground)).is_true()
    assert_that(np.array_equal(B, partition2_indices_ground)).is_true()
    assert_that(np.array_equal(D, D_ground)).is_true()
    
def get_kernighan_lin_clusters():
    '''
    Test function that returns clusters
    '''
    matrix = np.array([[1,0],[1,0],[0,1],[0,1]])
    labeling = create_random_labeling(matrix, 3)
    run_KL(labeling, matrix, 0)
    gene_names = np.array(['G1','G2','G3','G4'])
    pathway_names = np.array(['P1','P2'])
    frame = pd.DataFrame(labeling)
    frame['description'] = np.concatenate([gene_names, pathway_names])
    frame['is_gene'] = np.arange(frame.shape[0]) < matrix.shape[0]

    temp = frame[frame[0]==1]['description']
    clust1 = set(['G1','G2','P1'])
    assert_that(np.unique([x in clust1 for x in temp])[0]).is_true()

    temp = frame[frame[0]==0]['description']
    clust1 = set(['G3','G4','P2'])
    assert_that(np.unique([x in clust1 for x in temp])[0]).is_true()