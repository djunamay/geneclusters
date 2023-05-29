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
    Ic = np.zeros(len(labeling), dtype = int)
    compute_costs(0, labeling, matrix, Ic, internal=True)
    compute_costs(1, labeling, matrix, Ic, internal=True)
    
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
    Ec = np.zeros(len(labeling), dtype = int)
    compute_costs(0, labeling, matrix, Ec, internal = False)
    compute_costs(1, labeling, matrix, Ec, internal = False)
    ground_truth_Ec = np.array([2, 1, 1, 1, 0, 3, 1, 1])
    assert_that(np.array_equal(Ec, ground_truth_Ec)).is_true()
