import numpy as np
import gseapy

from .geneclusters import get_kernighan_lin_clusters, evaluate_cut, run_METIS
from sklearn.cluster import SpectralClustering

from tqdm import tqdm
import networkx as nx

def save_gset(name):
    """
    Saves a gene set library from gseapy to a NumPy binary file.

    This function retrieves a gene set library with the given name using gseapy,
    and then saves it as a NumPy binary file ('.npy') on the local filesystem.
    The file is named after the gene set library and appended with '.npy'.

    Parameters:
    name (str): The name of the gene set library to be retrieved and saved.

    Returns:
    None: This function does not return any value.
    """
    
    x = gseapy.get_library(name)
    np.save(name+'.npy', x)
    
def run_many_KL(N, size, C, KL_modified, random_labels, unweighted, matrix):
    """
    Executes the Kernighan-Lin (KL) clustering algorithm multiple times.

    This function runs the KL clustering algorithm 'N' times on a given matrix.
    Each iteration uses a different seed for random number generation. It saves the 
    cluster labels and loss values from each run into separate NumPy binary files.
    The function is designed to evaluate the performance of the KL algorithm under
    different initial conditions.

    Parameters:
    N (int): The number of times to run the KL algorithm.
    size (int): number of genes per cluster. Min = 1, Max = total number of genes and pathways
    C (float): probability of false negative pathway-gene association (0<=c<= 1)    KL_modified (bool): Specifies whether to use a modified version of the KL algorithm.
    random_labels (bool): If True, initializes with random labeling.
    unweighted (bool): Treats the matrix as unweighted if True.
    matrix (np.ndarray): The gene-pathway matrix on which the KL algorithm is applied.

    Returns:
    None: This function does not return any value but saves output to files.
    """
    loss_kl = np.empty(N)
    labs_kl = np.empty((N,np.sum(matrix.shape)))
    
    for i in tqdm(range(N)):
        frame, loss_temp = get_kernighan_lin_clusters(None, size, C, KL_modified, random_labels, unweighted, seed=i, no_progress=True, mat=matrix)
        frame.columns = ['cluster', 'description', 'is_gene']
        labs_kl[i] = np.array(frame['cluster'])
        loss_kl[i] = loss_temp
    
    np.save('./kl_labs.npy', labs_kl)
    np.save('./kl_loss.npy', loss_kl)
    
def make_symmetric(mat_sub):
    """
    Constructs a symmetric matrix from a non-square submatrix.

    This function takes a non-square matrix 'mat_sub' and creates a larger,
    square, symmetric matrix. The original matrix is placed in the bottom-left
    block of the new matrix, and its transpose in the top-right block.

    Parameters:
    mat_sub (numpy.ndarray): A non-square matrix to be converted into a symmetric matrix.

    Returns:
    numpy.ndarray: The symmetric square matrix formed from 'mat_sub' and its transpose.
    """
    full_mat = np.zeros((np.sum(mat_sub.shape), np.sum(mat_sub.shape)))
    full_mat[mat_sub.shape[1]:][:,:mat_sub.shape[1]] = mat_sub
    full_mat[:mat_sub.shape[1]][:,mat_sub.shape[1]:]=mat_sub.T
    return full_mat

def run_many_METIS(full_mat, mat_sub, nparts, N):
    """
    Repeatedly partitions a graph using the METIS algorithm and records the results.

    This function constructs a graph from the 'full_mat' matrix and then applies the
    METIS algorithm 'N' times to partition this graph into 'nparts' parts. Each run
    of the METIS algorithm is evaluated for partition quality (cut value), and the 
    results are saved to disk.

    Parameters:
    full_mat (numpy.ndarray): A square matrix representing the adjacency matrix of the graph.
    mat_sub (numpy.ndarray or pandas.DataFrame): A matrix used in the evaluation of the partition cut.
    nparts (int): The number of partitions to divide the graph into.
    N (int): The number of times the METIS algorithm is to be run.

    Returns:
    None: This function does not return any value but saves output to files.
    """
    g = nx.from_numpy_array(full_mat)
    labs_met = np.empty((N,np.sum(mat_sub.shape)))
    loss_met=np.empty(N)
    
    for i in tqdm(range(N)): 
        sc, cut = run_METIS(g, mat_sub, nparts, i)
        loss_met[i] = cut
        labs_met[i] = sc

    np.save('./met_labs.npy', labs_met)
    np.save('./met_loss.npy', loss_met)
    
    
def run_many_SPECTRAL(full_mat, mat_sub, nclust, N):
    """
    Repeatedly applies spectral clustering to a matrix and records the results.

    This function applies spectral clustering to 'full_mat' 'N' times, each time 
    partitioning the matrix into 'nclust' clusters. The quality of each clustering 
    is evaluated using a submatrix 'mat_sub', and the results are saved to disk.

    Parameters:
    full_mat (numpy.ndarray): A square matrix representing the data for clustering.
    mat_sub (numpy.ndarray or pandas.DataFrame): A matrix used in the evaluation of the clustering quality.
    nclust (int): The number of clusters for the spectral clustering.
    N (int): The number of times the spectral clustering is to be run.

    Returns:
    None: This function does not return any value but saves output to files.
    """
    loss_sc = np.empty(N)
    labs_sc = np.empty((N,np.sum(mat_sub.shape)))
    
    for i in tqdm(range(N)): 
        sc = SpectralClustering(nclust, assign_labels='kmeans', random_state=i, affinity='precomputed').fit(full_mat)
        loss_sc[i]=evaluate_cut(np.ascontiguousarray(mat_sub.values.T), sc.labels_, 0)
        labs_sc[i] = sc.labels_
    np.save('./spectral_labs.npy', labs_sc)
    np.save('./spectral_loss.npy', loss_sc)
    
def run_many_RAND(N, labs_init, mat_sub):
    """
    Evaluates random cluster labelings against a submatrix for partition quality.

    This function generates 'N' random permutations of an initial label set 'labs_init'.
    For each permutation, it computes a loss value representing the quality of the
    partitioning using the submatrix 'mat_sub'. The random labels and their 
    corresponding loss values are then saved to disk.

    Parameters:
    N (int): The number of random permutations to generate and evaluate.
    labs_init (numpy.ndarray): The initial set of labels from which random permutations are generated.
    mat_sub (numpy.ndarray or pandas.DataFrame): A matrix used in the evaluation of the clustering quality.

    Returns:
    None: This function does not return any value but saves output to files.
    """
    
    loss_rand = np.empty(N)
    labs_rand = np.empty((N,np.sum(mat_sub.shape)))

    for i in tqdm(range(N)):
        L = np.random.permutation(labs_init)
        labs_rand[i] = L
        loss_rand[i] = evaluate_cut(np.ascontiguousarray(mat_sub.values.T), L, 0)
    np.save('./rand_labs.npy', labs_rand)
    np.save('./rand_loss.npy', loss_rand)
