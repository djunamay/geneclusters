# Standard Library Imports
from ipdb import set_trace

# Data Manipulation and Numerical Computing
import numpy as np
import pandas as pd
import numpy.ma as ma

# Text Processing
from nltk import flatten

# Performance and Debugging
import numba as nb
from tqdm import tqdm

# Sparse Matrices and Graph Analysis
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import networkx
import metis

# Visualization
import matplotlib.pyplot as plt

@nb.njit()
def compute_jaccard(arr1, arr2):
    """
    Compute the Jaccard similarity coefficient between two arrays.

    This function calculates the Jaccard similarity coefficient, a statistic used 
    for comparing the similarity and diversity of sample sets. The Jaccard 
    coefficient measures similarity between finite sample sets, and is defined as 
    the size of the intersection divided by the size of the union of the sample sets.

    Parameters:
    arr1 (np.ndarray): The first input array.
    arr2 (np.ndarray): The second input array.

    Returns:
    float: The Jaccard similarity coefficient between the two input arrays.
    """
    outer = arr1-arr2
    i_outer = np.sum(outer>0)
    outer_sum = np.sum(np.abs(outer))
    shared = np.sum(arr1)-i_outer
    jaccard = shared/(outer_sum+shared)
    return jaccard

@nb.njit(parallel=True)
def compute_all_jaccard(mat_array):
    """
    Compute the Jaccard similarity coefficient for each pair of rows in a 2D numpy array.

    This function applies the compute_jaccard function to each possible pair of rows 
    in the provided 2D numpy array. The result is a 2D numpy array where the element 
    at [i, j] is the Jaccard similarity coefficient between the i-th and j-th rows 
    of the input array.

    Parameters:
    mat_array (np.ndarray): A 2D numpy array whose rows are to be compared. 
                            Each row represents a set of observations.

    Returns:
    np.ndarray: A 2D numpy array of shape (N, N), where N is the number of rows in 
                mat_array. Each element [i, j] contains the Jaccard similarity 
                coefficient between the i-th and j-th rows of the input array.
    """
    N = mat_array.shape[0]
    out = np.empty(shape=(N,N))
    for i in nb.prange(N):
        for j in nb.prange(N):
            out[i,j] = compute_jaccard(mat_array[i], mat_array[j])
    return out

@nb.njit()
def get_rand_index(x, y):
    """
    Calculate the Rand index between two clusterings.

    The Rand index measures the similarity between two data clusterings. It is 
    calculated as the number of agreements between the two clusterings divided 
    by the total number of pairs. The index ranges from 0 (no agreement) to 1 
    (complete agreement).

    Parameters:
    x (array-like): Cluster labels from the first clustering. Must be the same length as y.
    y (array-like): Cluster labels from the second clustering. Must be the same length as x.

    Returns:
    float: The Rand index, a value between 0 and 1, where higher values indicate 
           greater similarity between the two clusterings.
    """
    a=0
    b=0
    c=0
    d=0
    for i in range(len(x)):
        for j in range(len(x)):
            if (x[i]==x[j]) & (y[i]==y[j]):
                a+=1
            elif (x[i]!=x[j]) & (y[i]!=y[j]):
                b+=1
            elif (x[i]!=x[j]) & (y[i]==y[j]):
                c+=1
            elif (x[i]==x[j]) & (y[i]!=y[j]):
                d+=1
    R = (a+b)/(a+b+c+d)
    return R

def get_LP(full_mat):
    """
    Compute the Laplacian matrix from a given matrix.

    This function calculates the Laplacian matrix. The Laplacian matrix is computed by 
    negating the original matrix and setting the diagonal elements to the sum of 
    the corresponding row elements in the original matrix.

    Parameters:
    full_mat (np.ndarray): A square 2D numpy array representing the original matrix. 
                           It's assumed to be a connectivity or adjacency matrix 
                           of a graph.

    Returns:
    np.ndarray: A 2D numpy array representing the Laplacian matrix of the input matrix.
    """
    LP = full_mat*-1
    np.fill_diagonal(LP, np.sum(full_mat, axis=1))
    return LP

def get_spectral_partition(labels_sp, LP, i, index):
    """
    Perform spectral partitioning on a graph using its Laplacian matrix.

    This function applies spectral graph partitioning to divide a graph into two 
    components. It computes the eigenvalues and eigenvectors of the Laplacian matrix 
    of the graph, uses the second smallest eigenvector (Fiedler vector) to partition 
    the graph, and updates the partition labels.

    Parameters:
    labels_sp (np.ndarray): An array to store the partition labels. It is modified in-place.
    LP (np.ndarray): The Laplacian matrix of the graph.
    i (int): The index of the current partition iteration.
    index (list or np.ndarray): Indices of the nodes in the graph to be partitioned.
    """
    eigvals, eigvect = np.linalg.eig(LP)
    v2 = eigvect[:,np.argsort(eigvals)[1]]
    temp = (np.argsort(v2))
    N = len(temp)
    middle = int(N/2)
    x = labels_sp[0][index]

    x[temp[middle:]]=1
    x[temp[:middle]]=-1
    labels_sp[i][index] = x    

@nb.njit(parallel=True)
def compute_jaccard_all_clust(arr1, arr2):
    """
    Compute the Jaccard similarity coefficient for all pairs of clusters between two clustering arrays.

    This function calculates the Jaccard similarity coefficients for each pair of clusters 
    identified in two different clustering label arrays. The output is a matrix where the 
    element at [i, j] is the Jaccard similarity coefficient between the i-th cluster in 
    arr1 and the j-th cluster in arr2.

    Parameters:
    arr1 (np.ndarray): An array of cluster labels from the first clustering.
    arr2 (np.ndarray): An array of cluster labels from the second clustering.

    Returns:
    np.ndarray: A 2D numpy array of shape (N, N), where N is the number of unique clusters 
                in arr1 (assuming the same number in arr2). Each element [i, j] contains the 
                Jaccard similarity coefficient between the i-th cluster in arr1 and the j-th 
                cluster in arr2.
    """
    N = len(np.unique(arr1))
    out = np.empty(shape=(N,N))
    for i in nb.prange(N):
        for j in nb.prange(N):
            i_0 = (arr1==i)#.astype(int)
            j_0 = (arr2==j)#.astype(int)
            out[i,j] = compute_jaccard(i_0, j_0)
    return out

@nb.njit(parallel=True)
def get_all_rands(rands_out, N, labs):
    """
    Calculate the Rand index for each pair of cluster label arrays from a given set.

    This function computes the Rand index, a measure of similarity between two data 
    clusterings, for each pair of clustering label arrays in the provided set. The 
    results are stored in the provided output array.

    Parameters:
    rands_out (np.ndarray): A 2D numpy array to store the calculated Rand indices. 
                            This array should be pre-allocated with the appropriate shape.
    N (int): The number of clustering label arrays in the set.
    labs (np.ndarray): A 2D numpy array where each row is a different set of cluster labels.
    """
    for i in nb.prange(N):
        for j in nb.prange(N):
            rands_out[i, j] = get_rand_index(labs[i], labs[j])

@nb.njit(parallel=True)
def get_all_rands2grps(rands_out, N, labs1, labs2):
    """
    Calculate the Rand index for each pair of cluster label arrays, where each array 
    comes from a different set of labels.

    This function computes the Rand index, a measure of similarity between two data 
    clusterings, for each combination of clustering label arrays where one label array 
    is taken from labs1 and the other from labs2. The results are stored in the provided 
    output array.

    Parameters:
    rands_out (np.ndarray): A 2D numpy array to store the calculated Rand indices. 
                            This array should be pre-allocated with the appropriate shape.
    N (int): The number of clustering label arrays in each set.
    labs1 (np.ndarray): A 2D numpy array where each row is a different set of cluster labels.
    labs2 (np.ndarray): Another 2D numpy array where each row is a different set of cluster labels.
    """
    for i in nb.prange(N):
        for j in nb.prange(N):
            rands_out[i, j] = get_rand_index(labs1[i], labs2[j])


def group(matrix, cols_mapped, rows_mapped, mapping, num_groups):
    """
    Aggregate a matrix into a smaller matrix based on group mappings.

    This function takes an input matrix and aggregates its values into a new matrix with
    fewer rows and columns based on the provided mapping. Each element in the new matrix 
    represents the sum of values from the original matrix that fall into the corresponding 
    group combination.

    Parameters:
    matrix (np.ndarray): The original matrix to be aggregated.
    cols_mapped (list or np.ndarray): Mapping for columns of the matrix into groups.
    rows_mapped (list or np.ndarray): Mapping for rows of the matrix into groups.
    mapping: Not directly used in the function. May be intended for future use or an error.
    num_groups (int): The number of groups to aggregate the matrix into.

    Returns:
    np.ndarray: A new aggregated matrix of size (num_groups, num_groups).

    Each cell (i, j) in the result represents the sum of all elements matrix[k, l] where
    rows_mapped[k] = i and cols_mapped[l] = j.
    """
    # author: Guillaume
    result = np.zeros((num_groups, num_groups))
    for i, gi in zip(range(matrix.shape[0]), rows_mapped):
        for j, gj in zip(range(matrix.shape[1]), cols_mapped):
            result[gi, gj] += matrix[i, j]
    return result

def compute_groupped_matrix(matrix, cols, rows):
    """
    Compute a grouped version of a matrix based on row and column groupings.

    This function takes an input matrix and groups it into a smaller matrix according to the 
    specified row and column groupings. It creates a mapping of unique elements from both rows 
    and columns to a set of group indices and then uses this mapping to aggregate the original 
    matrix into a grouped matrix.

    Parameters:
    matrix (np.ndarray): The original matrix to be grouped.
    cols (list or np.ndarray): Column identifiers for grouping.
    rows (list or np.ndarray): Row identifiers for grouping.

    Returns:
    tuple: A tuple containing:
        - groupped_matrix (np.ndarray): The grouped matrix with aggregated values.
        - mapping (dict): The mapping from original row/column identifiers to group indices.

    The function first identifies all unique group identifiers across both rows and columns, 
    then maps these identifiers to a set of indices. It aggregates the values of the original 
    matrix based on these mappings to create the grouped matrix.
    """
    # author: Guillaume
    all_groups = list(sorted(list(set(cols) | set(rows))))
    num_groups = len(all_groups)
    mapping = dict(zip(all_groups, range(num_groups)))
    cols_mapped = np.vectorize(mapping.get)(cols)
    rows_mapped = np.vectorize(mapping.get)(rows)
    groupped_matrix = group(matrix, cols_mapped, rows_mapped, mapping, num_groups)
    return groupped_matrix, mapping

def find_similar_clusters(groupped_matrix, threshold=0.75):
    """
    Identify similar clusters in a grouped matrix based on a similarity threshold.

    This function compares each pair of clusters in a grouped matrix to determine if they are 
    similar based on a specified threshold. Similarity is assessed by comparing the sum of 
    diagonal elements (intra-cluster relationships) to the sum of anti-diagonal elements 
    (inter-cluster relationships) for each pair of clusters.

    Parameters:
    groupped_matrix (np.ndarray): The grouped matrix where each element represents a relationship 
                                  between clusters.
    threshold (float, optional): The threshold for determining similarity between clusters. 
                                 Default is 0.75.

    Returns:
    dict: A dictionary where keys are cluster indices and values are the indices of the clusters 
          they are similar to.

    Each cluster pair (g1, g2) is considered similar if the ratio of the sum of their anti-diagonal 
    elements to the sum of their diagonal elements exceeds the specified threshold.
    """
    # author: Guillaume
    replacements = {}
    for g1 in range(len(groupped_matrix)):
        for g2 in range(g1 + 1, len(groupped_matrix)):
            diag = groupped_matrix[g1, g1] + groupped_matrix[g2, g2]
            anti_diag = groupped_matrix[g1, g2] + groupped_matrix[g2, g1]
            
            if diag != 0 and anti_diag / diag > threshold:
                replacements[g2] = g1
    return replacements

def get_representative_name_per_cluster(bipartite_mat, colnames_mat, rownames_mat, description_table, cluster):
    """
    Identify the most representative name for a given cluster based on a bipartite matrix.

    This function examines a cluster in a bipartite matrix and determines the most representative
    name (e.g., pathway name) for that cluster. The representative name is selected based on the
    highest ratio of internal to total (internal + external) interactions within the cluster.

    Parameters:
    bipartite_mat (np.ndarray): The bipartite matrix representing interactions.
    colnames_mat (list or np.ndarray): Column names of the bipartite matrix, representing one set of entities (e.g., genes).
    rownames_mat (list or np.ndarray): Row names of the bipartite matrix, representing another set of entities (e.g., pathways).
    description_table (pd.DataFrame): A DataFrame with descriptions, indicating whether an entity is a gene and its associated cluster.
    cluster (int): The cluster number for which the representative name is to be found.

    Returns:
    tuple: Returns a tuple containing the cluster identifier, the most representative name, 
           the ratio of internal to total interactions, and the sum of internal interactions.

    If there are no pathways in the given cluster, the function returns NaN values for the 
    representative name and interaction sums.
    """
    genes = set(description_table.loc[description_table['is_gene']&(description_table['cluster']==cluster)]['description'])
    paths = set(description_table.loc[np.invert(description_table['is_gene'])&(description_table['cluster']==cluster)]['description'])
    if len(paths)==0:
        return 'C.'+str(cluster), np.nan, np.nan, np.nan
    else:
        index_col = [x in genes for x in colnames_mat]
        index_row = [x in paths for x in rownames_mat]

    
        sum_internal = np.sum(bipartite_mat[index_row][:,index_col], axis=1)
        sum_external = np.sum(bipartite_mat[index_row][:,np.invert(index_col)], axis=1)
        sum_ratio = sum_internal/(sum_external+sum_internal)
        S = np.argmax(sum_ratio)
        rep_name = rownames_mat[index_row][S]

        return 'C.'+str(cluster), rep_name.split(' (')[0], sum_ratio[S], sum_internal[S]

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

@nb.njit()
def get_full_matrix_from_bipartite(matrix):
    """
    Construct a full adjacency matrix from a bipartite matrix.

    This function takes a bipartite matrix and converts it into a full adjacency matrix 
    representing the connections in the bipartite graph. The resulting matrix is composed 
    of four quadrants: two of them are zero matrices, and the other two are the original 
    matrix and its transpose.

    Parameters:
    matrix (np.ndarray): The bipartite matrix to be transformed.

    Returns:
    np.ndarray: A full adjacency matrix derived from the bipartite matrix.

    The returned matrix has a shape of (n+m, n+m), where n and m are the dimensions of 
    the original bipartite matrix. The upper right and lower left quadrants represent the 
    bipartite connections, while the other quadrants are filled with zeros.
    """
    m1 = np.concatenate((np.zeros(shape=(matrix.shape[0],matrix.shape[0])), matrix), axis = 1)
    m2 = np.concatenate((matrix.T, (np.zeros(shape=(matrix.shape[1],matrix.shape[1])))), axis = 1)
    return np.concatenate((m2, m1), axis = 0)

@nb.njit()
def invert_weights(matrix, c):
    """
    Invert the weights of a matrix with a constant adjustment.

    This function inverts the weights of a given matrix. Each element of the matrix is first 
    compared with a constant value 'c'. If the element's value is greater than 'c', it is 
    replaced by the inverse of this value. If the value is equal to or less than 'c', no 
    inversion is performed. Zero values are left unchanged.

    Parameters:
    matrix (np.ndarray): The matrix whose weights are to be inverted.
    c (float): A constant value used for comparing each element of the matrix.

    Note:
    The function modifies the matrix in place, so the original matrix values are altered.
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = max(matrix[i,j], c)
            if val!=0:
                matrix[i,j] = 1/val
            elif val==0:
                matrix[i,j] = val

def create_nonrandom_labeling(matrix, centrality, unweighted, c, seed):
    """
    Create a non-random clustering assignment based on centrality measures.

    This function creates a non-random labeling of elements in a matrix. It utilizes centrality 
    measures to determine the importance of each element and assigns clusters accordingly. The 
    function can operate in either a weighted or unweighted mode, influenced by the 'c' parameter.

    Parameters:
    matrix (np.ndarray): The matrix used for clustering.
    centrality (int): The number of central elements to consider for cluster assignment.
    unweighted (bool): Flag to determine whether to use weighted or unweighted clustering.
    c (float): A constant used in the inversion of weights; if greater than 0, unweighted is False.
    seed (int): Seed value for random number generation, ensuring reproducibility.

    Returns:
    np.ndarray: An array representing the cluster assignment for each element in the matrix.

    The process involves inverting the weights of the matrix, converting it into a full adjacency 
    matrix, and then using shortest path calculations to determine cluster assignments based on 
    the centrality measures. The function includes a step to convert distances to probabilities, 
    which are then used to assign clusters.
    """
    np.random.seed(seed)
    
    if c>0:
        unweighted=False
    if not unweighted:
        matrix_temp = matrix.copy()
        invert_weights(matrix_temp, c)
        full_matrix = get_full_matrix_from_bipartite(matrix_temp)
    if unweighted:
        full_matrix = get_full_matrix_from_bipartite(matrix)
    index_genes = np.sum(full_matrix, axis = 0)
    index_genes = np.argsort(-index_genes)[:centrality]
    graph = csr_matrix(full_matrix)
    if unweighted:
        graph = graph.ceil()
    dist_matrix = shortest_path(csgraph=graph,directed=False, indices=index_genes, return_predecessors=False, unweighted = unweighted)
    proba = distance_to_probability(dist_matrix)
    assignment = assign_cluster_based_on_proba(proba)
    return assignment

@nb.njit()
def sigmoid(x):
    """
    Compute the sigmoid function for an input array or scalar.

    The sigmoid function, also known as the logistic function, is used to map
    input values to the range [0, 1].

    Parameters:
    x (array_like or float): The input value(s) for which to compute the sigmoid.

    Returns:
    array_like or float: The result of applying the sigmoid function to the input value(s).

    The sigmoid function is defined as:
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1/(1 + np.exp(-x))

@nb.njit()
def distance_to_probability(dist_matrix):
    """
    Convert a distance matrix to a probability matrix.

    Given a distance matrix, this function converts it into a probability matrix.
    It calculates the probability of each element being connected to others based on
    the inverse of the distances.

    Parameters:
    dist_matrix (np.ndarray): A distance matrix containing pairwise distances between elements.

    Returns:
    np.ndarray: A probability matrix where each element represents the probability of
    connection between corresponding elements.

    The function calculates the probability matrix by taking the inverse of the distances
    (with a small offset to avoid division by zero) and normalizing the values to sum to 1
    along each column. The resulting matrix represents the probabilities of connections
    between elements based on their distances.
    """
    inv_df = 1/(dist_matrix+10e-10)
    df = inv_df/np.sum(inv_df, axis = 0)
    return df

def assign_cluster_based_on_proba(probas):
    assignment = []
    for node in range(probas.shape[1]):
        x = np.random.choice(range(probas.shape[0]), size = 1, p = probas[:,node])[0]
        assignment.append(x)
    return np.array(assignment)

def create_random_labeling(matrix, threshold, seed=None):
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
    np.random.seed(seed)
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
    """
    Calculate the outer sum of distances between elements of two partitions.

    Given two partitions and their respective distance matrices, this function calculates
    the outer sum of distances between elements of the two partitions.

    Parameters:
    cross_costs (np.ndarray): An empty 2D array to store the calculated outer sums.
    D (np.ndarray): The distance matrix containing pairwise distances between elements.
    partition1_indices (list): A list of indices corresponding to elements in partition 1.
    partition2_indices (list): A list of indices corresponding to elements in partition 2.

    Returns:
    np.ndarray: The `cross_costs` array filled with the calculated outer sums.

    The function calculates the outer sum by iterating over elements in both partitions and
    computing the sum of distances between each pair of elements (one from each partition).
    The results are stored in the `cross_costs` array.
    """
    out = np.zeros_like(cross_costs)
    A = D[partition1_indices]
    B = D[partition2_indices]
    for i in range(len(A)):
        for j in range(len(B)):
            out[i,j] = A[i]+B[j]
    return out

@nb.njit()
def discard_done_swaps(all_improvements, done_a, done_b):
    """
    Discard swaps that have already been performed from the list of possible improvements.

    This function is used in the Kernighan-Lin clustering algorithm to remove swaps that 
    have already been applied to improve the clustering. Swaps that have been performed 
    should not be considered again in the optimization process.

    Parameters:
    all_improvements (np.ndarray): A 2D array representing the improvements for all possible 
                                   swaps in the clustering.
    done_a (list): A list of indices corresponding to elements that have been swapped (group A).
    done_b (list): A list of indices corresponding to elements that have been swapped (group B).

    Returns:
    None

    The function operates in-place by modifying the `all_improvements` array to mark already 
    performed swaps as invalid by setting their values to negative infinity.
    """
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

@nb.njit()
def full_kl_step(labeling, matrix, c, KL_modified, seed=None):
    '''
    Apply kernighan-lin algorithm to all partition pairs
        labeling 1D ndarray
            output vector from create_random_labeling()
        matrix ndarray
            gene x pathway matrix
        c float 
            probability of false negative pathway-gene association (0<=c<= 1)
    '''
    np.random.seed(seed)
    num_clusters = len(set(labeling))
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

def run_KL(labeling, matrix, c, KL_modified, no_progress=False, seed=None):
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
    if no_progress:
        while True:
            impr = full_kl_step(labeling, matrix, c, KL_modified, seed)
            tot += impr
            if impr==0:
                break
    else:
        with tqdm() as p:
            while True:
                impr = full_kl_step(labeling, matrix, c, KL_modified, seed)
                tot += impr
                p.set_postfix({
                        'tot_impr': tot,
                        'last_impr': impr,
                        'loss': evaluate_cut(matrix, labeling, c)
                })
                p.update()
                if impr==0:
                    break

def get_kernighan_lin_clusters(path, threshold, C, KL_modified=True, random_labels=True, unweighted=True, seed=5, no_progress=False, mat=None):
    """
    Computes clusters using the Kernighan-Lin algorithm on a gene-pathway matrix.

    This function applies the Kernighan-Lin (KL) algorithm to a matrix representing
    gene-pathway relationships. It can operate on a provided matrix or construct
    the matrix from a specified path. 

    Parameters:
    path (str): Path to the file containing the gene-pathway data.
    threshold (int): number of genes per cluster. Min = 1, Max = total number of genes and pathways
    C (float): probability of false negative pathway-gene association (0<=c<= 1)
    KL_modified (bool, optional): If True, uses a modified version of the KL algorithm. Defaults to True.
    random_labels (bool, optional): If True, initializes with random labeling. Defaults to True.
    unweighted (bool, optional): If True, treats the matrix as unweighted. Defaults to True.
    seed (int, optional): Seed for random number generator. Defaults to 5.
    no_progress (bool, optional): If True, progress information is not printed. Defaults to False.
    mat (pandas.DataFrame or None, optional): Precomputed gene-pathway matrix. If None, the matrix is generated from the given path. Defaults to None.

    Returns:
    tuple: A tuple containing two elements:
        - pandas.DataFrame: A DataFrame with cluster labels, description, and a boolean indicating if it's a gene.
        - float: The cut value representing the quality of the partition.
    """
    if mat is None:
        print('test')
        mat = get_gene_pathway_matrix(path)
    pathway_names = mat.index
    gene_names = mat.columns
    matrix = np.ascontiguousarray(mat.values.T)
    if random_labels:
        labeling = create_random_labeling(matrix, threshold, seed)
    else:
        labeling = create_nonrandom_labeling(matrix, threshold, unweighted, C, seed)
    run_KL(labeling, matrix, 0, KL_modified, no_progress, seed)
    frame = pd.DataFrame(labeling)
    frame['description'] = np.concatenate([gene_names, pathway_names])
    frame['is_gene'] = np.arange(frame.shape[0]) < matrix.shape[0]
    return frame, evaluate_cut(matrix, labeling, C)

def get_scores(path, C, KL_modified, random_labels, unweighted, no_progress, mat, seed, thresh):
    """
    Perform clustering using the Kernighan-Lin method and return the clustering scores.

    This function applies the Kernighan-Lin clustering algorithm to a given matrix and 
    returns the scores or results associated with the clustering. The function is capable 
    of operating in various modes, including weighted or unweighted, and with modifications 
    to the Kernighan-Lin algorithm.

    Parameters:
    path (str): Path to the input file or a resource.
    C (float): A constant used in the clustering algorithm.
    KL_modified (bool): Flag indicating whether a modified version of the Kernighan-Lin algorithm is used.
    random_labels (bool): Flag for using random labels in the clustering process.
    unweighted (bool): Flag to indicate whether the clustering should be unweighted.
    no_progress (bool): Flag to disable progress bar display.
    mat (np.ndarray or pd.DataFrame): The matrix to be clustered.
    seed (int): Seed for random number generation for reproducibility.
    thresh (float): Threshold value used in the clustering process.

    Returns:
    tuple: A tuple containing:
        - An np.ndarray representing the clustering scores or results.
        - Additional output o2, the nature of which depends on the specifics of the 
          clustering algorithm and its implementation.

    The function interfaces with an external Kernighan-Lin clustering implementation and 
    adapts its output to the specific needs of the caller.
    """
    o1, o2 = get_kernighan_lin_clusters(path, thresh, C, KL_modified, random_labels, unweighted, seed, no_progress, mat)
    return np.array(o1[0]), o2

def run_METIS(g, mat_sub, nparts, i):
    """
    Partitions a graph using the METIS algorithm and evaluates the partition quality.

    This function applies the METIS algorithm to partition a graph 'g' into a specified
    number of parts 'nparts'. After partitioning, it evaluates the cut value of the 
    partition using the submatrix 'mat_sub'.

    Parameters:
    g (networkx.Graph or similar): The graph to be partitioned.
    mat_sub (pandas.DataFrame or numpy.ndarray): The matrix used in evaluating the partition cut.
    nparts (int): The number of parts to divide the graph into.

    Returns:
    tuple: A tuple containing two elements:
        - numpy.ndarray: An array of size equal to the number of nodes in 'g', with each 
          element indicating the partition index a node belongs to.
        - float: The cut value representing the quality of the partition.
    """
    sc = metis.part_graph(g, nparts=nparts, tpwgts=None, ubvec=None, recursive=False, seed=i)[1] 
    cut = evaluate_cut(np.ascontiguousarray(mat_sub.values.T), sc, 0)
    return sc, cut

def run_SB(nclust, full_mat, mat_sub):
    """
    Performs hierarchical spectral biclustering on a given matrix.

    This function iteratively applies spectral biclustering to partition the input
    matrix 'full_mat' into a specified number of clusters 'nclust'. The clustering
    is done hierarchically, refining the partition in each iteration based on the
    results of the previous step. The final clustering labels and the corresponding
    cut value are computed and returned.

    Parameters:
    nclust (int): The target number of clusters to achieve in the biclustering process.
    full_mat (numpy.ndarray): A matrix representing the data to be clustered.

    Returns:
    tuple: A tuple containing two elements:
        - numpy.ndarray: An array representing the final clustering labels for each row in 'full_mat'.
        - float: The cut value representing the quality of the biclustering.
    """
    i = 0
    
    it = int(np.log2(nclust))
    labels_sp = np.zeros((nclust,full_mat.shape[0]))

    for x in range(it):
        grps = np.unique(labels_sp, axis=1)
        index = [[np.unique(labels_sp[:,x]==grps[:,y])[0] for x in range(labels_sp.shape[1])] for y in range(grps.shape[1])]
        for j in range(len(index)):
            get_spectral_partition(labels_sp,  get_LP(full_mat[index[j]][:,index[j]]), i, index[j])
            i+=1
    labels_sp = [np.argwhere(np.sum(np.unique(labels_sp, axis=1)-labels_sp[:,x].reshape(-1,1)==0, axis=0)==nclust)[0][0] for x in range(labels_sp.shape[1])] 
    loss_sp = evaluate_cut(np.ascontiguousarray(mat_sub.values.T), labels_sp, 0)
    
    return labels_sp, loss_sp