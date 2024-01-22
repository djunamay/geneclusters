# Standard library imports
from functools import partial
import re

# Third-party library imports for data manipulation and computation
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# Third-party library imports for parallel processing
from tqdm.contrib.concurrent import process_map

# Third-party library imports for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# Network analysis library
import networkx

# Bioinformatics and gene set enrichment analysis library
import gseapy

# Local module imports
from geneclusters.geneclusters import (
    get_scores, 
    get_kernighan_lin_clusters, 
    get_gene_pathway_matrix, 
    get_full_matrix_from_bipartite, 
    group, 
    compute_groupped_matrix, 
    find_similar_clusters, 
    get_representative_name_per_cluster
)


def plot_edges(layout, graph, pos):
    """
    Plot the edges of a graph based on the provided layout and a weight threshold.

    This function iterates through the edges of the graph and plots each edge if its 
    weight is greater than or equal to 0.5. The positions of the nodes are determined 
    by the 'layout' parameter, and the edges are drawn as lines between the nodes.

    Parameters:
    layout (dict): A dictionary mapping node identifiers to their positions in 2D space.
    graph (networkx.Graph): A graph object from the NetworkX library.
    pos: Not used in the function. May be intended for future use or an error.
    """
    for e_from, e_to in list(graph.edges):
        ss = np.array([layout[e_from], layout[e_to]])
        if graph.get_edge_data(e_from, e_to)['weight'] >=0.5:
            plt.plot(*(ss.T), c='black', alpha=0.1)

def plot_nodes(graph, selected_names, pos, cur_labels, unique_clusters, colors, S):
    """
    Plot the nodes of a graph based on their cluster membership.

    This function groups nodes of a graph into clusters based on their labels and plots
    each cluster of nodes using the 'plot_single_cluster' function. Nodes are colored and 
    treated differently based on their type and name.

    Parameters:
    graph (networkx.Graph): The graph whose nodes are to be plotted.
    selected_names (list): Names of nodes to be specially highlighted.
    pos (np.ndarray): Positions of the nodes in a 2D space.
    cur_labels (np.ndarray): Current cluster labels for each node in the graph.
    unique_clusters (list): A list of unique cluster identifiers.
    colors (list): A list of colors, one for each cluster.
    S (int or float): Size of the markers for the scatter plot.
    """
    types = np.array([graph.nodes[node]['type'] for node in graph.nodes])
    names = np.array([graph.nodes[node]['name'] for node in graph.nodes])  
    for i, cluster_name in enumerate(unique_clusters):
        index = cur_labels==cluster_name
        if np.sum(index)==0:
            continue
        plot_single_cluster(colors[i], pos[index], types[index], names[index], cluster_name, selected_names, S)

def plot_nodes_continuous(graph, selected_names, pos, cur_labels, unique_clusters, colors, S):
    """
    Plot nodes of a graph using a continuous coloring scheme.

    This function visualizes the nodes of a graph where each node's color is determined 
    by its corresponding label in 'cur_labels'. The color for each node is taken from 
    the 'colors' array. 

    Parameters:
    graph (networkx.Graph): The graph whose nodes are to be plotted.
    selected_names (list): Names of nodes to be specially highlighted (currently not used in the function).
    pos (np.ndarray): Positions of the nodes in a 2D space.
    cur_labels (np.ndarray): Continuous or categorical labels associated with each node.
    unique_clusters (list): List of unique cluster identifiers (currently not used in the function).
    colors (list): A list of colors corresponding to the labels in 'cur_labels'.
    S (int or float): Size of the markers for the scatter plot.
    """
    types = np.array([graph.nodes[node]['type'] for node in graph.nodes])
    names = np.array([graph.nodes[node]['name'] for node in graph.nodes])  
    for i, score in enumerate(cur_labels):
        plt.scatter(pos[i][0], pos[i][1], color=colors[i], zorder=5, s=S, cmap='tab20', edgecolor="black", linewidth=1)


def plot_single_cluster(col, pos_curr, types_curr, names_curr, cluster_name, selected_names, S):
    """
    Plot a single cluster of nodes with specific color and annotations.

    This function visualizes nodes of a single cluster. It uses different markers for 
    different types of nodes and highlights selected node names. The cluster name is 
    also displayed near the cluster center.

    Parameters:
    col (str or tuple): The color used to plot the nodes of this cluster.
    pos_curr (np.ndarray): 2D array of positions for the nodes in this cluster.
    types_curr (np.ndarray): Array indicating the type of each node (0 or 1 for different markers).
    names_curr (np.ndarray): Array of names for each node in this cluster.
    cluster_name (str): The name of the cluster to be displayed.
    selected_names (list): List of node names to be highlighted.
    S (int or float): Size of the markers for the scatter plot.
    """
    props = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor=col, linewidth=3)
    x, y = np.mean(pos_curr, axis = 0)
    x-=0.2
    plt.scatter(*pos_curr[types_curr==0].T, color=col, zorder=5, s=S, cmap='tab20', edgecolor="black", linewidth=1)
    plt.scatter(*pos_curr[types_curr==1].T, color=col, zorder=5, s=S, cmap='tab20', edgecolor="black", linewidth=1, marker='s')
    
    labels = ''
    n=0
    
    for i, name in enumerate(names_curr):
        if name in selected_names:
            plt.text(pos_curr[i,0], pos_curr[i,1], name,  bbox=props, c = col, fontsize = 10, zorder=6, style = "italic")
            
    plt.text(x, y+(0.045*(n+1)), cluster_name,  bbox=props, c = 'black', fontsize = 15, zorder=6, weight = "normal")

def get_le_clusters(le_path, mat, seed, Ngenes):
    """
    Extracts leading edge gene clusters from a given matrix and applies Kernighan-Lin clustering.

    This function reads a set of leading edge genes from a file and then filters a given matrix 
    to retain only columns corresponding to these genes. Further, it filters rows based on a sum 
    threshold and applies the Kernighan-Lin clustering algorithm to the resulting submatrix.

    Parameters:
    le_path (str): Path to the CSV file containing leading edge genes.
    mat (pd.DataFrame): The original matrix with genes as columns.
    seed (int): Seed for random number generator, used in clustering for reproducibility.
    Ngenes (int): The number of genes to consider for clustering.

    Returns:
    tuple: A tuple containing:
        - frame (pd.DataFrame): A DataFrame with cluster assignments.
        - mat_sub (pd.DataFrame): The subset of 'mat' used for clustering.
    """
    leading_edge = pd.read_csv(le_path, index_col=0)
    S = set(leading_edge['gene'])

    col_index = np.where([x in S for x in mat.columns])[0]
    mat_sub = mat.iloc[:,col_index]

    path_index = (np.sum(mat_sub, axis=1)>4)
    mat_sub = mat_sub.loc[path_index]
    mat_sub = mat_sub.loc[:,np.sum(mat_sub, axis=0)>0]
    
    print(mat_sub.shape)
    C = 0
    KL_modified = True
    random_labels = True
    unweighted = True

    frame, loss_temp = get_kernighan_lin_clusters(None, Ngenes, C, KL_modified, random_labels, unweighted, seed=seed, no_progress=False, mat=mat_sub)
    frame.columns = ['cluster', 'description', 'is_gene']
    return frame, mat_sub

def get_layout(frame, mat_sub, it, k=15):
    """
    Constructs a graph from provided data frames and computes its layout.

    This function creates a graph based on the clustering information in 'frame' and 
    the matrix 'mat_sub'. It computes the layout for graph visualization using the 
    spring layout algorithm.

    Parameters:
    frame (pd.DataFrame): DataFrame containing cluster and gene information.
    mat_sub (pd.DataFrame): Subset of the original matrix, used to build the graph.
    it (int): Number of iterations for the spring layout algorithm.
    k (float, optional): Optimal distance between nodes. Default is 15.

    Returns:
    tuple: A tuple containing:
        - graph (networkx.Graph): The constructed graph.
        - pos (np.ndarray): Positions of the nodes in 2D space.
        - cur_labels (np.ndarray): Current labels of the nodes in the graph.
        - unique_clusters (np.ndarray): Unique cluster identifiers.
        - colors (list): List of colors for each cluster.
        - layout (dict): Node layout for graph visualization.
    """
    cols = np.array(frame['cluster'][frame['is_gene']])
    g_names  = np.array(frame['description'][frame['is_gene']])
    rows = np.array(frame['cluster'][np.invert(frame['is_gene'])])
    p_names  = np.array(frame['description'][np.invert(frame['is_gene'])])

    matrix = np.array(mat_sub)

    # set up the graph
    N = len(rows) + len(cols)
    full_matrix = np.zeros((N, N), dtype=matrix.dtype)
    full_matrix[:matrix.shape[0],matrix.shape[0]:] = matrix
    graph = networkx.from_numpy_array(full_matrix)
    cluster_labels = np.concatenate([rows, cols])
    node_names = np.concatenate([p_names, g_names])
    for i, l in enumerate(cluster_labels):
        graph.nodes[i]['cluster_id'] = l
        graph.nodes[i]['type'] = i < matrix.shape[0]
        graph.nodes[i]['name'] = node_names[i]

    for e_i in range(N):
        for e_j in range(e_i + 1, N):
            if graph.nodes[e_i]['cluster_id'] == graph.nodes[e_j]['cluster_id'] and not graph.has_edge(e_i, e_j):
                graph.add_edge(e_i, e_j, weight=0.05)

    components = list(networkx.connected_components(graph))
    unique_clusters = np.unique(np.array([graph.nodes[node]['cluster_id'] for node in graph.nodes]))

    unique_clusters = np.unique(frame['cluster'])
    cm = plt.cm.get_cmap('tab20')
    colors = [cm(int(x)) for x in range(len(unique_clusters))]

    cur_labels = np.array([graph.nodes[node]['cluster_id'] for node in graph.nodes])
    layout = networkx.spring_layout(graph,k=k, iterations=it, weight='weight', seed=5)
    pos = np.array(list(layout.values()))

    return graph, pos, cur_labels, unique_clusters, colors, layout

def plot_graph(layout, pos, graph, cur_labels, unique_clusters, colors, out_path):
    """
    Create and save a graph visualization based on the given layout.

    This function visualizes a graph by plotting its edges and nodes using specified 
    layout positions. It calls 'plot_edges' to draw the edges and 'plot_nodes' for the nodes. 
    The final plot is saved to a PDF file.

    Parameters:
    layout (dict): Layout information for nodes in the graph.
    pos (np.ndarray): Positions of the nodes in a 2D space.
    graph (networkx.Graph): The graph to be visualized.
    cur_labels (np.ndarray): Current labels of the nodes in the graph.
    unique_clusters (np.ndarray): Unique cluster identifiers.
    colors (list): List of colors for each cluster.
    out_path (str): Base path for the output file. The filename will be appended with '_network.pdf'.
    """
    plt.figure(figsize = (7,5))
    plot_edges(layout, graph, pos)
    plot_nodes(graph, [], pos, cur_labels, unique_clusters, colors, 100)
    a = plt.gca()
    a.axis('off')
    plt.savefig(out_path+'_network.pdf', bbox_inches="tight")

def get_representative_name_per_cluster(bipartite_mat, colnames_mat, rownames_mat, description_table, cluster, N=5):
    """
    Retrieves representative names for a given cluster from a bipartite matrix.

    This function selects representative names (e.g., gene names) for a specified cluster 
    based on the information in a bipartite matrix. It focuses on entries that are part 
    of the given cluster and calculates the sum of interactions within the cluster.

    The metric used to identify representative pathway names is the sum of internal interactions within the cluster for each path, with higher sums indicating greater relevance or significance to the cluster.
 
    Parameters:
    bipartite_mat (np.ndarray): The bipartite matrix representing interactions.
    colnames_mat (list or np.ndarray): Column names of the bipartite matrix, representing one set of entities.
    rownames_mat (list or np.ndarray): Row names of the bipartite matrix, representing another set of entities.
    description_table (pd.DataFrame): A DataFrame with descriptions, indicating whether an entity is a gene 
                                      and its associated cluster.
    cluster (int): The cluster number for which representative names are to be found.
    N (int, optional): The number of top representative names to return. Default is 5.

    Returns:
    tuple: Returns a tuple where the first element is an array of representative names and 
           the remaining elements are np.nan if no paths are found, otherwise np.nan values 
           are replaced with calculated sums.
    """
    genes = set(description_table.loc[description_table['is_gene']&(description_table['cluster']==cluster)]['description'])
    paths = set(description_table.loc[np.invert(description_table['is_gene'])&(description_table['cluster']==cluster)]['description'])
    if len(paths)==0:
        return 'C.'+str(cluster), np.nan, np.nan, np.nan
    else:
        index_col = [x in genes for x in colnames_mat]
        index_row = [x in paths for x in rownames_mat]
            
        sum_internal = np.sum(bipartite_mat[index_row][:,index_col], axis=1)
        S = np.argsort(-1*sum_internal)[:N]
        rep_name = rownames_mat[index_row][S]

        return rep_name 

def plot_rep_names(pos, unique_clusters, colors, mat_sub, frame, out_path, N):
    """
    Create and save plots with representative names for each cluster.

    This function generates plots for each unique cluster, displaying representative names 
    derived from the cluster data. It calculates these names using the 
    `get_representative_name_per_cluster` function and then creates a plot for each 
    cluster with these names.

    Parameters:
    pos (np.ndarray): Positions of the clusters in 2D space.
    unique_clusters (np.ndarray): Array of unique cluster identifiers.
    colors (list): List of colors for each cluster.
    mat_sub (pd.DataFrame): Subset of the original matrix, used for clustering analysis.
    frame (pd.DataFrame): DataFrame containing clustering information.
    out_path (str): Base path for saving the output plots.
    N (int): Number of top representative names to be shown for each cluster.

    Each plot is saved as a separate PDF file, named with the cluster index and based on the 'out_path'.
    """
    colnames = np.array(mat_sub.columns)
    rownames = np.array(mat_sub.index)

    out = [get_representative_name_per_cluster(np.array(mat_sub), colnames, rownames, frame, x, N) for x in np.unique(frame['cluster'])]

    plt.figure(figsize = (1,1))
    a = plt.gca()
    a.axis('off')
    texts = []
    y = 0
    for i, cluster_name in enumerate(unique_clusters):
        index = cur_labels==cluster_name
        x, y = np.mean(pos[index], axis=0)

        props = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor=colors[i])
        temp = [re.split(r' [(]GO|WP| R-',x)[0] for x in out[i]]
        T = ('\n').join(temp)
        plt.text(0, 0, T,  bbox=props, c ='black', fontsize = 10)#, #style = "italic")#colors[i]

        a = plt.gca()
        a.axis('off')
        plt.savefig(out_path + str(i) + '.pdf', bbox_inches="tight")
        plt.figure()


def get_top_genes(frame, scores, cluster, N, celltype, mat_sub):
    """
    Identify and return the top N genes in a specified cluster.

    This function selects the top genes in a given cluster based on their scores and
    a threshold value. It filters genes based on their association with the cluster
    and the absolute value of their score in a specific cell type. It then checks
    if genes are more associated with the given cluster than with others.

    Parameters:
    frame (pd.DataFrame): DataFrame containing cluster information for genes.
    scores (pd.DataFrame): DataFrame containing scores for each gene in different cell types.
    cluster (int): The cluster number to analyze.
    N (int): The number of top genes to return.
    celltype (str): The cell type to consider for the gene scores.
    mat_sub (pd.DataFrame): A matrix with genes and their associations.

    Returns:
    pd.DataFrame: A DataFrame of the top N genes in the specified cluster, sorted by 
                  their score in the given cell type.

    The function filters genes with an absolute score greater than 1.3 in the specified cell type.
    It then checks the association of genes with pathways inside and outside the cluster and
    selects genes more associated with the cluster. The final list is sorted by the score.
    """
    temp = frame[frame['cluster']==cluster]
    temp = scores.loc[temp['description'][temp['is_gene']]]
    temp = temp[np.abs(temp[celltype])>1.3]
    
    mat_sub_T = mat_sub.T
    
    index = []
    for i in temp.index:
        x = np.array(mat_sub_T.columns)[np.array(mat_sub_T.loc[i]==1)]
        inside = len(x[[i in set(frame[frame['cluster']==cluster]['description']) for i in x]])
        outside = len(x[[i in set(frame[frame['cluster']!=cluster]['description']) for i in x]])
        index.append(((inside>=outside)&(inside>=1)))
    
    temp = temp[index]
    temp['gene'] = temp.index
    temp['score'] = np.abs(temp[celltype])
    temp = temp.sort_values(by='score', ascending=False)
    temp['score'] = (temp[celltype])
    temp = temp[:N]
    temp = temp.sort_values(by='score', ascending=False)
    return temp

def plot_sub_graph(cluster, frame, celltype, out_path, layout, graph, pos, cur_labels, scores, mat_sub, figsize0, figsize1, figsize2, N=15):
    """
    Create and save visual representations of a specific graph cluster.

    This function generates three types of plots for a specified cluster: 
    1. A highlight of the cluster within the graph.
    2. A bar plot of the top genes in the cluster.
    3. A bar plot of the top pathways associated with the cluster.

    Parameters:
    cluster (int): The specific cluster to be visualized.
    frame (pd.DataFrame): DataFrame containing cluster and gene/pathway information.
    celltype (str): Cell type for score calculation.
    out_path (str): Path for saving the output plots.
    layout (dict): Layout information for nodes in the graph.
    graph (networkx.Graph): The graph object.
    pos (np.ndarray): Positions of nodes.
    cur_labels (np.ndarray): Labels of nodes in the graph.
    scores (pd.DataFrame): DataFrame containing scores for genes in different cell types.
    mat_sub (pd.DataFrame): Matrix with gene-pathway associations.
    figsize0, figsize1, figsize2 (tuple): Figure sizes for the three plots.
    N (int, optional): Number of top genes/pathways to include in the bar plots. Default is 15.

    The function creates and saves:
    - A PDF highlighting the specified cluster within the graph.
    - A PDF of the top N genes bar plot.
    - A PDF of the top pathways bar plot.
    """    
    unique_clusters = np.unique(frame['cluster'])
    cm = plt.cm.get_cmap('tab20')
    colors = [cm(int(x)) for x in range(len(unique_clusters))]
    colors2 = list((colors[cluster]))
    
    plt.figure(figsize = (7,5))
    cur_labels_copy = cur_labels.copy()
    cur_labels_copy[cur_labels_copy!=cluster]=9

    # plot by cluster color
    pos = np.array(list(layout.values()))
    plot_edges(layout, graph, pos)
    index = cur_labels==cluster
    plot_nodes(graph.subgraph(np.where(np.invert(index))[0]), [], pos[np.invert(index)], cur_labels[np.invert(index)],list(unique_clusters[unique_clusters!=cluster]), [(0, 0, 0, 0.17) for x in range(len(unique_clusters))], 100)
    plot_nodes(graph.subgraph(np.where(index)[0]), [], pos[index], cur_labels[index],list([cluster]), list([colors[cluster]]), 100)
    
    a = plt.gca()
    a.axis('off')
    plt.savefig(out_path+str(cluster)+'_highlighted.pdf', dpi=300)
    
    # plot the genes
    temp = get_top_genes(frame, scores, cluster, N, celltype, mat_sub)

    plt.figure(figsize = figsize1)
    sns.barplot(data=temp, x=celltype, y='gene', color=colors[cluster])

    plt.xlabel('-log10(p-value)*sign(log2(FC))')
    plt.ylabel('')
    sns.despine(top=True, right=True, left=False, bottom=False)

    plt.savefig(out_path+str(cluster)+'_genes.pdf', bbox_inches="tight")
    
    # plot the pathways
    temp = frame[frame['cluster']==cluster]
    
    T = mat_sub[temp['description'][temp['is_gene']]].loc[temp['description'][np.invert(temp['is_gene'])]]
    temp = np.matmul(T,scores.loc[T.columns][celltype])
    temp = pd.DataFrame(temp/np.sum(T, axis=1))
    
    index = []
    for i in temp.index:
        x = np.array(mat_sub.columns)[np.array(mat_sub.loc[i]==1)]
        inside = len(x[[i in set(frame[frame['cluster']==cluster]['description']) for i in x]])
        outside = len(x[[i in set(frame[frame['cluster']!=cluster]['description']) for i in x]])
        index.append(((inside>=outside)&(inside>=5)))
    
    temp = temp[index]
    temp['pathway'] = [x.split(' WP')[0] for x in np.array(temp.index)]
    temp['score'] = np.abs(temp[0])
    temp = temp.sort_values(by='score', ascending=False)
    temp['score'] = (temp[0])
    temp = temp[:5]
    temp = temp.sort_values(by='score', ascending=False)
    plt.figure(figsize = figsize2)
    sns.barplot(data=temp, x='score', y='pathway', color=colors[cluster])
    plt.xlabel('score')
    plt.ylabel('')
    sns.despine(top=True, right=True, left=False, bottom=False)

    plt.savefig(out_path+str(cluster)+'_bars.pdf', bbox_inches="tight")

def plot_scores(frame, scores, celltype, unique_clusters, out_path):
    """
    Create and save a plot of scores for gene clusters.

    This function plots the scores for different gene clusters based on a specific cell type.
    It first computes the sum of scores for each cluster and then generates a bar plot
    showing these scores.

    Parameters:
    frame (pd.DataFrame): DataFrame containing cluster information for genes.
    scores (pd.DataFrame): DataFrame containing scores for each gene in different cell types.
    celltype (str): The cell type for which scores are considered.
    unique_clusters (np.ndarray or list): Array or list of unique cluster identifiers.
    out_path (str): Path for saving the output plot.

    Returns:
    pd.DataFrame: DataFrame with the sum of scores for each cluster.

    The function saves the plot as a PDF file using the specified 'out_path'.
    """
    # plot scores
    score_sum = get_score(frame, scores, celltype, unique_clusters)
    
    plt.figure(figsize = (4,3))
    plt.xlabel('ABCA7 LoF perturbation score')
    plt.ylabel('Gene Cluster')
    plt.savefig(out_path+'_bars.pdf', bbox_inches="tight")
    return score_sum

def plot_sub_graph_only(cluster, frame, celltype, out_path, layout, graph, pos, cur_labels, scores, mat_sub, figsize0, figsize1, figsize2, N=15):
    """
    Create and save a visualization of a specific cluster within a graph.

    This function generates a plot highlighting a specified cluster in a graph against the background
    of other clusters. It uses pre-defined functions 'plot_edges' and 'plot_nodes' to visualize the
    graph's structure and the specific cluster.

    Parameters:
    cluster (list or int): The cluster(s) to be highlighted in the visualization.
    frame (pd.DataFrame): DataFrame containing cluster information.
    celltype (str): The cell type, used for other analyses (not directly used in this function).
    out_path (str): The file path to save the plot.
    layout (dict): Layout information for nodes in the graph.
    graph (networkx.Graph): The graph object.
    pos (np.ndarray): Positions of the nodes.
    cur_labels (np.ndarray): Labels of the nodes in the graph.
    scores (pd.DataFrame): DataFrame containing scores (not directly used in this function).
    mat_sub (pd.DataFrame): Matrix with additional data (not directly used in this function).
    figsize0, figsize1, figsize2 (tuple): Figure sizes for different plots (figsize0 not used here).
    N (int): Number of elements to consider (not directly used in this function).

    The function highlights the specified cluster(s) with distinct colors and sets the rest of the graph
    nodes to a translucent color. The result is saved as a high-resolution PDF.
    """
    unique_clusters = np.unique(frame['cluster'])
    cm = plt.cm.get_cmap('tab20')
    colors = [cm(int(x)) for x in range(len(unique_clusters))]

    plt.figure(figsize = (7,5))

    # plot by cluster color
    pos = np.array(list(layout.values()))
    plot_edges(layout, graph, pos)
    index = [x in set(cluster) for x in cur_labels]
    plot_nodes(graph.subgraph(np.where(np.invert(index))[0]), [], pos[np.invert(index)], cur_labels[np.invert(index)],list(unique_clusters[[x not in set(cluster) for x in unique_clusters]]), [(0, 0, 0, 0.17) for x in range(len(unique_clusters))], 100)
    plot_nodes(graph.subgraph(np.where(index)[0]), [], pos[index], cur_labels[index],cluster, [colors[x] for x in cluster], 100)
    
    a = plt.gca()
    a.axis('off')
    plt.savefig(out_path+str(cluster)+'_highlighted.pdf', dpi=300)

def score_rep_paths(frame, mat_sub, scores, cluster, celltype, thresh=5):
    """
    Calculate and return scores for representative pathways associated with a specific cluster.

    This function identifies and scores pathways related to a given cluster based on their 
    association strength and relevance. The scoring is based on the proportion of interaction 
    strength in the provided cell type, as well as a comparison of interactions within and 
    outside the cluster.

    Parameters:
    frame (pd.DataFrame): DataFrame containing information about genes and pathways in clusters.
    mat_sub (pd.DataFrame): Matrix with gene-pathway associations.
    scores (pd.DataFrame): DataFrame containing scores for different cell types.
    cluster (int): The cluster number for which pathway scores are to be calculated.
    celltype (str): The cell type for which scores are considered.
    thresh (int, optional): The threshold for considering an interaction as significant. Default is 5.

    Returns:
    pd.DataFrame: A DataFrame containing the top scored pathways for the specified cluster, 
                  with each pathway's score and its truncated name.

    The function filters pathways based on their interaction strength and relevance to the cluster,
    compares the number of internal and external interactions, and sorts them based on their scores.
    """
    temp = frame[frame['cluster']==cluster]
    
    T = mat_sub[temp['description'][temp['is_gene']]].loc[temp['description'][np.invert(temp['is_gene'])]]
    temp = np.matmul(T,scores.loc[T.columns][celltype])
    temp = pd.DataFrame(temp/np.sum(T, axis=1))
    
    index = []
    for i in temp.index:
        x = np.array(mat_sub.columns)[np.array(mat_sub.loc[i]==1)]
        inside = len(x[[i in set(frame[frame['cluster']==cluster]['description']) for i in x]])
        outside = len(x[[i in set(frame[frame['cluster']!=cluster]['description']) for i in x]])
        index.append(((inside>=outside)&(inside>=thresh)))
    
    temp = temp[index]
    temp['pathway'] = [x.split(' WP')[0] for x in np.array(temp.index)]
    temp['score'] = np.abs(temp[0])
    temp = temp.sort_values(by='score', ascending=False)
    temp['score'] = (temp[0])
    temp = temp[:5]
    temp = temp.sort_values(by='score', ascending=False)
    temp['cluster'] = cluster
    
    return temp

def plot_component(graph, selected_names, unique_clusters, colors, k, iterations, scale, component = None, center=None, seed=None, S=200):  
    if component is None:
        graph_temp = graph
    else:
        graph_temp = graph.subgraph(component)
    cur_labels = np.array([graph.nodes[node]['cluster_id'] for node in graph_temp.nodes])

    layout = networkx.spring_layout(graph_temp,k=k, iterations=iterations, weight='weight', scale=scale, seed=seed, center=center)
    pos = np.array(list(layout.values()))
    plot_edges(layout, graph_temp, pos)
    plot_nodes(graph_temp, selected_names, pos, cur_labels, unique_clusters, colors, S)

