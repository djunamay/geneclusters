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

