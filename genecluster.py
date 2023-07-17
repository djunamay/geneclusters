# load libraries
import os
from geneclusters.cluster_genes_new import get_gene_pathway_matrix, create_random_labeling, run_KL, evaluate_cut
import numpy as np
from os import path
from tqdm.contrib.concurrent import process_map
from functools import partial

# Functions

def loop_once(matrix, threshold, C=0, KL_modified=True, no_progress=False, seed=None):
    labeling = create_random_labeling(matrix, threshold, seed)
    run_KL(labeling, matrix, C, KL_modified, no_progress, seed)
    loss = evaluate_cut(matrix, labeling, C)                           
    return labeling, loss


# define vars
task_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])

# load matrix
p = './examples/GO_Biological_Process_2023.npy'
mat = get_gene_pathway_matrix(p)
runs_per_job = 100
N = 1000

# generate memmaps
mmap_1 = './examples/GO_Biological_Process_2023_labels.npy'
mmap_2 = './examples/GO_Biological_Process_2023_losses.npy'

if not path.exists(mmap_1):
    mode = 'w+'
else:
    mode = 'r+'

labels = np.lib.format.open_memmap(mmap_1, shape=(N, sum(mat.shape)), dtype=int, mode=mode)
losses = np.lib.format.open_memmap(mmap_2, shape=(1, N), dtype=int, mode=mode)

# initialize labels
matrix = np.ascontiguousarray(mat.values.T)
                                   
start = runs_per_job * task_ID
end = runs_per_job + start

seed = np.arange(start, end)

threshold = 30
C = 0
KL_modified = True
no_progress = True 

results = process_map(partial(loop_once, matrix, threshold, C, KL_modified, no_progress), seed)

labels[start:end] = np.stack([x[0] for x in results])
losses[0][start:end] = np.stack([x[1] for x in results])




