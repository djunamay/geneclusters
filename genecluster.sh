#!/bin/bash
#SBATCH --job-name=ad_dm    # Job name
#SBATCH --mem=50G                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH -c 20 
#SBATCH -p xeon-p8
#SBATCH --array 0-99

module load anaconda/2022a
python genecluster.py