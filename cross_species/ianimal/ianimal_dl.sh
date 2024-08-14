#!/bin/bash

#SBATCH --array=14,15
#SBATCH --mem=32g
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daknowles@nygenome.org  

source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
conda activate asb

python ianimal_dl.py $SLURM_ARRAY_TASK_ID