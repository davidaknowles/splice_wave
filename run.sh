#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64g    
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daknowles@nygenome.org  

source ~/venv/mamba_ssm/bin/activate

python run.py