#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32g    
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daknowles@nygenome.org  

source ~/venv/mamba_ssm/bin/activate

python run_mamba.py