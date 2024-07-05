#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32g    
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daknowles@nygenome.org  

source ~/venv/mamba_ssm/bin/activate

python run_tcn_backwards_batch.py
