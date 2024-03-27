#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32g    
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daknowles@nygenome.org  

source ~/venv/mamba_ssm/bin/activate

# Define objective choices
objectives=("pred_noise" "pred_x0" "pred_v")

# Define learning rates
learning_rates=("1e-3" "1e-4" "1e-5")

# Loop over objective choices
for objective in "${objectives[@]}"; do
    echo "Objective: $objective"

    # Loop over learning rates
    for lr in "${learning_rates[@]}"; do
        echo "Learning Rate: $lr"

        # Run your Python script with the current objective and learning rate
        python run_diffusion.py "$objective" --self_cond --lr "$lr"
        python run_diffusion.py "$objective" --lr "$lr"
    done
done