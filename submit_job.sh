#!/bin/bash
#SBATCH --job-name=LLEGO
#SBATCH --output=slurm/slurm_%j.log
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=5:00:00
#SBATCH --mem=28G


# 1. Go to your folder
cd /home2/nchw73/vanDerSchaarWork/LLEGO

# 2. Debug: Job info
echo "Job running on node: $(hostname)"
echo "------------------------------------------------------"

# 3. Run the LLEGO experiment
.venv/bin/python sae_project/nb_main.py