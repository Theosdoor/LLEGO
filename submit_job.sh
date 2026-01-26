#!/bin/bash
#SBATCH --job-name=LLEGO
#SBATCH --output=slurm/slurm_%j.log
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=5:00:00
#SBATCH --mem=16G


# 1. Go to your folder
cd /home2/nchw73/vanDerSchaarWork/LLEGO
source .venv/bin/activate

# 2. Debug: Job info
echo "Job running on node: $(hostname)"
echo "------------------------------------------------------"

# 3. Run the LLEGO experiment
python3 experiments/exp_llego.py dataset=credit-g max_depth=3 seed=0 exp_name=classification
python3 analysis/read_results.py