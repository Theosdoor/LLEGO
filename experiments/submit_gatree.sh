#!/bin/bash
#SBATCH --job-name=GATree_full
#SBATCH --output=../slurm/slurm_%j.log
#SBATCH --error=../slurm/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G

# Navigate to repository
cd /home2/nchw73/vanDerSchaarWork/LLEGO
source .venv/bin/activate

# Job info
echo "Job running on node: $(hostname)"
echo "Starting GATree experiments at: $(date)"
echo "------------------------------------------------------"

# Run all GATree experiments
cd experiments
bash run_gatree.sh

echo "------------------------------------------------------"
echo "GATree experiments completed at: $(date)"
