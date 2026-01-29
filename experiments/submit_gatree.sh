#!/bin/bash
#SBATCH --job-name=GATree_full
#SBATCH --output=slurm/slurm_%j.log
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G

# Navigate to repository
cd /home2/nchw73/vanDerSchaarWork/LLEGO

# Ensure base environment is set up first
echo "Setting up base environment..."
uv sync
echo ""

# Install external dependencies (gatree, bonsai-dt, pydl8.5)
echo "Installing external dependencies..."
bash install_external.sh
echo ""

# Job info
echo "Job running on node: $(hostname)"
echo "Starting GATree experiments at: $(date)"
echo "------------------------------------------------------"

# Run all GATree experiments
# Stay in root directory so imports work correctly
bash experiments/run_gatree.sh

echo "------------------------------------------------------"
echo "GATree experiments completed at: $(date)"
