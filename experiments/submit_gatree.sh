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

# Handle potential stale file handles from network filesystem
if [ -d ".venv" ]; then
    echo "Checking existing virtual environment..."
    # Try to access it, if it fails, remove and recreate
    if ! ls .venv/lib > /dev/null 2>&1; then
        echo "Stale venv detected, removing..."
        rm -rf .venv 2>/dev/null || true
        # If rm fails, try to force unmount/clear (NFS issue workaround)
        find .venv -delete 2>/dev/null || true
    fi
fi

# Ensure base environment is set up first (uv sync creates/updates .venv)
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
