#!/bin/bash
#SBATCH --job-name=LLEGO
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

# Ensure base environment is set up first
echo "Setting up base environment..."
uv sync
echo ""

# Job info
echo "Job running on node: $(hostname)"
echo "Starting LLEGO experiments at: $(date)"
echo "------------------------------------------------------"

# Run all LLEGO experiments
# Stay in root directory so imports work correctly
bash experiments/run_llego.sh

echo "------------------------------------------------------"
echo "LLEGO experiments completed at: $(date)"
