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
# Extract SAE priors for all datasets
.venv/bin/python sae_project/extract_sae_priors.py --datasets breast heart liver credit-g

# Run validation with depth=3 (matching LLEGO paper)
.venv/bin/python mi_analysis/sae_validation.py \
    --sae-prior-dir sae_project/priors \
    --datasets breast heart liver credit-g \
    --max-depth 3 \
    --n-seeds 5

# Run validation with depth=4 (matching LLEGO paper)
.venv/bin/python mi_analysis/sae_validation.py \
    --sae-prior-dir sae_project/priors \
    --datasets breast heart liver credit-g \
    --max-depth 4 \
    --n-seeds 5