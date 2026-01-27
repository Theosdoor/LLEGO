#!/bin/bash
#SBATCH --job-name=rigorous_validation
#SBATCH --output=logs/rigorous_validation_%j.out
#SBATCH --error=logs/rigorous_validation_%j.err
#SBATCH --partition=cnu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Phase 3: Rigorous Validation Experiments
# Matches LLEGO paper setup exactly
# No GPU needed - this is pure CPU

echo "========================================"
echo "Phase 3: Rigorous Validation"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

cd /home/$USER/workspace/LLEGO

# Activate environment
source ~/.bashrc
conda activate llego

# Create logs directory
mkdir -p logs
mkdir -p mi_analysis/results/phase3

# Run rigorous validation with full LLEGO paper setup
echo "Running rigorous validation..."
echo "  Datasets: breast, heart-statlog, diabetes"
echo "  Seeds: 5"
echo "  Population: 25, Generations: 25"
echo "  Metric: Balanced Accuracy"
echo "  Init: CART-bootstrapped"
echo ""

python mi_analysis/validation_rigorous.py \
    --datasets breast heart-statlog diabetes \
    --n-seeds 5 \
    --pop-size 25 \
    --n-generations 25 \
    --output-dir mi_analysis/results/phase3

echo ""
echo "========================================"
echo "Validation complete!"
echo "End time: $(date)"
echo "========================================"
