#!/bin/bash
#SBATCH --job-name=LLEGO-SAE
#SBATCH --output=slurm/slurm_%j.log
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=8:00:00
#SBATCH --mem=28G

# 1. Go to your folder
cd /home2/nchw73/vanDerSchaarWork/LLEGO

# 2. Debug: Job info
echo "Job running on node: $(hostname)"
echo "------------------------------------------------------"

# Configuration (matching LLEGO paper)
DATASETS=(breast heart-statlog liver credit-g)
MAX_DEPTHS=(3 4)
SEEDS=(0 1 2 3 4)

# ============================================================================
# Phase 1: SAE Extraction (one-shot, ~1 min)
# ============================================================================
echo "=== Phase 1: Extracting SAE Priors ==="
.venv/bin/python sae_project/extract_sae_priors.py --datasets breast heart liver credit-g

# ============================================================================
# Phase 2: Run Distilled Experiments (our method, ~10 min)
# ============================================================================
echo "=== Phase 2: Running Distilled Experiments ==="

for depth in "${MAX_DEPTHS[@]}"; do
    echo "Running SAE Validation with max_depth=$depth"
    .venv/bin/python mi_analysis/sae_validation.py \
        --sae-prior-dir sae_project/priors \
        --datasets breast heart liver credit-g \
        --max-depth $depth \
        --n-seeds 5
done

# ============================================================================
# Phase 3: Run Official GATree Baseline (~30 min)
# ============================================================================
echo "=== Phase 3: Running GATree Baseline ==="

for seed in "${SEEDS[@]}"; do
    for depth in "${MAX_DEPTHS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            echo "GATree: dataset=$dataset, depth=$depth, seed=$seed"
            .venv/bin/python experiments/exp_gatree.py \
                dataset=$dataset \
                max_depth=$depth \
                seed=$seed \
                exp_name=sae_comparison \
                log_wandb=False
        done
    done
done

# ============================================================================
# Phase 4: Run LLEGO Baseline (optional - requires API key, expensive)
# Uncomment if you want to run LLEGO for comparison
# ============================================================================
# echo "=== Phase 4: Running LLEGO Baseline ==="
# 
# for seed in "${SEEDS[@]}"; do
#     for depth in "${MAX_DEPTHS[@]}"; do
#         for dataset in "${DATASETS[@]}"; do
#             echo "LLEGO: dataset=$dataset, depth=$depth, seed=$seed"
#             .venv/bin/python experiments/exp_llego.py \
#                 dataset=$dataset \
#                 max_depth=$depth \
#                 seed=$seed \
#                 exp_name=sae_comparison
#         done
#     done
# done

echo "=== All experiments complete! ==="
echo "Results saved to:"
echo "  - SAE Validation: mi_analysis/results/sae_validation/"
echo "  - GATree: sae_comparison/<depth>/<dataset>/GATREE/"