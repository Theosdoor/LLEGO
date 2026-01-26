#!/bin/bash
#SBATCH --job-name=mi_semantic_ablation
#SBATCH --output=slurm/mi_%j.log
#SBATCH --error=slurm/mi_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=8:00:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=2

# =============================================================================
# Semantic Ablation Experiment - MI Analysis Phase 1
# =============================================================================
# 
# This script runs the semantic ablation study to understand whether LLMs
# leverage semantic feature knowledge in decision tree crossover.
#
# Usage:
#   sbatch mi_analysis/submit_semantic_ablation.sh
#
# Or with custom arguments:
#   sbatch mi_analysis/submit_semantic_ablation.sh --n-crossovers 10 --use-nnsight
#
# =============================================================================

# Create slurm output directory
mkdir -p slurm
mkdir -p mi_analysis/results
mkdir -p mi_analysis/cache

# Navigate to project root
# UPDATE THIS PATH to your cluster home directory
PROJECT_DIR="${PROJECT_DIR:-/home2/nchw73/vanDerSchaarWork/LLEGO}"
cd "$PROJECT_DIR"

# Activate environment
source .venv/bin/activate

# Debug info
echo "=============================================="
echo "MI Analysis: Semantic Ablation Study"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "=============================================="

# Set HuggingFace cache directory (important for cluster)
export HF_HOME=/home2/nchw73/.cache/huggingface
export TRANSFORMERS_CACHE=/home2/nchw73/.cache/huggingface/transformers
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE

# Default arguments (can be overridden via command line)
MODEL=${MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
DATASETS=${DATASETS:-"breast heart-statlog"}
N_CROSSOVERS=${N_CROSSOVERS:-10}
SEEDS=${SEEDS:-"0 1 2"}
USE_NNSIGHT=${USE_NNSIGHT:-""}

# Parse any additional arguments passed to sbatch
EXTRA_ARGS="$@"

# Build command
CMD="python mi_analysis/semantic_ablation.py \
    --model $MODEL \
    --datasets $DATASETS \
    --n-crossovers $N_CROSSOVERS \
    --seeds $SEEDS \
    --device cuda"

# Add nnsight flag if requested
if [ -n "$USE_NNSIGHT" ]; then
    CMD="$CMD --use-nnsight"
fi

# Add any extra arguments
CMD="$CMD $EXTRA_ARGS"

echo "Running command:"
echo "$CMD"
echo "=============================================="

# Run the experiment
$CMD

# Check exit status
EXIT_STATUS=$?
echo "=============================================="
echo "Experiment finished with exit status: $EXIT_STATUS"
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_STATUS
