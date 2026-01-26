#!/bin/bash
#SBATCH --job-name=mi_phase1
#SBATCH --output=slurm/mi_phase1_%j.log
#SBATCH --error=slurm/mi_phase1_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=2

# =============================================================================
# MI Analysis Phase 1: Full Diagnostic Suite
# =============================================================================
#
# Runs all Phase 1 experiments:
# 1. Semantic Ablation Study
# 2. Fitness Attention Analysis  
# 3. Structural Prior Elicitation (if implemented)
#
# Usage:
#   sbatch mi_analysis/submit_phase1.sh
#
# GPU Options (edit #SBATCH --gres line):
#   Turing (often available): gpu:turing:1
#   Pascal: gpu:pascal:1
#   A6000: gpu:a6000:1
#   Ampere (long wait): gpu:ampere:1
#
# =============================================================================

set -e  # Exit on error

# Create directories
mkdir -p slurm
mkdir -p mi_analysis/results
mkdir -p mi_analysis/cache

# Navigate to project root
# UPDATE THIS PATH to your cluster home directory  
PROJECT_DIR="${PROJECT_DIR:-/home2/nchw73/vanDerSchaarWork/LLEGO}"
cd "$PROJECT_DIR"

# Activate environment
source .venv/bin/activate

# Set cache directories (UPDATE THESE for your cluster)
CACHE_BASE="${CACHE_BASE:-/home2/nchw73/.cache}"
export HF_HOME="$CACHE_BASE/huggingface"
export HF_DATASETS_CACHE="$CACHE_BASE/huggingface/datasets"
mkdir -p $HF_HOME $HF_DATASETS_CACHE

# Print job info
echo "=============================================="
echo "MI Analysis Phase 1: Full Diagnostic Suite"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start time: $(date)"
echo "=============================================="

# Model to use (Llama 3.1 8B in 4-bit quantization fits on 11GB Turing GPUs)
MODEL="meta-llama/Llama-3.1-8B-Instruct"
# Use 4-bit quantization for 11GB GPUs, remove flag for larger GPUs
QUANTIZATION="--load-in-4bit"

# =============================================================================
# Experiment 1: Semantic Ablation
# =============================================================================
echo ""
echo "=============================================="
echo "Running Experiment 1: Semantic Ablation Study"
echo "=============================================="

python mi_analysis/semantic_ablation.py \
    --model $MODEL \
    --datasets breast heart-statlog diabetes \
    --n-crossovers 15 \
    --seeds 0 1 2 \
    --device cuda \
    --output-dir mi_analysis/results/phase1 \
    $QUANTIZATION

# =============================================================================
# Experiment 2: Fitness Attention Analysis
# =============================================================================
echo ""
echo "=============================================="
echo "Running Experiment 2: Fitness Attention Analysis"
echo "=============================================="

python mi_analysis/fitness_attention.py \
    --model $MODEL \
    --n-samples 30 \
    --device cuda \
    --output-dir mi_analysis/results/phase1 \
    $QUANTIZATION

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Phase 1 Complete"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved to: mi_analysis/results/phase1/"
echo ""
echo "Next steps:"
echo "  1. Review results in mi_analysis/results/phase1/"
echo "  2. Run analysis notebook: mi_analysis/analyze_phase1.ipynb"
echo "  3. Based on findings, proceed to Phase 2 (distillation)"
echo "=============================================="
