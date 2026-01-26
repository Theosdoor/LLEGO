#!/bin/bash
# =============================================================================
# Quick local test for semantic ablation (dry run)
# =============================================================================
# Run this locally to verify the code works before submitting to cluster.
#
# Usage:
#   bash mi_analysis/run_local_test.sh
# =============================================================================

cd "$(dirname "$0")/.."

echo "Running semantic ablation dry run test..."

python mi_analysis/semantic_ablation.py \
    --dry-run \
    --datasets breast heart-statlog \
    --n-crossovers 3 \
    --seeds 0 \
    --output-dir mi_analysis/results/test

echo "Done! Check mi_analysis/results/test/ for output."
