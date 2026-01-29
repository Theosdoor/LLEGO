#!/bin/bash
#SBATCH --job-name=LLEGO
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
echo "Starting LLEGO experiments at: $(date)"
echo "------------------------------------------------------"

# Run all LLEGO experiments
cd experiments
bash run_llego.sh

echo "------------------------------------------------------"
echo "LLEGO experiments completed at: $(date)"
