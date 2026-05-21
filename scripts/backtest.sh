#!/bin/bash
#SBATCH --job-name=finrl-backtest
#SBATCH --time=6:00:00
#SBATCH --partition=ampere
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1

set -euo pipefail

START_TIME=$(date +%s)

# Set up paths
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate environment
CONDA_VENV="finrl_env"
conda activate "$CONDA_VENV"

# Run the source file
cd "$ROOT_DIR"
PYTHONPATH=src python3 "src/backtest.py" "$@"

END_TIME=$(date +%s)
ELAPSED_S=$((END_TIME - START_TIME))
ELAPSED_H=$(( ELAPSED_S / 3600 ))
ELAPSED_M=$(( (ELAPSED_S % 3600) / 60 ))

echo "Job ${SLURM_JOB_ID:-unknown} finished successfully after ${ELAPSED_H}h ${ELAPSED_M}m."
