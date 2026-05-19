#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16

set -euo pipefail

START_TIME=$(date +%s)

# Paths
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create a fresh conda environment
CONDA_VENV="finrl_env"
conda remove -n "$CONDA_VENV" --all -y || true
conda env create -f "$ROOT_DIR/environment.yml"

# Install PyTorch with CUDA support 
# (override stable-baselines3's pytorch installation)
conda activate "$CONDA_VENV"
pip uninstall -y torch
pip install torch==2.11.0 \
--index-url https://download.pytorch.org/whl/cu128

END_TIME=$(date +%s)
ELAPSED_S=$((END_TIME - START_TIME))
ELAPSED_H=$(( ELAPSED_S / 3600 ))
ELAPSED_M=$(( (ELAPSED_S % 3600) / 60 ))

echo "Job ${SLURM_JOB_ID:-unknown} finished successfully after ${ELAPSED_H}h ${ELAPSED_M}m."