#!/bin/bash
#SBATCH --job-name=fjsp_train
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/train_seed%a_%j.out
#SBATCH --error=logs/train_seed%a_%j.err

set -euo pipefail

# Experiment name: change this for each experiment configuration
EXP_NAME="${EXP_NAME:?Set EXP_NAME before submitting, e.g.: export EXP_NAME=sagc_20x10}"

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "${REPO_ROOT}"
mkdir -p logs

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="/home/grafyann/master_thesis_env/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing ${PYTHON}" >&2; exit 1; }

SEED=${SLURM_ARRAY_TASK_ID}

echo "=== Training seed=${SEED} exp=${EXP_NAME} ==="
srun nvidia-smi || true

exec srun "${PYTHON}" train.py --seed "${SEED}" --exp_name "${EXP_NAME}"
