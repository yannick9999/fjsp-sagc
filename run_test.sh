#!/bin/bash
#SBATCH --job-name=fjsp_test
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/test_seed%a_%j.out
#SBATCH --error=logs/test_seed%a_%j.err

set -euo pipefail

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
# Experiment name comes from config.json (single source of truth)
EXP_NAME=$("${PYTHON}" -c "import json; print(json.load(open('config.json'))['experiment']['name'])")
MODEL_DIR="./save/${EXP_NAME}/seed${SEED}"

# Verify model exists before launching
if ! ls "${MODEL_DIR}"/*.pt 1>/dev/null 2>&1; then
    echo "ERROR: No .pt file found in ${MODEL_DIR}" >&2
    exit 1
fi

echo "=== Testing seed=${SEED} exp=${EXP_NAME} model_dir=${MODEL_DIR} ==="
srun nvidia-smi || true

# exec srun "${PYTHON}" run_test_suite.py --seed "${SEED}" --diagnose
exec srun "${PYTHON}" run_test_suite.py --seed "${SEED}"