#!/usr/bin/env bash
set -euo pipefail

#BSUB -J master_thesis_experiments
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -n 4
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -o logs/experiments_%J.out
#BSUB -e logs/experiments_%J.err
#BSUB -N

# -- Load Required Modules --
module load cuda/11.6
module load python3/3.10.18

# -- Activate Environment --
source my_env/bin/activate

# -- Set Environment Variables --
export PYTHONPATH="$(pwd)"
export PM4PY_SHOW_PROGRESS_BAR=False
export PM4PY_SHOW_INTERNAL_WARNINGS=False

# -- Configurable Parameters --
RUNS="${RUNS:-10}"
BASE_SEED="${BASE_SEED:-42}"
TASK="${TASK:-next_activity}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:--1}"

# -- Run Script --
python -u experiments/streaming/run_robustness_experiments.py \
  --base-path="$BASE_PATH" \
  --workers="$WORKERS" \
  --runs="$RUNS" \
  --base-seed="$BASE_SEED" \
  --task="$TASK" \
  --experiment-name="$EXPERIMENT_NAME" \
  --device="$DEVICE"