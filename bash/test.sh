#!/bin/bash
set -euo pipefail

#BSUB -J test
#BSUB -q hpc
#BSUB -W 15
#BSUB -R "rusage[mem=5GB]"
#BSUB -o logs/test_%J.out
#BSUB -e logs/test_%J.err
#BSUB -B
#BSUB -N

# -- Activate Environment --
source my_env/bin/activate

# -- Set Environment Variables --
export PYTHONPATH=$(pwd)
export PM4PY_SHOW_PROGRESS_BAR=False
export PM4PY_SHOW_INTERNAL_WARNINGS=False

# -- Run Script --
python -u experiments/streaming/run_train.py "$CONFIG_PATH" --device=auto