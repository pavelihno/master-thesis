#!/usr/bin/env bash
set -euo pipefail

#BSUB -J master_thesis_hypersearch
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -n 4
#BSUB -R "rusage[mem=512MB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o logs/hypersearch_%J.out
#BSUB -e logs/hypersearch_%J.err
#BSUB -N

# -- Load Required Modules --
module load cuda/11.6
module load python3/3.10.18

# -- Activate Environment --
source my_env/bin/activate

# -- Set Environment Variables --
export PYTHONPATH=$(pwd)
export PM4PY_SHOW_PROGRESS_BAR=False
export PM4PY_SHOW_INTERNAL_WARNINGS=False

# -- Run Script --
python -u experiments/streaming/run_hyperparam_search.py "$CONFIG_PATH" --device=cuda