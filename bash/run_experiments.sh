#!/usr/bin/env bash
set -euo pipefail

#BSUB -J master_thesis_experiments
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -n 4
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o logs/experiments_%J.out
#BSUB -e logs/experiments_%J.err
#BSUB -N

# -- Load Required Modules --
module load cuda/11.6
module load python3/3.10.18

# -- Activate Environment --
source /zhome/5b/6/228538/projects/master-thesis/my_env/bin/activate

# -- Set Environment Variables --
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PM4PY_SHOW_PROGRESS_BAR=False
export PM4PY_SHOW_INTERNAL_WARNINGS=False

# -- Run Script --
python -u experiments/streaming/run_experiments.py --base-path=conf/experiments/streaming/next_activity/baseline/process_transformer --report-dir=experiments/outputs/streaming/next_activity/baseline/process_transformer --device=cuda --workers=4