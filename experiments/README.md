# Experiment Framework

This framework allows you to run experiments using YAML configuration files for reproducible and configurable machine learning experiments.

## Quick Start

### 1. Create a YAML Configuration File

Configuration files are organized by model type in `conf/experiments/<model_type>/<Dataset_Name>.yaml`.

**IMPORTANT:** Filenames must exactly match dataset names from `datasets/links.json`.

Example structure:

```
conf/experiments/
├── xgboost/
│   ├── Traffic_Fines.yaml
│   ├── Helpdesk.yaml
│   └── BPIC_12.yaml
├── random_forest/
│   └── Traffic_Fines.yaml
├── svm/
│   └── Traffic_Fines.yaml
└── logistic_regression/
    └── Traffic_Fines.yaml
```

Create a YAML file with the following structure:

```yaml
experiment_name: my_experiment

dataset:
    type: outcome # outcome, next_activity, or remaining_time
    dataset_name: Traffic_Fines
    dataset_folder: datasets/raw
    labels_folder: datasets/labels/binary
    train_ratio: 0.8
    min_prefix: 3
    max_prefix: 10

bucketer:
    type: prefix_length # none or prefix_length
    case_id_col: prefix_id

transformer:
    type: aggregate # aggregate, last_state, or index_based
    case_id_col: prefix_id
    cat_cols:
        - concept:name
    num_cols:
        - time_since_last_event
        - time_since_start

model:
    type: xgboost # logistic_regression, random_forest, xgboost, or svm
    params:
        n_estimators: 100
        max_depth: 5
        learning_rate: 0.1

output:
    folder: outputs
```

### 2. Run the Experiment

```bash
# Single experiment
python experiments/run_train.py conf/experiments/xgboost/Traffic_Fines.yaml

# With custom experiment name
python experiments/run_train.py conf/experiments/xgboost/Traffic_Fines.yaml --name my_custom_name

# Run all experiments for a model (PowerShell)
Get-ChildItem conf/experiments/xgboost/*.yaml | ForEach-Object { python experiments/run_train.py $_.FullName }

# Run all experiments using batch runner
python run_batch_experiments.py --model xgboost

# Run specific dataset across all models
python run_batch_experiments.py --dataset Traffic_Fines
```

### 3. View Results

Results are saved to **`outputs/{experiment_name}_{timestamp}.txt`**

Example output files:

- `outputs/xgb_traffic_fines_baseline_2026-02-12_14-30-45.txt`
- `outputs/rf_helpdesk_baseline_2026-02-12_15-22-10.txt`

Each file contains:

- Configuration used
- Classification report (precision, recall, f1-score)
- Summary metrics (accuracy, macro/weighted F1)
