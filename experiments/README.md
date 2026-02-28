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
    folder: experiments/outputs
    save_model: true
    model_folder: experiments/models
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

Results are saved to **`experiments/outputs/{experiment_name}_{timestamp}.txt`**

Example output files:

- `experiments/outputs/xgb_traffic_fines_baseline_2026-02-12_14-30-45.txt`
- `experiments/outputs/rf_helpdesk_baseline_2026-02-12_15-22-10.txt`

Each file contains:

- Configuration used
- Classification report (precision, recall, f1-score)
- Summary metrics (accuracy, macro/weighted F1)
- Per-bucket statistics (performance metrics for each prefix length)

### 4. Save Models

If `save_model: true` in config, trained models are saved to **`experiments/models/{experiment_name}_{timestamp}/`**

Model structure:

```
experiments/models/xgb_traffic_fines_baseline_2026-02-12_14-30-45/
├── metadata.json         # Model metadata and configuration
├── bucketer.pkl          # Trained bucketer
└── bucket_*/             # Separate model for each bucket
    ├── model.pkl         # Trained model (XGBoost, RF, etc.)
    └── transformer.pkl   # Fitted transformer
```

**Load model in code:**

```python
from utils.pipelines import ProcessPredictorPipeline

# Load saved pipeline
pipeline = ProcessPredictorPipeline.load('experiments/models/xgb_traffic_fines_baseline_2026-02-12_14-30-45')
```

---

## Streaming Experiments

### 1. Create a YAML Configuration File

Configuration files live in `conf/experiments/streaming/<model_type>/<Dataset_Name>.yaml`.

Example structure:

```
conf/experiments/streaming/
└── srp/
    └── BPIC_20_DD.yaml
```

Create a YAML file with the following structure:

```yaml
experiment_name: srp_cf_bpic20dd

dataset:
    dataset_name: BPIC_20_DD
    dataset_path: datasets/raw/BPIC_20_DD.xes

# Encoding strategy for trace prefixes
transformer:
    type: cf # cf | data | index | dim
    max_events: null # max prefix positions to encode (index and dim only)

# Online classifier
model:
    type: srp # srp | arf | aht
    params:
        n_models: 100
        subspace_size: 0.6
        lam: 6
        drift_detector: 1.0e-5 # ADWIN delta – converted automatically
        warning_detector: 1.0e-4
        seed: 42

output:
    folder: experiments/outputs/streaming/BPIC_20_DD
```

### 2. Run the Experiment

```bash
# Single experiment
python experiments/streaming/run_train.py conf/experiments/streaming/srp/BPIC_20_DD.yaml

# With custom experiment name
python experiments/streaming/run_train.py conf/experiments/streaming/srp/BPIC_20_DD.yaml --name my_custom_name

# Run all configs discovered sequentially
python experiments/streaming/run_experiments.py

# Run all configs in parallel (4 workers)
python experiments/streaming/run_experiments.py --workers 4

# Run all configs using all available CPUs
python experiments/streaming/run_experiments.py --workers -1

# Run all configs for a specific model type
python experiments/streaming/run_experiments.py --model srp

# Run all configs for a specific dataset
python experiments/streaming/run_experiments.py --dataset BPIC_20_DD
```

### 3. View Results

Results are saved to **`experiments/outputs/streaming/<dataset_name>/`**

Each run produces two files:

- `{experiment_name}_{timestamp}.csv` — full per-prediction log (one row per prediction with `n_pred`, `y_true`, `y_pred`, `correct`, `accuracy`, `macro_f1`, `trace_id`, `prefix_len`, `event_time`, `time_s`)
- `{experiment_name}_{timestamp}.txt` — summary with configuration and final metrics

Example:

```
experiments/outputs/streaming/BPIC_20_DD/
├── srp_cf_bpic20dd_2026-02-27_14-30-45.csv
└── srp_cf_bpic20dd_2026-02-27_14-30-45.txt
```

---

## Hyperparameter Search

The framework includes a hyperparameter search tool that efficiently searches over parameter grids without re-processing data.

### 1. Create Hyperparameter Search Config

Create a YAML config with `param_grid` instead of `params`:

```yaml
experiment_name: xgb_hypersearch

dataset:
    type: outcome
    dataset_name: BPIC_12_O
    dataset_folder: datasets/raw
    labels_folder: datasets/labels/binary
    train_ratio: 0.8
    min_prefix: 3
    max_prefix: 10

bucketer:
    type: prefix_length
    case_id_col: prefix_id

transformer:
    type: aggregate
    case_id_col: prefix_id
    cat_cols:
        - concept:name
    num_cols:
        - time_since_last_event
        - time_since_start

model:
    type: xgboost

    # Use param_grid instead of params
    # Specify lists for parameters to search over
    # Specify single values for fixed parameters
    param_grid:
        n_estimators: [100, 200, 300]
        max_depth: [3, 5, 7]
        learning_rate: [0.01, 0.1, 0.2]
        subsample: [0.8, 1.0]
        colsample_bytree: 1.0 # Fixed
        random_state: 42 # Fixed

output:
    folder: experiments/outputs
    save_best_model: true # Save best model after search
    model_folder: experiments/models
```

### 2. Run Hyperparameter Search

```bash
# Run hyperparameter search
python experiments/run_hyperparam_search.py conf/experiments/xgboost/BPIC_12_O_hypersearch.yaml

# With custom name
python experiments/run_hyperparam_search.py conf/experiments/xgboost/BPIC_12_O_hypersearch.yaml --name my_search
```

### 3. View Search Results

Results are saved to **`experiments/outputs/{experiment_name}_hypersearch_{timestamp}.txt`**
