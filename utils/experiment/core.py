import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from utils.base.datasets import BaseLogDataset, OutcomeDataset


def select_device(device_name: str | None) -> torch.device | None:
    if device_name is None:
        return None

    requested = device_name.lower()
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but not available. Falling back to CPU.')
        return torch.device('cpu')

    return torch.device(requested)


def get_config_hash(config: dict, length: int = 8) -> str:
    stable_config = {
        key: config[key]
        for key in ('model', 'transformer', 'dataset', 'bucketer')
        if key in config
    }
    payload = json.dumps(stable_config, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def make_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return str(value)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def compute_bucket_statistics(prefixes_df, y_true, y_pred, bucketer):
    """Compute per-bucket performance statistics."""

    bucket_ids_per_prefix = bucketer.predict(prefixes_df)
    unique_prefix_ids = prefixes_df[bucketer.case_id_col].unique()
    prefix_to_bucket = dict(zip(unique_prefix_ids, bucket_ids_per_prefix, strict=True))
    bucket_ids = prefixes_df[bucketer.case_id_col].map(prefix_to_bucket).values

    unique_buckets = np.unique(bucket_ids)
    bucket_stats = []

    for bucket_id in sorted(unique_buckets):
        mask = bucket_ids == bucket_id
        bucket_y_true = y_true[mask]
        bucket_y_pred = y_pred[mask]

        # Get unique prefixes in this bucket
        bucket_prefixes = prefixes_df[mask]
        n_prefixes = len(bucket_prefixes[bucketer.case_id_col].unique())
        n_events = len(bucket_prefixes)

        # Compute metrics
        accuracy = accuracy_score(bucket_y_true, bucket_y_pred)
        precision = precision_score(
            bucket_y_true, bucket_y_pred, average='weighted', zero_division=0
        )
        recall = recall_score(
            bucket_y_true, bucket_y_pred, average='weighted', zero_division=0
        )
        f1_macro = f1_score(
            bucket_y_true, bucket_y_pred, average='macro', zero_division=0
        )

        bucket_stats.append(
            {
                'bucket': int(bucket_id),
                'prefixes': n_prefixes,
                'events': n_events,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_macro': f1_macro,
            }
        )

    return pd.DataFrame(bucket_stats)


def load_and_prepare_data(dataset: BaseLogDataset):
    """Load and prepare dataset for training/evaluation."""

    if isinstance(dataset, OutcomeDataset):
        dataset.filter_by_labels()

    train_df, test_df = dataset.train_test_split()

    train_prefixes = dataset.get_prefixes(train_df)
    test_prefixes = dataset.get_prefixes(test_df)

    y_train = dataset.get_labels(train_prefixes)
    y_test = dataset.get_labels(test_prefixes)

    return train_prefixes, test_prefixes, y_train, y_test


def ensure_output_dir(config):
    """Ensure output directory exists."""
    output_folder = Path(config.get('output', {}).get('folder', 'outputs'))
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder
