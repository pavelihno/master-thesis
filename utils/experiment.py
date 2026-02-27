from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def compute_bucket_statistics(prefixes_df, y_true, y_pred, bucketer):
    """Compute per-bucket performance statistics."""
    # Get bucket assignments for test data
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
        f1_weighted = f1_score(
            bucket_y_true, bucket_y_pred, average='weighted', zero_division=0
        )
        f1_macro = f1_score(
            bucket_y_true, bucket_y_pred, average='macro', zero_division=0
        )

        bucket_stats.append(
            {
                'Bucket': int(bucket_id),
                'Prefixes': n_prefixes,
                'Events': n_events,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Weighted': f1_weighted,
                'F1_Macro': f1_macro,
            }
        )

    return pd.DataFrame(bucket_stats)


def load_and_prepare_data(dataset, config):
    """Load and prepare dataset for training/evaluation."""
    print('\n[2/6] Loading and preprocessing data...')
    dataset.load_and_preprocess()

    if config['dataset']['type'] == 'outcome':
        dataset.filter_by_labels()

    print('\n[3/6] Splitting data...')
    train_df, test_df = dataset.train_test_split()

    print('\n[4/6] Generating prefixes...')
    train_prefixes = dataset.get_prefixes(train_df)
    test_prefixes = dataset.get_prefixes(test_df)

    print('\n[5/6] Preparing labels...')
    y_train = dataset.prepare_labels(train_prefixes)
    y_test = dataset.prepare_labels(test_prefixes)

    from utils.stats import print_class_balance

    print_class_balance(y_train, dataset_names=[dataset.dataset_name, 'Train'])
    print_class_balance(y_test, dataset_names=[dataset.dataset_name, 'Test'])

    return train_prefixes, test_prefixes, y_train, y_test


def evaluate_model(pipeline, test_prefixes, y_test, dataset, bucketer, verbose=True):
    """Evaluate model and compute metrics."""
    if verbose:
        print('Evaluating...')

    y_pred = pipeline.predict(test_prefixes)
    y_pred_decoded = dataset.decode_labels(y_pred)
    y_test_decoded = dataset.decode_labels(y_test)

    # Overall metrics
    report = classification_report(
        y_test_decoded, y_pred_decoded, output_dict=True, zero_division=0
    )
    report_str = classification_report(y_test_decoded, y_pred_decoded, zero_division=0)

    # Per-bucket statistics
    bucket_stats_df = compute_bucket_statistics(test_prefixes, y_test, y_pred, bucketer)

    if verbose:
        print('\n' + '=' * 80)
        print('RESULTS')
        print('=' * 80)
        print(report_str)

        print('\n' + '=' * 80)
        print('PER-BUCKET STATISTICS')
        print('=' * 80)
        print(bucket_stats_df.to_string(index=False))
        print()

    return {
        'y_pred': y_pred,
        'y_pred_decoded': y_pred_decoded,
        'y_test_decoded': y_test_decoded,
        'report': report,
        'report_str': report_str,
        'bucket_stats': bucket_stats_df,
        'accuracy': report['accuracy'],
        'f1_macro': report['macro avg']['f1-score'],
        'f1_weighted': report['weighted avg']['f1-score'],
    }


def save_results(
    output_file,
    experiment_name,
    config_path,
    timestamp,
    config,
    evaluation_results,
):
    """Save experiment results to file."""
    import yaml

    with open(output_file, 'w') as f:
        f.write(f'Experiment: {experiment_name}\n')
        f.write(f'Config: {config_path}\n')
        f.write(f'Timestamp: {timestamp}\n')
        f.write('=' * 80 + '\n\n')

        f.write('Configuration:\n')
        f.write('-' * 80 + '\n')
        f.write(yaml.dump(config, default_flow_style=False))
        f.write('\n')

        f.write('Results:\n')
        f.write('-' * 80 + '\n')
        f.write(evaluation_results['report_str'])
        f.write('\n')

        f.write('\nSummary Metrics:\n')
        f.write('-' * 80 + '\n')
        f.write(f'Accuracy: {evaluation_results["accuracy"]:.4f}\n')
        f.write(f'Macro Avg F1-Score: {evaluation_results["f1_macro"]:.4f}\n')
        f.write(f'Weighted Avg F1-Score: {evaluation_results["f1_weighted"]:.4f}\n')

        f.write('\n\nPer-Bucket Statistics:\n')
        f.write('-' * 80 + '\n')
        f.write(evaluation_results['bucket_stats'].to_string(index=False))
        f.write('\n')


def ensure_output_dir(config):
    """Ensure output directory exists."""
    output_folder = Path(config.get('output', {}).get('folder', 'outputs'))
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder
