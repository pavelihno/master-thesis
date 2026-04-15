import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import yaml


def get_config_hash(config: dict, length: int = 8) -> str:
    stable_config = {
        key: config[key] for key in ('model', 'transformer', 'dataset') if key in config
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


def prepare_results_frame(config: dict, results_df):
    prepared_df = results_df.copy()
    prepared_df['dataset_name'] = config.get('dataset', {}).get('dataset_name')
    prepared_df['model'] = config.get('model', {}).get('type')
    return prepared_df


def build_run_summary(
    *,
    run_id: str,
    config_path: str,
    timestamp: str,
    config_hash: str,
    config: dict,
    results_df,
    last_row,
    elapsed_seconds: float,
    output_dir: str | None,
) -> dict:
    model_params = config.get('model', {}).get('params', {})
    drift_detector = model_params.get('drift_detector', {})
    warning_detector = model_params.get('warning_detector', {})
    safe_params = make_json_safe(model_params)

    n_traces = int(results_df['trace_id'].nunique()) if 'trace_id' in results_df else 0
    n_events = int(results_df['event_n'].max()) if 'event_n' in results_df else 0
    n_preds = int(results_df['n_pred'].max()) if 'n_pred' in results_df else 0

    return {
        'run_id': run_id,
        'config_path': str(config_path),
        'timestamp': timestamp,
        'config_hash': config_hash,
        'task_type': config.get('task', {}).get('type'),
        'task_mode': config.get('task', {}).get('mode'),
        'dataset_name': config.get('dataset', {}).get('dataset_name'),
        'dataset_path': config.get('dataset', {}).get('dataset_path'),
        'feature_encoding': config.get('transformer', {}).get('type'),
        'transformer_type': config.get('transformer', {}).get('type'),
        'model_type': config.get('model', {}).get('type'),
        'drift_detector_type': drift_detector.get('type'),
        'warning_detector_type': warning_detector.get('type'),
        'model_params': safe_params,
        'model_params_json': json.dumps(safe_params, sort_keys=True),
        'n_preds': n_preds,
        'n_traces': n_traces,
        'n_events': n_events,
        'n_pred': n_preds,
        'accuracy': float(last_row.get('accuracy', 0)),
        'macro_f1': float(last_row.get('macro_f1', 0)),
        'loss': float(last_row.get('loss', 0) or 0),
        'n_drifts': int(last_row.get('n_drifts', 0)),
        'time_s': float(elapsed_seconds),
        'output_dir': output_dir,
    }


def write_results(
    output_dir: Path,
    run_id: str,
    config_path: str,
    timestamp: str,
    config: dict,
    results_df,
    elapsed_seconds: float,
) -> None:
    results_path = output_dir / 'results.csv'

    results_df.to_csv(results_path, index=False)

    last_row = results_df.iloc[-1]
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as file:
        n_traces = (
            int(results_df['trace_id'].nunique()) if 'trace_id' in results_df else 0
        )
        n_events = int(results_df['event_n'].max()) if 'event_n' in results_df else 0
        n_preds = int(results_df['n_pred'].max()) if 'n_pred' in results_df else 0

        file.write(f'Run ID: {run_id}\n')
        file.write(f'Config     : {config_path}\n')
        file.write(f'Timestamp  : {timestamp}\n')
        file.write('=' * 60 + '\n\n')
        file.write('Configuration:\n')
        file.write('-' * 60 + '\n')
        file.write(yaml.safe_dump(config, sort_keys=False))
        file.write('\n')
        file.write('Summary Metrics:\n')
        file.write('-' * 60 + '\n')
        file.write(f'n_preds={n_preds}\n')
        file.write(f'n_traces={n_traces}\n')
        file.write(f'n_events={n_events}\n')
        file.write(f'n_drifts={int(last_row.get("n_drifts", 0))}\n')
        file.write(f'accuracy={last_row.get("accuracy", 0):.4f}\n')
        file.write(f'macro_f1={last_row.get("macro_f1", 0):.4f}\n')
        file.write(f'time={elapsed_seconds:.2f}s\n')

    print(f'Results -> {results_path}')
    print(f'Summary -> {summary_path}')


def save_model(output_dir: Path, model) -> None:
    if hasattr(model, 'save_checkpoint'):
        model_path = output_dir / 'model.pt'
        model.save_checkpoint(model_path)
    else:
        model_path = output_dir / 'model.pkl'
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
    print(f'Model -> {model_path}')


def load_saved_model(model_path: Path, model=None):
    if hasattr(model, 'load_checkpoint'):
        model.load_checkpoint(model_path)
        return model

    with open(model_path, 'rb') as file:
        return pickle.load(file)
