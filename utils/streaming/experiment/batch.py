import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


def _load_config(config_path: Path) -> dict | None:
    try:
        import yaml

        with open(config_path, encoding='utf-8') as file:
            return yaml.safe_load(file) or {}
    except Exception:
        return None


def _read_config_metadata(config_path: Path) -> tuple[str | None, str | None]:
    config = _load_config(config_path) or {}
    dataset_name = config.get('dataset', {}).get('dataset_name')
    model_name = config.get('model', {}).get('type')
    return dataset_name, model_name


def find_config_files(
    base_dir: str,
    model_name: str | None = None,
    dataset_name: str | None = None,
    exclude_hyperparam_search: bool = False,
) -> list[Path]:
    base_path = Path(base_dir)

    if not base_path.exists():
        return []

    config_files = sorted(
        file_path
        for file_path in base_path.rglob('*.yaml')
        if not any(part.startswith('_') for part in file_path.parts)
    )

    if model_name or dataset_name:
        model_filter = model_name.lower() if model_name else None
        dataset_filter = dataset_name.lower() if dataset_name else None

        filtered_files: list[Path] = []
        for file_path in config_files:
            config = _load_config(file_path) or {}
            config_model = str(config.get('model', {}).get('type', '')).lower()
            config_dataset = str(
                config.get('dataset', {}).get('dataset_name', '')
            ).lower()

            if model_filter and model_filter not in config_model:
                continue
            if dataset_filter and dataset_filter not in config_dataset:
                continue
            filtered_files.append(file_path)

        config_files = filtered_files

    if exclude_hyperparam_search:
        filtered_files = []
        for file_path in config_files:
            config = _load_config(file_path) or {}
            if config.get('hyperparam_search', False):
                continue
            filtered_files.append(file_path)
        config_files = filtered_files

    return config_files


def run_config_process(
    config_path: Path,
    runner_path: Path,
    python_executable: str,
    metrics_dir: Path,
    results_dir: Path,
    save_artifacts: bool,
) -> dict:
    path_key = hashlib.sha1(str(config_path).encode()).hexdigest()[:10]
    metrics_path = metrics_dir / f'{config_path.stem}_{path_key}.json'
    results_path = results_dir / f'{config_path.stem}_{path_key}.csv'
    command = [
        python_executable,
        str(runner_path),
        str(config_path),
        '--metrics-json',
        str(metrics_path),
        '--results-csv',
        str(results_path),
    ]
    if not save_artifacts:
        command.append('--no-save-artifacts')

    completed = subprocess.run(command, capture_output=True, text=True)

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, encoding='utf-8') as file:
            metrics = json.load(file)
        metrics_path.unlink(missing_ok=True)

    return {
        'config_path': str(config_path),
        'ok': completed.returncode == 0,
        'returncode': completed.returncode,
        'metrics': metrics,
        'results_csv_path': str(results_path) if results_path.exists() else None,
        'stdout_tail': '\n'.join(completed.stdout.splitlines()[-10:]),
        'stderr_tail': '\n'.join(completed.stderr.splitlines()[-10:]),
    }


def save_comparison_reports(
    results: list[dict],
    report_dir: Path,
    report_name: str,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    prediction_frames = []
    for result in results:
        metrics = result.get('metrics') or {}
        config_path = Path(result['config_path'])

        cfg_dataset, cfg_model_name = _read_config_metadata(config_path)
        dataset_name = cfg_dataset
        model_name = cfg_model_name

        row = {
            'config_path': str(config_path),
            'status': 'ok' if result['ok'] else 'failed',
            'returncode': result['returncode'],
            'error': metrics.get('error') if isinstance(metrics, dict) else None,
        }
        if isinstance(metrics, dict):
            row.update(metrics)
        row['dataset_name'] = dataset_name
        row['model_name'] = model_name
        summary_rows.append(row)

        results_csv_path = result.get('results_csv_path')
        if result['ok'] and results_csv_path:
            prediction_path = Path(results_csv_path)
            prediction_df = pd.read_csv(prediction_path)
            if 'dataset_name' not in prediction_df.columns:
                prediction_df['dataset_name'] = dataset_name
            if 'model' not in prediction_df.columns:
                prediction_df['model'] = model_name
            prediction_df.insert(0, 'config_path', str(config_path))
            prediction_frames.append(prediction_df)
            prediction_path.unlink(missing_ok=True)

    results_df = pd.DataFrame(summary_rows)

    sort_columns = [
        column
        for column in ['dataset_name', 'model_name', 'feature_encoding', 'macro_f1']
        if column in results_df.columns
    ]
    if 'macro_f1' in results_df.columns:
        ascending = [column != 'macro_f1' for column in sort_columns]
        results_df = results_df.sort_values(sort_columns, ascending=ascending)

    summary_filename = (
        f'{report_name}_comparison_summary.csv'
        if not report_name.lower().endswith('.csv')
        else report_name
    )
    summary_path = report_dir / summary_filename
    results_df.to_csv(summary_path, index=False)

    prediction_path = report_dir / f'{report_name}_predictions.csv'
    if prediction_frames:
        predictions_df = pd.concat(prediction_frames, ignore_index=True)
    else:
        predictions_df = pd.DataFrame()
    predictions_df.to_csv(prediction_path, index=False)

    print(f'Comparison CSV (summary)    -> {summary_path}')
    print(f'Comparison CSV (predictions) -> {prediction_path}')
    return summary_path


def print_batch_summary(results: list[dict]) -> int:
    success_count = sum(1 for result in results if result['ok'])
    failure_count = len(results) - success_count

    print(f'\n{"=" * 60}')
    print('BATCH SUMMARY')
    print('=' * 60)
    print(f'Total      : {len(results)}')
    print(f'Successful : {success_count}')
    print(f'Failed     : {failure_count}')

    if failure_count:
        print('\nFailed experiments:')
        for result in results:
            if not result['ok']:
                print(f'  - {result["config_path"]}')
        return 1

    return 0
