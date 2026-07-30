import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from utils.experiment.naming import read_run_info


def run_config_process(
    config_path: Path,
    runner_path: Path,
    python_executable: str,
    metrics_dir: Path,
    results_dir: Path,
    save_artifacts: bool = True,
    device: str | None = None,
    extra_args: list[str] | None = None,
    run_suffix: str | None = None,
) -> dict:
    path_key = hashlib.sha1(str(config_path).encode()).hexdigest()[:10]
    suffix = f'_{run_suffix}' if run_suffix else ''
    metrics_path = metrics_dir / f'{config_path.stem}_{path_key}{suffix}.json'
    results_path = results_dir / f'{config_path.stem}_{path_key}{suffix}.csv'
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
    if extra_args:
        command.extend(extra_args)
    if device is not None:
        command.extend(['--device', device])

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
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    prediction_frames = []
    for result in results:
        metrics = result.get('metrics') or {}
        config_path = Path(result['config_path'])

        cfg_dataset, cfg_model_name = read_run_info(config_path)
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
            if (
                'dataset_name' not in prediction_df.columns
                and 'dataset' in prediction_df.columns
            ):
                prediction_df = prediction_df.rename(
                    columns={'dataset': 'dataset_name'}
                )
            if 'dataset_name' not in prediction_df.columns:
                prediction_df['dataset_name'] = dataset_name
            if 'model' not in prediction_df.columns:
                prediction_df['model'] = model_name
            prediction_df.insert(0, 'config_path', str(config_path))
            prediction_frames.append(prediction_df)
            prediction_path.unlink(missing_ok=True)

    results_df = pd.DataFrame(summary_rows)

    summary_columns = [
        'run_id',
        'config_path',
        'config_hash',
        'status',
        'error',
        'task_type',
        'task_mode',
        'time_target',
        'dataset_name',
        'model_name',
        'n_preds',
        'n_traces',
        'n_events',
        'accuracy',
        'macro_f1',
        'mae',
        'rmse',
        'loss',
        'n_drifts',
        'time_s',
    ]
    results_df = results_df[
        [column for column in summary_columns if column in results_df.columns]
    ]

    sort_columns = [
        column
        for column in ['dataset_name', 'model_name', 'feature_encoding']
        if column in results_df.columns
    ]

    results_df = results_df.sort_values(sort_columns)

    summary_path = report_dir / 'summary.csv'
    results_df.to_csv(summary_path, index=False)

    prediction_path = report_dir / 'predictions.csv'
    if prediction_frames:
        predictions_df = pd.concat(prediction_frames, ignore_index=True)
    else:
        predictions_df = pd.DataFrame()
    predictions_df.to_csv(prediction_path, index=False)

    print(f'\n{"=" * 60}')
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
