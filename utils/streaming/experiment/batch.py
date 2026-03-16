import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


def find_config_files(
    base_dir: str,
    model_name: str | None = None,
    dataset_name: str | None = None,
) -> list[Path]:
    base_path = Path(base_dir)

    if model_name:
        model_dir = base_path / model_name
        config_files = sorted(model_dir.glob('*.yaml')) if model_dir.exists() else []
    else:
        config_files = sorted(
            file_path
            for child in base_path.iterdir()
            if child.is_dir() and not child.name.startswith('_')
            for file_path in child.glob('*.yaml')
        )

    if dataset_name:
        dataset_filter = dataset_name.lower()
        config_files = [
            file_path
            for file_path in config_files
            if dataset_filter in file_path.stem.lower()
        ]

    return config_files


def run_config_process(
    config_path: Path,
    runner_path: Path,
    python_executable: str,
    metrics_dir: Path,
    save_artifacts: bool,
) -> dict:
    path_key = hashlib.sha1(str(config_path).encode()).hexdigest()[:10]
    metrics_path = metrics_dir / f'{config_path.stem}_{path_key}.json'
    command = [
        python_executable,
        str(runner_path),
        str(config_path),
        '--metrics-json',
        str(metrics_path),
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
        'stdout_tail': '\n'.join(completed.stdout.splitlines()[-10:]),
        'stderr_tail': '\n'.join(completed.stderr.splitlines()[-10:]),
    }


def save_comparison_reports(
    results: list[dict],
    report_dir: Path,
    report_name: str,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for result in results:
        metrics = result.get('metrics') or {}
        rows.append(
            {
                'config_path': result['config_path'],
                'status': 'ok' if result['ok'] else 'failed',
                'returncode': result['returncode'],
                'error': metrics.get('error') if isinstance(metrics, dict) else None,
                **(metrics if isinstance(metrics, dict) else {}),
            }
        )

    results_df = pd.DataFrame(rows)

    sort_columns = [
        column
        for column in ['dataset_name', 'feature_encoding', 'macro_f1']
        if column in results_df.columns
    ]
    if 'macro_f1' in results_df.columns:
        ascending = [column != 'macro_f1' for column in sort_columns]
        results_df = results_df.sort_values(sort_columns, ascending=ascending)

    raw_report = report_dir / f'{report_name}_comparison_raw.csv'
    results_df.to_csv(raw_report, index=False)

    summary_columns = [
        'dataset_name',
        'feature_encoding',
        'model_type',
        'accuracy',
        'macro_f1',
        'n_drifts',
        'n_pred',
        'time_s',
        'run_id',
        'config_path',
        'status',
    ]
    summary_df = results_df[
        [column for column in summary_columns if column in results_df.columns]
    ].copy()
    summary_report = report_dir / f'{report_name}_comparison_summary.csv'
    summary_df.to_csv(summary_report, index=False)

    if {'dataset_name', 'feature_encoding', 'macro_f1'}.issubset(summary_df.columns):
        ranking_df = summary_df[summary_df['status'] == 'ok'].copy()
        ranking_df['rank'] = (
            ranking_df.groupby(['dataset_name', 'feature_encoding'])['macro_f1']
            .rank(method='dense', ascending=False)
            .astype(int)
        )
        ranking_df['is_best'] = ranking_df['rank'] == 1
        ranking_df.to_csv(
            report_dir / f'{report_name}_comparison_ranked.csv',
            index=False,
        )

    print(f'Comparison CSV (raw)     -> {raw_report}')
    print(f'Comparison CSV (summary) -> {summary_report}')
    return raw_report


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
