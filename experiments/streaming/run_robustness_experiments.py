import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from river import metrics as river_metrics

from utils.experiment.batch import find_config_files
from utils.experiment.core import select_device
from utils.experiment.naming import (
    build_output_folder_name,
    get_timestamp,
    read_run_info,
)
from utils.streaming.experiment.batch import (
    print_batch_summary,
    run_config_process,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

START_EVENT_JSON_PATHS = {
    'next_activity': PROJECT_ROOT / 'datasets/results/next_activity/start_events.json'
}

CLASSIFICATION_METRIC_FACTORIES = {
    'macro_f1': river_metrics.MacroF1,
    'accuracy': river_metrics.Accuracy,
}

REGRESSION_METRIC_FACTORIES = {
    'mae': river_metrics.MAE,
    'rmse': river_metrics.RMSE,
    'mape': river_metrics.MAPE,
}


def get_metric_factories(task: str = 'next_activity') -> dict:
    if task == 'next_activity':
        return CLASSIFICATION_METRIC_FACTORIES
    elif task == 'outcome':
        return CLASSIFICATION_METRIC_FACTORIES
    elif task == 'remaining_time':
        return REGRESSION_METRIC_FACTORIES
    elif task == 'next_activity_time':
        return REGRESSION_METRIC_FACTORIES
    else:
        raise ValueError(f'Unknown task: {task}')


def load_start_events(task: str) -> dict[str, int]:
    start_events_path = START_EVENT_JSON_PATHS.get(task)
    if not start_events_path:
        raise ValueError(f'Unknown task: {task}')
    if not start_events_path.exists():
        raise FileNotFoundError(
            f'Start events JSON file not found: {start_events_path}'
        )

    with start_events_path.open(encoding='utf-8') as handle:
        return {str(key): int(value) for key, value in json.load(handle).items()}


def recompute_running_metrics(
    predictions_df: pd.DataFrame,
    metric_factories: dict = CLASSIFICATION_METRIC_FACTORIES,
    start_event_num: int = 0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Recompute running metrics from a given event onwards."""
    filtered_df = (
        predictions_df[predictions_df['event_n'] >= start_event_num]
        .sort_values('event_n')
        .reset_index(drop=True)
        .copy()
    )

    if filtered_df.empty:
        return filtered_df, {}

    metrics_dict = {name: factory() for name, factory in metric_factories.items()}

    y_trues = filtered_df['y_true'].values
    y_preds = filtered_df['y_pred'].values
    running_values = {name: [] for name in metrics_dict}

    for y_true, y_pred in zip(y_trues, y_preds, strict=True):
        if pd.notna(y_true) and pd.notna(y_pred):
            t_val = int(y_true) if isinstance(y_true, (int, np.integer)) else y_true
            p_val = int(y_pred) if isinstance(y_pred, (int, np.integer)) else y_pred

            for metric in metrics_dict.values():
                metric.update(t_val, p_val)

        for metric_name, metric in metrics_dict.items():
            running_values[metric_name].append(metric.get())

    for metric_name, values in running_values.items():
        filtered_df[metric_name] = values

    final_metrics = {name: metric.get() for name, metric in metrics_dict.items()}
    return filtered_df, final_metrics


def recompute_metrics(
    result_dict: dict,
    start_event: int,
    metric_factories: dict | None = None,
) -> dict[str, float]:
    metric_factories = metric_factories or CLASSIFICATION_METRIC_FACTORIES

    results_csv_path = Path(result_dict.get('results_csv_path'))

    if not results_csv_path.exists():
        raise FileNotFoundError(f'Results CSV file not found: {results_csv_path}')

    df_preds = pd.read_csv(results_csv_path, low_memory=False)

    required_columns = {'event_n', 'y_true', 'y_pred'}
    if not required_columns.issubset(df_preds.columns):
        metrics = result_dict.get('metrics') or {}
        if isinstance(metrics, dict):
            return {
                key: value for key, value in metrics.items() if key in metric_factories
            }
        return {}

    _, final_metrics = recompute_running_metrics(
        df_preds,
        metric_factories=metric_factories,
        start_event_num=start_event,
    )
    return final_metrics


def save_metrics_csv(
    output_dir: Path,
    config_path: Path,
    rows: list[dict],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_name = config_path.stem
    metrics_path = output_dir / f'{config_name}.csv'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        pd.DataFrame().to_csv(metrics_path, index=False)
        return metrics_path

    rows_df = pd.DataFrame(rows)
    rows_df = rows_df.sort_values(['run_idx'], kind='stable')

    rows_df.to_csv(metrics_path, index=False)
    return metrics_path


def run_batch_with_seeds(
    config_files: list[Path],
    runs_per_config: int = 5,
    workers: int = 1,
    device: str | None = None,
    base_seed: int = 42,
    experiment_name: str = 'robustness_experiment',
    task: str = 'next_activity',
) -> list[dict]:
    runner_path = Path(__file__).with_name('run_train.py')
    python_executable = sys.executable

    metrics_dir = Path('experiments/outputs/streaming/.tmp_metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path('experiments/outputs/streaming/.tmp_results')
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_timestamp()

    output_dir_name = build_output_folder_name(
        custom_name=experiment_name, timestamp=timestamp
    )

    output_dir = Path(
        f'experiments/outputs/streaming/robustness_reports/{output_dir_name}'
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    start_events = load_start_events(task)

    metric_factories = get_metric_factories(task)

    rng = random.Random(base_seed)
    tasks = []
    for config_path in config_files:
        for run_idx in range(runs_per_config):
            seed = rng.randint(1000, 999999)
            tasks.append((config_path, run_idx, seed))

    print(
        'Found '
        f'{len(config_files)} config(s). Total runs: {len(tasks)} '
        f'({runs_per_config} runs/config).'
    )

    results: list[dict] = []
    rows_by_config: dict[Path, list[dict]] = {}

    def _execute_single_run(config_path: Path, run_idx: int, seed: int) -> dict:
        dataset_name, model_name = read_run_info(config_path)
        extra_args = ['--seed', str(seed)]
        run_suffix = f'seed{seed}_run{run_idx}'

        start_event = start_events.get(dataset_name, 0)

        res = run_config_process(
            config_path=config_path,
            runner_path=runner_path,
            python_executable=python_executable,
            metrics_dir=metrics_dir,
            results_dir=results_dir,
            device=device,
            save_artifacts=False,
            extra_args=extra_args,
            run_suffix=run_suffix,
        )
        res['seed'] = seed
        res['run_idx'] = run_idx
        res['dataset_name'] = dataset_name or 'unknown'
        res['model_name'] = model_name or 'unknown'

        recomputed_metrics = recompute_metrics(
            res,
            start_event=start_event,
            metric_factories=metric_factories,
        )
        res['recomputed_metrics'] = recomputed_metrics

        row = {
            'run_idx': run_idx,
            'seed': seed,
            'ok': res['ok'],
            'returncode': res['returncode'],
        }
        row.update(recomputed_metrics)
        res['metrics_row'] = row
        return res

    if workers == 1:
        for config_path, run_idx, seed in tasks:
            dataset_name, model_name = read_run_info(config_path)
            print(f'\n{"=" * 60}')
            print(
                'Running '
                f'[{run_idx + 1}/{runs_per_config}]: {config_path.name} '
                f'| Seed: {seed}'
            )
            print('=' * 60)
            res = _execute_single_run(config_path, run_idx, seed)
            results.append(res)
            rows_by_config.setdefault(config_path, []).append(res['metrics_row'])
    else:
        print(f'Launching up to {workers} workers in parallel.\n')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_execute_single_run, cfg, r_idx, s): (cfg, r_idx, s)
                for cfg, r_idx, s in tasks
            }
            for future in as_completed(futures):
                res = future.result()
                cfg, r_idx, s = futures[future]
                status = 'OK' if res['ok'] else 'FAILED'
                print(f'[{status}] {cfg.name} | Run {r_idx + 1}/{runs_per_config}')
                results.append(res)
                rows_by_config.setdefault(cfg, []).append(res['metrics_row'])

    print()
    metrics_paths_by_config = {}
    for config_path, rows in rows_by_config.items():
        metrics_path = save_metrics_csv(output_dir, config_path, rows)
        metrics_paths_by_config[config_path] = metrics_path
        print(f'Metrics CSV -> {metrics_path}')

    summary_path = output_dir / 'summary.txt'
    total_runs = len(tasks)
    success_count = sum(1 for r in results if r.get('ok'))
    with summary_path.open('w', encoding='utf-8') as fh:
        fh.write(f'Experiment: {experiment_name}\n')
        fh.write(f'Timestamp: {timestamp}\n')
        fh.write(f'Task: {task}\n')
        fh.write(f'Runs per config: {runs_per_config}\n')
        fh.write(f'Base seed: {base_seed}\n')
        fh.write(f'Total runs: {total_runs}\n')
        fh.write(f'Successful runs: {success_count}\n')
        fh.write('\nConfigs used:\n')
        for cfg in config_files:
            fh.write(f'- {cfg}\n')
        fh.write('\nProduced CSVs:\n')
        for cfg, path in metrics_paths_by_config.items():
            fh.write(f'- {cfg.stem}: {path}\n')

    print(f'Summary -> {summary_path}')

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run multi-seed streaming experiments and save raw metric CSV tables.'
        )
    )

    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--base-path', type=str, default='conf/experiments/streaming')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument(
        '--runs', type=int, default=5, help='Number of random seeds per config.'
    )
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument(
        '--task', type=str, default='next_activity', help='Prediction task.'
    )
    parser.add_argument(
        '--start-event',
        type=int,
        default=None,
        help='Optional global start event override.',
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='robustness_experiment',
        help='Name of the experiment.',
    )
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    base_path = Path(args.base_path)

    config_files = find_config_files(
        base_path,
        model_name=args.model,
        dataset_name=args.dataset,
        exclude_hyperparam_search=True,
    )

    if not config_files:
        print(f'No configs found in {base_path}.')
        sys.exit(1)

    results = run_batch_with_seeds(
        config_files,
        runs_per_config=args.runs,
        workers=args.workers,
        device=device.type if device is not None else None,
        base_seed=args.base_seed,
        experiment_name=args.experiment_name,
        task=args.task,
    )

    exit_code = print_batch_summary(results)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
