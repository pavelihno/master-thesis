import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils.experiment.batch import find_config_files
from utils.experiment.core import select_device
from utils.experiment.naming import build_output_folder_name, read_run_info
from utils.streaming.experiment.batch import (
    print_batch_summary,
    run_config_process,
    save_comparison_reports,
)


def run_batch(
    config_files: list[Path],
    workers: int = 1,
    save_artifacts: bool = False,
    device: str | None = None,
) -> list[dict]:
    runner_path = Path(__file__).with_name('run_train.py')
    python_executable = sys.executable

    print(f'Found {len(config_files)} config(s).')

    if workers == -1:
        import os

        workers = os.cpu_count() or 1

    metrics_dir = Path('experiments/outputs/streaming/.tmp_metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path('experiments/outputs/streaming/.tmp_results')
    results_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    if workers == 1:
        for config_path in config_files:
            dataset_name, model_name = read_run_info(config_path)
            print(f'\n{"=" * 60}')
            print(
                f'Running: {config_path}\n'
                f'Dataset: {dataset_name or "unknown"}\n'
                f'Model: {model_name or "unknown"}'
            )
            print('=' * 60)
            result = run_config_process(
                config_path,
                runner_path,
                python_executable,
                metrics_dir,
                results_dir,
                save_artifacts,
                device,
            )
            status = 'OK' if result['ok'] else 'FAILED'
            print(f'[{status}] {config_path}')
            results.append(result)
            if not result['ok'] and result['stderr_tail']:
                print(result['stderr_tail'])
    else:
        print(f'Launching up to {workers} experiments in parallel.\n')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for config_path in config_files:
                dataset_name, model_name = read_run_info(config_path)
                print(
                    f'Running: {config_path}\n'
                    f'Dataset: {dataset_name or "unknown"}\n'
                    f'Model: {model_name or "unknown"}'
                )
                future = pool.submit(
                    run_config_process,
                    config_path,
                    runner_path,
                    python_executable,
                    metrics_dir,
                    results_dir,
                    save_artifacts,
                    device,
                )
                futures[future] = config_path
            for future in as_completed(futures):
                result = future.result()
                status = 'OK' if result['ok'] else 'FAILED'
                config_path = Path(result['config_path'])
                print(f'[{status}] {config_path}')
                results.append(result)
                if not result['ok'] and result['stderr_tail']:
                    print(result['stderr_tail'])

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a batch of streaming experiments.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Filter configs by model.type value.',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Filter configs by dataset.dataset_name value.',
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='conf/experiments/streaming',
        help='Directory with streaming configs.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Parallel workers (-1 uses all CPUs).',
    )
    parser.add_argument(
        '--report-dir',
        type=str,
        default='experiments/outputs/streaming/comparisons',
        help='Base directory for batch output folders.',
    )
    parser.add_argument(
        '--comparison-name',
        type=str,
        default=None,
        help='Optional run output folder name under --report-dir.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help="Execution device: 'auto', 'cuda', or 'cpu'.",
    )
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
        print(f'No configs found. Check {base_path}.')
        sys.exit(1)

    results = run_batch(
        config_files,
        workers=args.workers,
        save_artifacts=False,
        device=device.type if device is not None else None,
    )

    run_folder_name = build_output_folder_name(
        custom_name=args.comparison_name,
        dataset_name=args.dataset,
        model_name=args.model,
    )

    output_dir = Path(args.report_dir) / run_folder_name
    save_comparison_reports(results, output_dir)

    exit_code = print_batch_summary(results)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
