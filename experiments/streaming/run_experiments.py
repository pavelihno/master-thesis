import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from utils.streaming.experiment.batch import (
    find_config_files,
    print_batch_summary,
    run_config_process,
    save_comparison_reports,
)


def run_batch(
    config_files: list[Path],
    workers: int = 1,
    save_artifacts: bool = False,
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
            print(f'\n{"=" * 60}')
            print(f'Running: {config_path.name}')
            print('=' * 60)
            result = run_config_process(
                config_path,
                runner_path,
                python_executable,
                metrics_dir,
                results_dir,
                save_artifacts,
            )
            status = 'OK' if result['ok'] else 'FAILED'
            print(f'[{status}] {config_path.name}')
            results.append(result)
            if not result['ok'] and result['stderr_tail']:
                print(result['stderr_tail'])
    else:
        print(f'Launching up to {workers} experiments in parallel.\n')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_config_process,
                    c,
                    runner_path,
                    python_executable,
                    metrics_dir,
                    results_dir,
                    save_artifacts,
                ): c
                for c in config_files
            }
            for future in as_completed(futures):
                result = future.result()
                status = 'OK' if result['ok'] else 'FAILED'
                config_path = Path(result['config_path'])
                print(f'[{status}] {config_path.name}')
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
        help='Model subfolder to run.',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Filter config names by dataset substring.',
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
        help='Directory for the combined CSV report.',
    )
    parser.add_argument(
        '--comparison-name',
        type=str,
        default=None,
        help='Optional combined CSV filename prefix.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    )

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    comparison_name = args.comparison_name or f'streaming_comparison_{timestamp}'
    save_comparison_reports(results, Path(args.report_dir), comparison_name)

    exit_code = print_batch_summary(results)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
