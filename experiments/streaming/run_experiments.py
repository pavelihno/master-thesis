import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def find_configs(
    base_path: str,
    model_type: str | None = None,
    dataset: str | None = None,
) -> list[Path]:
    """Return sorted YAML configs."""
    base = Path(base_path)

    if model_type:
        search_root = base / model_type
        configs = sorted(search_root.glob('*.yaml')) if search_root.exists() else []
    else:
        configs = sorted(
            p
            for d in base.iterdir()
            if d.is_dir() and not d.name.startswith('_')
            for p in d.glob('*.yaml')
        )

    if dataset:
        configs = [c for c in configs if dataset.lower() in c.stem.lower()]

    return configs


def _run_subprocess(config_path: Path, python_exe: str) -> tuple[Path, bool]:
    """Launch a single experiment in a child process."""
    cmd = [python_exe, str(Path(__file__).parent / 'run_train.py'), str(config_path)]
    result = subprocess.run(cmd, capture_output=False)
    return config_path, result.returncode == 0


def run_experiments(
    configs: list[Path],
    workers: int = 1,
    python_exe: str = 'python',
) -> dict[Path, bool]:
    """Run a list of experiment configs, optionally in parallel."""

    print(f'Found {len(configs)} config(s).')

    if workers == -1:
        import os

        workers = os.cpu_count() or 1

    results: dict[Path, bool] = {}

    if workers == 1:
        for config_path in configs:
            print(f'\n{"=" * 60}')
            print(f'Running: {config_path.name}')
            print('=' * 60)
            _, ok = _run_subprocess(config_path, python_exe)
            results[config_path] = ok
    else:
        print(f'Launching up to {workers} experiments in parallel.\n')
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_subprocess, c, python_exe): c for c in configs}
            for future in as_completed(futures):
                config_path, ok = future.result()
                status = 'OK' if ok else 'FAILED'
                print(f'[{status}] {config_path.name}')
                results[config_path] = ok

    return results


def _print_summary(results: dict[Path, bool]) -> int:
    successful = sum(results.values())
    failed = len(results) - successful

    print(f'\n{"=" * 60}')
    print('BATCH SUMMARY')
    print('=' * 60)
    print(f'Total      : {len(results)}')
    print(f'Successful : {successful}')
    print(f'Failed     : {failed}')

    if failed:
        print('\nFailed experiments:')
        for path, ok in results.items():
            if not ok:
                print(f'  - {path}')
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch streaming experiment runner.')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model sub-folder to filter (e.g. streaming_random_patches).',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Filter configs by dataset name substring (e.g. BPIC_20_DD).',
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='conf/experiments/streaming',
        help='Base directory for streaming experiment configs.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of experiments to run in parallel (-1 = all CPUs, default: 1).',
    )
    parser.add_argument(
        '--python',
        type=str,
        default='python',
        help='Python executable to use (default: python).',
    )
    args = parser.parse_args()

    configs = find_configs(args.base_path, model_type=args.model, dataset=args.dataset)

    if not configs:
        print('No configs found. Check conf/experiments/streaming/.')
        sys.exit(1)

    run_experiments(
        configs,
        workers=args.workers,
        python_exe=args.python,
    )


if __name__ == '__main__':
    main()
