import argparse
import subprocess
import sys
from pathlib import Path


def find_configs(base_path, model_type=None, dataset=None):
    """Find experiment configurations matching criteria."""
    base = Path(base_path)
    configs = []

    if model_type:
        # Search in specific model directory
        model_dir = base / model_type
        if model_dir.exists():
            if dataset:
                # Specific dataset
                config = model_dir / f'{dataset}.yaml'
                if config.exists():
                    configs.append(config)
            else:
                # All datasets for this model
                configs.extend(model_dir.glob('*.yaml'))
    else:
        # Search all model directories
        for model_dir in base.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('_'):
                if dataset:
                    # Specific dataset across all models
                    config = model_dir / f'{dataset}.yaml'
                    if config.exists():
                        configs.append(config)
                else:
                    # All configs
                    configs.extend(model_dir.glob('*.yaml'))

    return sorted(configs)


def run_experiment(config_path, python_exe='python'):
    """Run a single experiment."""
    cmd = [python_exe, 'experiments/run_train.py', str(config_path)]
    print(f'\n{"=" * 80}')
    print(f'Running: {config_path}')
    print(f'{"=" * 80}\n')

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run multiple experiments in batch')
    parser.add_argument(
        '--model',
        type=str,
        help='Model type to run (e.g., xgboost, random_forest)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset to run (e.g., traffic_fines, helpdesk)',
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='conf/experiments',
        help='Base experiments directory',
    )
    parser.add_argument(
        '--python',
        type=str,
        default='python',
        help='Python executable to use',
    )

    args = parser.parse_args()

    # Find matching configs
    configs = find_configs(args.base_path, args.model, args.dataset)

    if not configs:
        print('No matching configurations found.')
        print(f'Base path: {args.base_path}')
        if args.model:
            print(f'Model filter: {args.model}')
        if args.dataset:
            print(f'Dataset filter: {args.dataset}')
        return 1

    print(f'Found {len(configs)} configuration(s):')
    for config in configs:
        print(f'  - {config}')
    print()

    # Run experiments
    results = {}
    for config in configs:
        success = run_experiment(config, args.python)
        results[config] = success

    # Summary
    print(f'\n{"=" * 80}')
    print('BATCH EXPERIMENT SUMMARY')
    print(f'{"=" * 80}\n')

    successful = sum(results.values())
    failed = len(results) - successful

    print(f'Total: {len(results)}')
    print(f'Successful: {successful}')
    print(f'Failed: {failed}')
    print()

    if failed > 0:
        print('Failed experiments:')
        for config, success in results.items():
            if not success:
                print(f'  - {config}')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
