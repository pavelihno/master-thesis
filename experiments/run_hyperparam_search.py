import argparse
import itertools
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from experiment_utils import evaluate_model, load_and_prepare_data, load_config
from sklearn.metrics import classification_report

from utils.factories import (
    create_bucketer,
    create_dataset,
    create_model,
    create_transformer,
)
from utils.pipelines import ProcessPredictorPipeline


def expand_param_grid(param_config):
    """Expand parameter configuration into list of all combinations."""
    if not param_config:
        return [{}]

    # Separate single values from lists
    param_lists = {}
    for key, value in param_config.items():
        if isinstance(value, list):
            param_lists[key] = value
        else:
            param_lists[key] = [value]

    # Generate all combinations
    keys = list(param_lists.keys())
    values = list(param_lists.values())

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo, strict=True)))

    return combinations


def format_params_string(params):
    """Format parameters dictionary as readable string."""
    param_strs = []
    for key, value in params.items():
        if isinstance(value, float):
            param_strs.append(f'{key}={value:.4g}')
        else:
            param_strs.append(f'{key}={value}')
    return ', '.join(param_strs)


def run_hyperparam_search(config_path, experiment_name=None):
    """Run hyperparameter search experiment from config file."""
    config = load_config(config_path)

    if experiment_name is None:
        experiment_name = config.get('experiment_name', 'hyperparam_search')

    print('=' * 80)
    print(f'Running Hyperparameter Search: {experiment_name}')
    print('=' * 80)

    # Data preparation (done once)
    print('\n[1/6] Creating dataset...')
    dataset = create_dataset(config['dataset'])

    # Load and prepare data (steps 2-5)
    train_prefixes, test_prefixes, y_train, y_test = load_and_prepare_data(
        dataset, config
    )

    # Create fixed bucketer and transformer
    print('\n[6/6] Creating bucketer and transformer...')
    bucketer = create_bucketer(config['bucketer'])
    transformer = create_transformer(config['transformer'])

    # Generate parameter grid
    print('\nGenerating parameter grid...')
    param_grid = expand_param_grid(config['model'].get('param_grid', {}))

    if len(param_grid) == 0 or (len(param_grid) == 1 and not param_grid[0]):
        print('Warning: No parameter grid specified. Using default parameters.')
        param_grid = [config['model'].get('params', {})]

    print(f'Total configurations to test: {len(param_grid)}')
    print()

    # Store results
    results = []
    best_score = -1
    best_idx = 0

    # Run grid search
    for idx, params in enumerate(param_grid):
        print('-' * 80)
        print(f'Configuration {idx + 1}/{len(param_grid)}')
        print(f'Parameters: {format_params_string(params)}')
        print('-' * 80)

        # Create model with current parameters
        model_config = {'type': config['model']['type'], 'params': params}

        num_classes = len(dataset.label_encoder.classes_)
        model = create_model(model_config, num_classes=num_classes)

        # Build and train pipeline
        pipeline = ProcessPredictorPipeline(bucketer, transformer, model)

        print('Training...')
        pipeline.fit(train_prefixes, y_train)

        # Evaluate model
        eval_results = evaluate_model(
            pipeline, test_prefixes, y_test, dataset, bucketer, verbose=False
        )

        # Store results
        result_entry = {
            'config_idx': idx + 1,
            'params': params,
            'accuracy': eval_results['accuracy'],
            'f1_macro': eval_results['f1_macro'],
            'f1_weighted': eval_results['f1_weighted'],
            'report': eval_results['report'],
            'bucket_stats': eval_results['bucket_stats'],
        }
        results.append(result_entry)

        # Track best model (by F1 weighted)
        if eval_results['f1_weighted'] > best_score:
            best_score = eval_results['f1_weighted']
            best_idx = idx

        print(f'Accuracy: {eval_results["accuracy"]:.4f}')
        print(f'F1 Macro: {eval_results["f1_macro"]:.4f}')
        print(f'F1 Weighted: {eval_results["f1_weighted"]:.4f}')
        print()

    # Display results table
    print('\n' + '=' * 80)
    print('HYPERPARAMETER SEARCH RESULTS')
    print('=' * 80)

    results_df = pd.DataFrame(
        [
            {
                'Config': r['config_idx'],
                **{f'param_{k}': v for k, v in r['params'].items()},
                'Accuracy': r['accuracy'],
                'F1_Macro': r['f1_macro'],
                'F1_Weighted': r['f1_weighted'],
                'Best': '★' if i == best_idx else '',
            }
            for i, r in enumerate(results)
        ]
    )

    print(results_df.to_string(index=False))
    print()

    # Best model details
    best_result = results[best_idx]
    print('=' * 80)
    print(f'BEST MODEL (Configuration {best_idx + 1})')
    print('=' * 80)
    print(f'Parameters: {format_params_string(best_result["params"])}')
    print(f'Accuracy: {best_result["accuracy"]:.4f}')
    print(f'F1 Macro: {best_result["f1_macro"]:.4f}')
    print(f'F1 Weighted: {best_result["f1_weighted"]:.4f}')
    print()
    print('Per-Bucket Statistics (Best Model):')
    print('-' * 80)
    print(best_result['bucket_stats'].to_string(index=False))
    print()

    # Save results to output file
    output_folder = Path(config.get('output', {}).get('folder', 'outputs'))
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = output_folder / f'{experiment_name}_hypersearch_{timestamp}.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'Hyperparameter Search: {experiment_name}\n')
        f.write(f'Config: {config_path}\n')
        f.write(f'Timestamp: {timestamp}\n')
        f.write(f'Total Configurations Tested: {len(param_grid)}\n')
        f.write('=' * 80 + '\n\n')

        f.write('Configuration:\n')
        f.write('-' * 80 + '\n')
        f.write(yaml.dump(config, default_flow_style=False))
        f.write('\n')

        f.write('Results Summary:\n')
        f.write('-' * 80 + '\n')
        f.write(results_df.to_string(index=False))
        f.write('\n\n')

        f.write('=' * 80 + '\n')
        f.write(f'BEST MODEL (Configuration {best_idx + 1})\n')
        f.write('=' * 80 + '\n')
        f.write(f'Parameters: {format_params_string(best_result["params"])}\n')
        f.write(f'Accuracy: {best_result["accuracy"]:.4f}\n')
        f.write(f'F1 Macro: {best_result["f1_macro"]:.4f}\n')
        f.write(f'F1 Weighted: {best_result["f1_weighted"]:.4f}\n')
        f.write('\n')

        f.write('Per-Bucket Statistics (Best Model):\n')
        f.write('-' * 80 + '\n')
        f.write(best_result['bucket_stats'].to_string(index=False))
        f.write('\n\n')

        f.write('Detailed Classification Report (Best Model):\n')
        f.write('-' * 80 + '\n')

        # Reconstruct classification report
        y_pred = pipeline.predict(test_prefixes)
        y_pred_decoded = dataset.decode_labels(y_pred)
        y_test_decoded = dataset.decode_labels(y_test)
        report_str = classification_report(
            y_test_decoded, y_pred_decoded, zero_division=0
        )
        f.write(report_str)
        f.write('\n')

    print(f'Results saved to: {output_file}')

    # Optionally save best model
    if config.get('output', {}).get('save_best_model', False):
        model_dir = Path(config.get('output', {}).get('model_folder', 'models'))
        model_save_path = model_dir / f'{experiment_name}_best_{timestamp}'

        print('\nTraining and saving best model...')

        # Retrain with best parameters
        best_model_config = {
            'type': config['model']['type'],
            'params': best_result['params'],
        }
        best_model = create_model(best_model_config, num_classes=num_classes)
        best_pipeline = ProcessPredictorPipeline(bucketer, transformer, best_model)
        best_pipeline.fit(train_prefixes, y_train)

        best_pipeline.save(model_save_path)
        print(f'Best model saved to: {model_save_path}')

    return {
        'experiment_name': experiment_name,
        'config_path': config_path,
        'timestamp': timestamp,
        'results': results,
        'best_idx': best_idx,
        'best_result': best_result,
        'output_file': str(output_file),
    }


def main():
    """Main entry point for hyperparameter search script."""
    parser = argparse.ArgumentParser(
        description='Run hyperparameter search from YAML config'
    )
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to YAML configuration file with parameter grid',
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (overrides config file)',
    )

    args = parser.parse_args()

    run_hyperparam_search(args.config_path, args.name)


if __name__ == '__main__':
    main()
