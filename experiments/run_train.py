import argparse
from datetime import datetime
from pathlib import Path

import yaml
from sklearn.metrics import classification_report

from utils.factories import (
    create_bucketer,
    create_dataset,
    create_model,
    create_transformer,
)
from utils.pipelines import ProcessPredictorPipeline
from utils.stats import print_class_balance


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(config_path, experiment_name=None):
    """Run a single experiment from config file."""
    config = load_config(config_path)

    if experiment_name is None:
        experiment_name = config.get('experiment_name', 'unnamed_experiment')

    print('=' * 80)
    print(f'Running experiment: {experiment_name}')
    print('=' * 80)

    print('\n[1/6] Creating dataset...')
    dataset = create_dataset(config['dataset'])

    print('\n[2/6] Loading and preprocessing data...')
    dataset.load_and_preprocess()

    if config['dataset']['type'] == 'outcome':
        dataset.filter_by_labels()

    print('\n[3/6] Splitting data...')
    train_df, test_df = dataset.train_test_split()

    print('\n[4/6] Generating prefixes...')
    train_prefixes = dataset.get_prefixes(train_df)
    test_prefixes = dataset.get_prefixes(test_df)

    print('\n[5/6] Preparing labels...')
    y_train = dataset.prepare_labels(train_prefixes)
    y_test = dataset.prepare_labels(test_prefixes)

    print_class_balance(y_train, dataset_names=[dataset.dataset_name, 'Train'])
    print_class_balance(y_test, dataset_names=[dataset.dataset_name, 'Test'])

    print('\n[6/6] Building and training pipeline...')
    bucketer = create_bucketer(config['bucketer'])
    transformer = create_transformer(config['transformer'])

    num_classes = len(dataset.label_encoder.classes_)
    model = create_model(config['model'], num_classes=num_classes)

    pipeline = ProcessPredictorPipeline(bucketer, transformer, model)

    print('Training...')
    pipeline.fit(train_prefixes, y_train)

    print('Evaluating...')
    y_pred = pipeline.predict(test_prefixes)

    y_pred_decoded = dataset.decode_labels(y_pred)
    y_test_decoded = dataset.decode_labels(y_test)

    report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)
    report_str = classification_report(y_test_decoded, y_pred_decoded)

    print('\n' + '=' * 80)
    print('RESULTS')
    print('=' * 80)
    print(report_str)

    # Save results to output file
    output_folder = Path(config.get('output', {}).get('folder', 'outputs'))
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = output_folder / f'{experiment_name}_{timestamp}.txt'

    with open(output_file, 'w') as f:
        f.write(f'Experiment: {experiment_name}\n')
        f.write(f'Config: {config_path}\n')
        f.write(f'Timestamp: {timestamp}\n')
        f.write('=' * 80 + '\n\n')

        f.write('Configuration:\n')
        f.write('-' * 80 + '\n')
        f.write(yaml.dump(config, default_flow_style=False))
        f.write('\n')

        f.write('Results:\n')
        f.write('-' * 80 + '\n')
        f.write(report_str)
        f.write('\n')

        f.write('\nSummary Metrics:\n')
        f.write('-' * 80 + '\n')
        f.write(f'Accuracy: {report["accuracy"]:.4f}\n')
        f.write(f'Macro Avg F1-Score: {report["macro avg"]["f1-score"]:.4f}\n')
        f.write(f'Weighted Avg F1-Score: {report["weighted avg"]["f1-score"]:.4f}\n')

    print(f'\nResults saved to: {output_file}')

    # Save model if configured
    if config.get('output', {}).get('save_model', False):
        model_dir = Path(config.get('output', {}).get('model_folder', 'models'))
        model_save_path = model_dir / f'{experiment_name}_{timestamp}'

        print('\nSaving model...')
        pipeline.save(model_save_path)
        print(f'Model saved to: {model_save_path}')

    return {
        'experiment_name': experiment_name,
        'config_path': config_path,
        'timestamp': timestamp,
        'report': report,
        'output_file': str(output_file),
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Run experiment from YAML config')
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to YAML configuration file',
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (overrides config file)',
    )

    args = parser.parse_args()

    run_experiment(args.config_path, args.name)


if __name__ == '__main__':
    main()
