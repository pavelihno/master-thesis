import argparse
from datetime import datetime
from pathlib import Path

from experiment_utils import (
    ensure_output_dir,
    evaluate_model,
    load_and_prepare_data,
    load_config,
    save_results,
)

from utils.factories import (
    create_bucketer,
    create_dataset,
    create_model,
    create_transformer,
)
from utils.pipelines import ProcessPredictorPipeline


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

    # Load and prepare data (steps 2-5)
    train_prefixes, test_prefixes, y_train, y_test = load_and_prepare_data(
        dataset, config
    )

    print('\n[6/6] Building and training pipeline...')
    bucketer = create_bucketer(config['bucketer'])
    transformer = create_transformer(config['transformer'])

    num_classes = len(dataset.label_encoder.classes_)
    model = create_model(config['model'], num_classes=num_classes)

    pipeline = ProcessPredictorPipeline(bucketer, transformer, model)

    print('Training...')
    pipeline.fit(train_prefixes, y_train)

    # Evaluate model
    eval_results = evaluate_model(
        pipeline, test_prefixes, y_test, dataset, bucketer, verbose=True
    )

    # Save results to output file
    output_folder = ensure_output_dir(config)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = output_folder / f'{experiment_name}_{timestamp}.txt'

    save_results(
        output_file, experiment_name, config_path, timestamp, config, eval_results
    )

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
        'report': eval_results['report'],
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
