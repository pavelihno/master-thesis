import argparse

from utils.base.pipelines import PredictionPipeline
from utils.experiment.core import (
    compute_bucket_statistics,
    ensure_output_dir,
    get_config_hash,
    load_and_prepare_data,
    load_config,
    select_device,
)
from utils.experiment.naming import (
    build_output_folder_name,
    get_dataset_name,
    get_model_name,
    get_timestamp,
)
from utils.factories import (
    create_bucketer,
    create_dataset,
    create_model,
    create_transformer,
)


def run_config_file(config_path: str) -> dict:
    config = load_config(config_path)
    return run_config(config, config_path=config_path)


def run_config(
    config: dict,
    config_path: str,
    *,
    device=None,
) -> dict:
    """Run a single experiment from config file."""
    dataset_name = get_dataset_name(config)
    model_name = get_model_name(config)
    config_hash = get_config_hash(config)
    timestamp = get_timestamp()
    run_id = build_output_folder_name(
        dataset_name=dataset_name,
        model_name=model_name,
        config_hash=config_hash,
        fallback_name='run',
        timestamp=timestamp,
    )

    print('=' * 60)
    print(f'Run ID: {run_id}')
    print('=' * 60)

    print(f'Config: {config_path}')
    print(f'Dataset: {dataset_name}')
    print(f'Transformer: {config["transformer"]["type"]}')
    print(f'Model: {model_name}')
    if device is not None:
        print(f'Device: {device.type}')

    # Create dataset and load data
    dataset = create_dataset(config['dataset'])
    train_prefixes, test_prefixes, y_train, y_test = load_and_prepare_data(dataset)

    # Create pipeline components
    bucketer = create_bucketer(config['bucketer'])
    transformer = create_transformer(config['transformer'])
    model = create_model(config['model'], device=device)

    # Train pipeline
    pipeline = PredictionPipeline(bucketer, transformer, model)
    pipeline.fit(train_prefixes, y_train)

    # Generate predictions
    y_train_pred = pipeline.predict(train_prefixes)
    y_test_pred = pipeline.predict(test_prefixes)

    # Compute statistics
    train_bucket_stats_df = compute_bucket_statistics(
        train_prefixes, y_train, y_train_pred, bucketer
    )
    test_bucket_stats_df = compute_bucket_statistics(
        test_prefixes, y_test, y_test_pred, bucketer
    )

    # Save results to output folder
    output_folder = ensure_output_dir(config)
    output_folder = output_folder / run_id
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save training statistics
    train_bucket_stats_df.to_csv(output_folder / 'train_bucket_stats.csv', index=False)
    test_bucket_stats_df.to_csv(output_folder / 'test_bucket_stats.csv', index=False)

    results = {
        'run_id': run_id,
        'dataset': dataset_name,
        'model': model_name,
        'output_folder': str(output_folder),
    }

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run one experiment')
    parser.add_argument('config_path', type=str, help='Path to the YAML config.')
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help="Execution device: 'auto', 'cuda', or 'cpu'.",
    )
    return parser.parse_args()


def main():
    """Main entry point for training script."""
    args = parse_args()
    device = select_device(args.device)

    try:
        config = load_config(args.config_path)
        results = run_config(
            config,
            config_path=args.config_path,
            device=device,
        )
        print('\n' + '=' * 60)
        print('Experiment completed successfully!')
        print(f'Results saved to: {results["output_folder"]}')
        print('=' * 60)
    except Exception as exc:
        print(f'\nExperiment failed with error: {exc}')
        raise


if __name__ == '__main__':
    main()
