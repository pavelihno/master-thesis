import argparse
import json
from datetime import datetime
from pathlib import Path

from utils.experiment import ensure_output_dir, load_config
from utils.streaming.experiment import (
    build_run_summary,
    get_config_hash,
    load_saved_model,
    make_json_safe,
    save_model,
    save_plots,
    write_results,
)
from utils.streaming.factories import create_model, create_pipeline, create_transformer


def run_config_file(config_path: str) -> dict:
    config = load_config(config_path)
    return run_config(config, config_path=config_path)


def run_config(
    config: dict,
    config_path: str,
    *,
    save_artifacts: bool = True,
    metrics_json_path: str | None = None,
) -> dict:
    model_type = config['model']['type'].lower()
    config_hash = get_config_hash(config)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_id = f'{model_type}_{config_hash}_{timestamp}'

    print('=' * 60)
    print(f'Run ID: {run_id}')
    print('=' * 60)

    dataset_path = config['dataset']['dataset_path']

    print(f'Task: {config["task"]["type"]}')
    print(f'Mode: {config["task"].get("mode")}')
    print(f'Dataset: {config["dataset"]["dataset_name"]}')
    print(f'Transformer: {config["transformer"]["type"]}')
    print(f'Model: {config["model"]["type"]}')

    transformer = create_transformer(config['transformer'])
    model = create_model(config['model'])

    pretrain_path = config['model'].get('pretrain_path')
    if pretrain_path:
        pretrain_path = Path(pretrain_path)
        if not pretrain_path.exists():
            raise FileNotFoundError(f'pretrain_path not found: {pretrain_path}')
        print(f'Pretrain: {pretrain_path}')
        model = load_saved_model(pretrain_path, model)

    end_events_config = config['dataset'].get('end_events')
    end_events = set(end_events_config) if end_events_config else None

    pipeline = create_pipeline(
        config['task'],
        model=model,
        transformer=transformer,
        end_events=end_events,
    )

    results_df, model, elapsed_seconds = pipeline.run(dataset_path=dataset_path)

    last_row = results_df.iloc[-1]
    print(
        f'\nn_pred={int(last_row.get("n_pred", 0))}, '
        f'accuracy={last_row.get("accuracy", 0):.4f}, '
        f'macro_f1={last_row.get("macro_f1", 0):.4f}, '
        f'n_drifts={int(last_row.get("n_drifts", 0))}, '
        f'time={elapsed_seconds:.2f}s'
    )

    output_folder = None
    if save_artifacts:
        output_folder = ensure_output_dir(config) / run_id
        output_folder.mkdir(parents=True, exist_ok=True)
        write_results(output_folder, run_id, config_path, timestamp, config, results_df)
        save_model(output_folder, model)
        save_plots(output_folder, results_df, config['dataset'].get('drift_points'))

    run_summary = build_run_summary(
        run_id=run_id,
        config_path=config_path,
        timestamp=timestamp,
        config_hash=config_hash,
        config=config,
        last_row=last_row,
        elapsed_seconds=elapsed_seconds,
        output_dir=str(output_folder) if output_folder else None,
    )

    if metrics_json_path:
        metrics_path = Path(metrics_json_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(make_json_safe(run_summary), f, indent=2)
        print(f'Metrics JSON -> {metrics_path}')

    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run one streaming experiment.')
    parser.add_argument('config_path', type=str, help='Path to the YAML config.')
    parser.add_argument(
        '--no-save-artifacts',
        action='store_true',
        help='Run without saving results, model, or plots.',
    )
    parser.add_argument(
        '--metrics-json',
        type=str,
        default=None,
        help='Optional JSON path for structured run metrics.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        run_config(
            load_config(args.config_path),
            config_path=args.config_path,
            save_artifacts=not args.no_save_artifacts,
            metrics_json_path=args.metrics_json,
        )
    except Exception as exc:
        if args.metrics_json:
            failure_path = Path(args.metrics_json)
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with open(failure_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'config_path': args.config_path,
                        'status': 'failed',
                        'error': str(exc),
                    },
                    f,
                    indent=2,
                )
        raise


if __name__ == '__main__':
    main()
