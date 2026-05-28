import argparse
import json
from pathlib import Path

from utils.experiment.core import (
    ensure_output_dir,
    get_config_hash,
    load_config,
    make_json_safe,
    select_device,
)
from utils.experiment.naming import (
    build_output_folder_name,
    get_dataset_and_model,
    get_timestamp,
)
from utils.streaming.experiment.core import (
    build_run_summary,
    format_metric_values,
    get_stream_summary_values,
    prepare_results_frame,
    save_model,
    write_results,
)
from utils.streaming.factories import (
    create_dataset,
    create_model,
    create_pipeline,
    create_transformer,
)


def run_config_file(config_path: str) -> dict:
    config = load_config(config_path)
    return run_config(config, config_path=config_path)


def run_config(
    config: dict,
    config_path: str,
    *,
    save_artifacts: bool = True,
    metrics_json_path: str | None = None,
    results_csv_path: str | None = None,
    device=None,
) -> dict:
    dataset_name, model_name = get_dataset_and_model(config)
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

    print(f'Task: {config["task"]["type"]}')
    print(f'Mode: {config["task"].get("mode")}')
    print(f'Dataset: {dataset_name}')
    print(f'Transformer: {config["transformer"]["type"]}')
    print(f'Model: {model_name}')
    if device is not None:
        print(f'Device: {device.type}')

    transformer = create_transformer(config['transformer'])
    model = create_model(config['model'], device=device)
    dataset_kwargs, dataset_source, terminator = create_dataset(config['dataset'])

    pipeline = create_pipeline(
        config['task'],
        model=model,
        transformer=transformer,
        terminator=terminator,
        source_mode=dataset_source,
    )

    results_df, model, elapsed_seconds = pipeline.run(**dataset_kwargs)

    summary_values = get_stream_summary_values(results_df)
    metric_text = format_metric_values(summary_values)

    print_metrics = [
        f'n_preds={summary_values["n_preds"]}',
        f'n_traces={summary_values["n_traces"]}',
        f'n_events={summary_values["n_events"]}',
        f'n_drifts={summary_values["n_drifts"]}',
    ]
    if metric_text:
        print_metrics.append(metric_text)
    print_metrics.append(f'time={elapsed_seconds:.2f}s')

    print('\n' + ', '.join(print_metrics))

    output_folder = None

    if save_artifacts:
        output_folder = ensure_output_dir(config) / run_id
        output_folder.mkdir(parents=True, exist_ok=True)
        write_results(
            output_folder,
            run_id,
            config_path,
            timestamp,
            config,
            results_df,
            elapsed_seconds=elapsed_seconds,
        )
        save_model(output_folder, model)

    if results_csv_path:
        results_path = Path(results_csv_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        prepare_results_frame(config, results_df).to_csv(results_path, index=False)

    run_summary = build_run_summary(
        run_id=run_id,
        config_path=config_path,
        timestamp=timestamp,
        config_hash=config_hash,
        config=config,
        results_df=results_df,
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
        '--save-artifacts',
        type=bool,
        default=True,
        help='Optional saving artifacts.',
    )
    parser.add_argument(
        '--metrics-json',
        type=str,
        default=None,
        help='Optional JSON path for structured run metrics.',
    )
    parser.add_argument(
        '--results-csv',
        type=str,
        default=None,
        help='Optional CSV path for per-row prediction results.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help="Execution device: 'auto', 'cuda', or 'cpu'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)

    try:
        run_config(
            load_config(args.config_path),
            config_path=args.config_path,
            save_artifacts=args.save_artifacts,
            metrics_json_path=args.metrics_json,
            results_csv_path=args.results_csv,
            device=device,
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
