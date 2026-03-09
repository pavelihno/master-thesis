import argparse
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import yaml

from utils.experiment import ensure_output_dir, load_config
from utils.streaming.evaluation import plot_stream_metric
from utils.streaming.factories import create_model, create_pipeline, create_transformer


def compute_config_hash(config: dict, length: int = 8) -> str:
    """Short deterministic hash of the stable config fields."""
    stable = {k: config[k] for k in ('model', 'transformer', 'dataset') if k in config}
    serialized = json.dumps(stable, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:length]


def save_results(
    output_path: Path,
    run_id: str,
    config_path: str,
    timestamp: str,
    config: dict,
    df,
) -> None:
    """Persist the prediction DataFrame and a summary text file."""
    # Raw predictions CSV
    csv_file = output_path / 'results.csv'
    df.to_csv(csv_file, index=False)

    # Summary text
    last = df.iloc[-1]
    summary_file = output_path / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f'Run ID: {run_id}\n')
        f.write(f'Config     : {config_path}\n')
        f.write(f'Timestamp  : {timestamp}\n')
        f.write('=' * 60 + '\n\n')

        f.write('Configuration:\n')
        f.write('-' * 60 + '\n')
        f.write(yaml.dump(config, default_flow_style=False))
        f.write('\n')

        f.write('Summary Metrics:\n')
        f.write('-' * 60 + '\n')
        f.write(f'Predictions: {int(last["n_pred"])}\n')
        f.write(f'Accuracy: {last["accuracy"]:.4f}\n')
        f.write(f'Macro F1: {last["macro_f1"]:.4f}\n')

        f.write(f'Num Drifts: {int(last["n_drifts"])}\n')
        f.write(f'Time (s): {last["time_s"]:.2f}\n')

    print(f'Results → {csv_file}')
    print(f'Summary → {summary_file}')


def save_model(
    output_path: Path,
    model,
) -> None:
    model_file = output_path / 'model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model → {model_file}')


def save_plots(
    output_path: Path,
    df,
    drift_points: list[int] | None = None,
) -> None:

    plot_path = output_path / 'plots'
    plot_path.mkdir(exist_ok=True)

    metrics = ['accuracy', 'f1']

    windows = [100, 200]

    centered_windows = [True, False]

    for metric in metrics:
        for window in windows:
            for center_window in centered_windows:
                center_str = 'centered' if center_window else 'trailing'
                fig_file = plot_path / f'{metric}_w{window}_{center_str}.png'
                plot_stream_metric(
                    df,
                    metric=metric,
                    window=window,
                    center_window=center_window,
                    save_path=fig_file,
                    actual_drift_points=drift_points,
                )


def run_experiment(config_path: str) -> dict:
    """Run a single streaming experiment from a YAML config file."""
    config = load_config(config_path)

    model_type = config['model']['type'].lower()
    config_hash = compute_config_hash(config)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run_id = f'{model_type}_{config_hash}_{timestamp}'

    print('=' * 60)
    print(f'Run ID: {run_id}')
    print('=' * 60)

    dataset_path = config['dataset']['dataset_path']

    print(f'Dataset: {config["dataset"]["dataset_name"]}')
    print(f'Transformer: {config["transformer"]["type"]}')
    print(f'Model: {config["model"]["type"]}')

    # Build components
    transformer = create_transformer(config['transformer'])

    pretrain_path = config['model'].get('pretrain_path')
    if pretrain_path:
        pretrain_path = Path(pretrain_path)
        if not pretrain_path.exists():
            raise FileNotFoundError(f'pretrain_path not found: {pretrain_path}')
        print(f'Pretrain: {pretrain_path}')
        with open(pretrain_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = create_model(config['model'])

    end_events_cfg = config['dataset'].get('end_events')
    end_events = set(end_events_cfg) if end_events_cfg else None

    pipeline = create_pipeline(
        config['task'],
        model=model,
        transformer=transformer,
        end_events=end_events,
    )

    # Run
    results_df, model = pipeline.run(dataset_path)
    results_df['experiment'] = run_id

    # Report
    last = results_df.iloc[-1]
    print(
        f'\nn_pred={int(last.get("n_pred", 0))}, '
        f'accuracy={last.get("accuracy", 0):.4f}, '
        f'macro_f1={last.get("macro_f1", 0):.4f}, '
        f'n_drifts={int(last.get("n_drifts", 0))}, '
        f'time={last.get("time_s", 0):.2f}s'
    )

    # Save
    output_folder = ensure_output_dir(config) / run_id
    output_folder.mkdir(parents=True, exist_ok=True)
    save_results(output_folder, run_id, config_path, timestamp, config, results_df)
    save_model(output_folder, model)
    save_plots(output_folder, results_df, config['dataset'].get('drift_points'))

    return {
        'run_id': run_id,
        'n_pred': int(last.get('n_pred', 0)),
        'accuracy': float(last.get('accuracy', 0)),
        'macro_f1': float(last.get('macro_f1', 0)),
        'n_drifts': int(last.get('n_drifts', 0)),
        'time_s': float(last.get('time_s', 0)),
        'output_dir': str(output_folder),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run a single streaming experiment from a YAML config.'
    )
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to the YAML configuration file.',
    )
    args = parser.parse_args()
    run_experiment(args.config_path)


if __name__ == '__main__':
    main()
