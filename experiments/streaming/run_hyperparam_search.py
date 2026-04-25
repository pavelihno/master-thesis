import argparse
import json
from pathlib import Path

import pandas as pd

from utils.experiment import load_config
from utils.streaming.experiment import (
    build_output_folder_name,
    get_dataset_and_model,
    make_json_safe,
    select_device,
)
from utils.streaming.hyperparam import (
    extract_trial_rows,
    run_hyperopt,
    save_cfg_yaml,
    save_top_models,
)


def run_search(
    config_path: str,
    *,
    report_dir: str | None = None,
    search_name: str | None = None,
    top_k: int = 10,
    device=None,
) -> dict:

    config = load_config(config_path)
    dataset_name, model_name = get_dataset_and_model(config)

    search_name = build_output_folder_name(
        custom_name=search_name,
        dataset_name=dataset_name,
        model_name=model_name,
        suffix='hypersearch',
        fallback_name='search',
    )

    if report_dir is None:
        report_dir = config.get('output', {}).get('folder')
        if report_dir:
            report_path = Path(report_dir)
        else:
            report_path = Path('experiments/outputs/streaming/hyperparam_search')
    else:
        report_path = Path(report_dir)

    report_path = report_path / search_name

    report_path.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print(f'Hyperparameter Search: {search_name}')
    print('=' * 80)
    print(f'Config file         : {config_path}')
    if device is not None:
        print(f'Device              : {device.type}')

    hyperopt_results = run_hyperopt(config, device=device)
    metric_name = hyperopt_results['metric']
    is_max = hyperopt_results['maximize']
    metric_mode = 'max' if is_max else 'min'
    print(f'Search metric       : {metric_name} ({metric_mode})')

    trial_rows = extract_trial_rows(hyperopt_results)
    print(f'Total trials        : {len(trial_rows)}')

    rows = []
    for idx, trial_row in enumerate(trial_rows, start=1):
        row = {
            'trial_idx': idx,
            'search_metric': metric_name,
            'score': float(trial_row['score']),
            'loss': float(trial_row['loss'] or 0),
            'time_s': float(trial_row.get('elapsed_time', 0.0)),
            'model_cfg': make_json_safe(trial_row.get('model_cfg')),
            'transformer_cfg': make_json_safe(trial_row.get('transformer_cfg')),
            'trial_params_json': json.dumps(
                {
                    'model_cfg': make_json_safe(trial_row.get('model_cfg')),
                    'transformer_cfg': make_json_safe(trial_row.get('transformer_cfg')),
                },
                sort_keys=True,
            ),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError('No hyperparameter trials were executed.')

    results_df = pd.DataFrame(rows)
    sorted_df = results_df.sort_values('loss', ascending=True).reset_index(drop=True)
    sorted_df['is_best'] = False
    sorted_df.loc[0, 'is_best'] = True

    summary_columns = [
        'trial_idx',
        'time_s',
        'score',
        'loss',
        'is_best',
    ]

    print('\n' + '=' * 80)
    print('HYPERPARAMETER SEARCH RESULTS')
    print('=' * 80)
    preview_columns = [c for c in summary_columns if c in sorted_df.columns]
    print(sorted_df[preview_columns].to_string(index=False))

    trials_csv = report_path / 'trials.csv'
    sorted_df.to_csv(trials_csv, index=False)

    best_row = trial_rows[0]
    best_score = float(best_row['score'])
    best_model_cfg = best_row['model_cfg']
    best_transformer_cfg = best_row['transformer_cfg']

    best_config_yaml = report_path / 'best_config.yaml'
    best_config = save_cfg_yaml(
        config,
        model_cfg=best_model_cfg,
        transformer_cfg=best_transformer_cfg,
        save_path=best_config_yaml,
    )

    top_models_txt = report_path / 'top_models.txt'
    top_rows, _ = save_top_models(
        hyperopt_results,
        config,
        k=top_k,
        save_path=top_models_txt,
    )

    best_metrics_json = report_path / 'best_metrics.json'
    with open(best_metrics_json, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'search_name': search_name,
                'search_metric': metric_name,
                'maximize': is_max,
                'best_score': best_score,
                'top_k': min(top_k, len(top_rows)),
                'best_model_cfg': make_json_safe(best_model_cfg),
                'best_transformer_cfg': make_json_safe(best_transformer_cfg),
                'best_config_yaml': str(best_config_yaml),
                'top_models_txt': str(top_models_txt),
            },
            f,
            indent=2,
        )

    print('\n' + '=' * 80)
    print('BEST TRIAL')
    print('=' * 80)
    print(f'{metric_name}: {best_score:.6f}')
    print(f'Best cfg  : {best_config_yaml}')
    print('\nSaved files:')
    print(f'Trials CSV       -> {trials_csv}')
    print(f'Best config YAML -> {best_config_yaml}')
    print(f'Top models TXT   -> {top_models_txt}')
    print(f'Best metrics JSON-> {best_metrics_json}')

    return {
        'search_name': search_name,
        'config_path': config_path,
        'search_metric': metric_name,
        'maximize': is_max,
        'trials_csv': str(trials_csv),
        'best_config_yaml': str(best_config_yaml),
        'top_models_txt': str(top_models_txt),
        'best_metrics_json': str(best_metrics_json),
        'best_score': best_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run streaming hyperparameter search.')
    parser.add_argument(
        'config_path',
        type=str,
        help=(
            'Path to a streaming YAML config. Search candidates are inferred '
            'from list-valued fields in transformer and model.params.'
        ),
    )
    parser.add_argument(
        '--report-dir',
        type=str,
        default=None,
        help='Directory for search reports.',
    )
    parser.add_argument(
        '--search-name',
        type=str,
        default=None,
        help='Optional output name prefix.',
    )
    parser.add_argument('--top-k', type=int, default=10, help='Top-k models in TXT.')
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help="Execution device: 'auto', 'cuda', or 'cpu'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    run_search(
        args.config_path,
        report_dir=args.report_dir,
        search_name=args.search_name,
        top_k=args.top_k,
        device=device,
    )


if __name__ == '__main__':
    main()
