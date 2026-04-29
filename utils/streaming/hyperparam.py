import copy
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope

from utils.streaming.factories import (
    create_dataset,
    create_model,
    create_pipeline,
    create_transformer,
)


def get_hyperopt_space(cfg: dict, prefix: str = '') -> dict:
    space = {}

    def process_value(key: str, value):
        if isinstance(value, dict):
            if 'search' in value:
                search_type = str(value['search']).lower()
                is_exp = bool(value.get('exp', False))
                is_log = bool(value.get('log', False))
                base = float(value.get('base', 2))

                if is_exp and is_log:
                    raise ValueError(f"Use only one of 'exp' or 'log' for '{key}'.")

                if search_type == 'choice':
                    opts = value['options']
                    proc_opts = [
                        process_value(f'{key}__opt{i}', opt)
                        if isinstance(opt, dict)
                        else opt
                        for i, opt in enumerate(opts)
                    ]
                    return hp.choice(key, proc_opts)

                if search_type in ('int', 'float'):
                    low, high = value['low'], value['high']
                    is_int = search_type == 'int'

                    if is_exp:
                        if is_int:
                            e = scope.int(
                                hp.quniform(f'{key}__exp', int(low), int(high), 1)
                            )
                            return scope.int(scope.pow(base, e))
                        e = hp.uniform(f'{key}__exp', float(low), float(high))
                        return scope.pow(base, e)

                    if is_log:
                        low_f, high_f = float(low), float(high)
                        if low_f <= 0 or high_f <= 0:
                            raise ValueError(
                                f"Log scale requires low/high > 0 for '{key}'."
                            )
                        if is_int:
                            return scope.int(
                                hp.loguniform(key, math.log(low_f), math.log(high_f))
                            )
                        return hp.loguniform(key, math.log(low_f), math.log(high_f))

                    if is_int:
                        return scope.int(hp.quniform(key, int(low), int(high), 1))
                    return hp.uniform(key, float(low), float(high))

                raise ValueError(f"Unsupported search type '{search_type}' at '{key}'.")

            return {k: process_value(f'{key}_{k}', v) for k, v in value.items()}

        return value

    for key, value in cfg.items():
        space[key] = process_value(f'{prefix}_{key}' if prefix else key, value)

    return space


def create_cfg_yaml(cfg: dict, model_cfg: dict, transformer_cfg: dict) -> dict:
    base_cfg = copy.deepcopy(cfg)

    task_cfg = base_cfg.get('task', {})
    dataset_cfg = base_cfg.get('dataset', {})
    dataset_name = dataset_cfg.get('dataset_name', 'unnamed_dataset')

    final_cfg = {
        'task': task_cfg,
        'dataset': dataset_cfg,
        'transformer': copy.deepcopy(transformer_cfg),
        'model': copy.deepcopy(model_cfg),
        'output': {'folder': f'experiments/outputs/streaming/{dataset_name}'},
    }

    return final_cfg


def save_cfg_yaml(
    cfg: dict,
    model_cfg: dict,
    transformer_cfg: dict,
    save_path: str | Path,
) -> dict:
    final_cfg = create_cfg_yaml(cfg, model_cfg, transformer_cfg)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(final_cfg, file, sort_keys=False)

    return final_cfg


def run_hyperopt(cfg: dict, device: torch.device | None = None) -> dict:
    cfg = copy.deepcopy(cfg)

    task_cfg = cfg.get('task', {})
    dataset_cfg = cfg.get('dataset', {})
    dataset_kwargs, dataset_source, end_events = create_dataset(dataset_cfg)

    transformer_cfg = cfg.get('transformer', {})
    model_cfg = cfg.get('model', {})

    search_cfg = cfg.get('search', {})
    max_trials = int(search_cfg.get('max_trials', 1))
    metric = search_cfg.get('metric', None)
    if metric is None:
        raise ValueError("Search configuration must include a 'metric' to optimize.")
    maximize = bool(search_cfg.get('maximize', True))
    seed = int(search_cfg.get('seed', 123))

    model_space = get_hyperopt_space(model_cfg)
    transformer_space = get_hyperopt_space(transformer_cfg, prefix='transformer')

    hyperopt_space = {
        'model_cfg': model_space,
        'transformer_cfg': transformer_space,
    }

    trials = Trials()

    def objective(sample: dict):
        sampled_model_cfg = sample['model_cfg']
        sampled_transformer_cfg = sample['transformer_cfg']

        model = create_model(sampled_model_cfg, device=device)
        transformer = create_transformer(sampled_transformer_cfg)

        pipeline = create_pipeline(
            task_cfg,
            model=model,
            transformer=transformer,
            end_events=end_events,
            source_mode=dataset_source,
        )

        results_df, model, elapsed_time = pipeline.run(**dataset_kwargs)
        pred_df = results_df.dropna(subset=['y_pred'])

        if pred_df.empty:
            raise RuntimeError(
                'No predictions were produced during hyperparameter tuning.'
            )

        score = float(pred_df[metric].iloc[-1])
        loss = -score if maximize else score

        return {
            'loss': loss,
            'score': score,
            'status': STATUS_OK,
            'metric': metric,
            'maximize': maximize,
            'elapsed_time': float(elapsed_time),
            'model_cfg': sampled_model_cfg,
            'transformer_cfg': sampled_transformer_cfg,
        }

    best_raw = fmin(
        fn=objective,
        space=hyperopt_space,
        algo=tpe.suggest,
        max_evals=max_trials,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )

    best_all = space_eval(hyperopt_space, best_raw)

    return {
        'best_raw': best_raw,
        'best_model_cfg': best_all['model_cfg'],
        'best_transformer_cfg': best_all['transformer_cfg'],
        'metric': metric,
        'maximize': maximize,
        'trials': trials,
    }


def extract_trial_rows(hyperopt_results: dict) -> list[dict]:
    trials_obj = hyperopt_results['trials']

    rows = []
    for trial in trials_obj.trials:
        result = trial.get('result', {})
        if result.get('status') != 'ok':
            continue

        loss = result.get('loss')
        if loss is None:
            continue

        rows.append(
            {
                'trial_id': trial.get('tid'),
                'loss': float(loss),
                'score': float(result.get('score')),
                'metric': result.get('metric', hyperopt_results.get('metric')),
                'maximize': result.get(
                    'maximize',
                    hyperopt_results.get('maximize', True),
                ),
                'elapsed_time': float(result.get('elapsed_time', 0.0)),
                'model_cfg': result.get('model_cfg'),
                'transformer_cfg': result.get('transformer_cfg'),
            }
        )

    rows.sort(key=lambda row: row['loss'])
    return rows


def build_top_models_report(
    hyperopt_results: dict,
    cfg: dict,
    k: int = 5,
) -> tuple[list[dict], str]:
    top_rows = extract_trial_rows(hyperopt_results)[:k]

    lines = [f'Top {len(top_rows)} trials', '=' * 80]
    for rank, row in enumerate(top_rows, start=1):
        cfg_yaml = create_cfg_yaml(
            cfg,
            model_cfg=row['model_cfg'],
            transformer_cfg=row['transformer_cfg'],
        )
        cfg_yaml_text = yaml.safe_dump(cfg_yaml, sort_keys=False).rstrip()

        lines.append(f'{rank}) score: {row["score"]}')
        lines.append('cfg_yaml:')
        lines.append(cfg_yaml_text)
        lines.append('=' * 80)

    text_content = '\n'.join(lines)
    return top_rows, text_content


def save_top_models(
    hyperopt_results: dict,
    cfg: dict,
    k: int = 5,
    save_path: str | Path = 'top_models.txt',
) -> tuple[list[dict], str]:
    top_rows, text_content = build_top_models_report(hyperopt_results, cfg, k=k)

    print(text_content)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(text_content)

    print(f'Saved top models report to: {save_path}')
    return top_rows, text_content
