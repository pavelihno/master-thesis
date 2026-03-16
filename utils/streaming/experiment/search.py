import copy
import itertools
from typing import Any


def build_param_grid(param_space: dict | None) -> list[dict]:
    if not param_space:
        return [{}]

    normalized = {
        key: values if isinstance(values, list) else [values]
        for key, values in param_space.items()
    }
    keys = list(normalized.keys())
    value_lists = list(normalized.values())
    return [
        dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)
    ]


def format_params(params: dict) -> str:
    if not params:
        return '<default>'

    parts = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f'{key}={value:.4g}')
        else:
            parts.append(f'{key}={value}')
    return ', '.join(parts)


def print_search_space(param_space: dict | None) -> None:
    print('\nParameter search space:')
    print('-' * 80)

    if not param_space:
        print('No explicit search space found.')
        return

    for key, values in param_space.items():
        options = values if isinstance(values, list) else [values]
        print(f'- {key}: {options}')


def set_nested_value(target: dict, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split('.')
    node = target
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def get_nested_value(target: dict, dotted_path: str):
    node = target
    for part in dotted_path.split('.'):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing config path: '{dotted_path}'.")
        node = node[part]
    return node


def collect_search_lists(node: Any, path: str, out: dict) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            next_path = f'{path}.{key}' if path else key
            collect_search_lists(value, next_path, out)
        return

    if not isinstance(node, list):
        return

    if not node:
        raise ValueError(f"Search candidate list at '{path}' is empty.")
    if any(isinstance(value, (dict, list)) for value in node):
        raise ValueError(
            f"Unsupported nested list at '{path}'. Only scalar lists are supported."
        )
    out[path] = node


def extract_search_space(config: dict) -> dict:
    param_space = {}
    for root_path in ['transformer', 'model.params']:
        collect_search_lists(
            get_nested_value(config, root_path), root_path, param_space
        )
    return param_space


def apply_params(config: dict, params: dict) -> dict:
    updated = copy.deepcopy(config)
    for key, value in params.items():
        set_nested_value(updated, key, value)
    return updated


def remove_search_config(config: dict) -> dict:
    cleaned = copy.deepcopy(config)
    if 'model' in cleaned:
        cleaned['model'].pop('param_grid', None)
    cleaned.pop('search', None)
    return cleaned
