from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


def get_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def slugify(value: str) -> str:
    slug = ''.join(ch.lower() if ch.isalnum() else '_' for ch in value)
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug.strip('_')


def get_dataset_and_model(config: Mapping[str, Any]) -> tuple[str | None, str | None]:
    dataset_name = config.get('dataset', {}).get('dataset_name')
    model_name = config.get('model', {}).get('type')
    return dataset_name, model_name


def load_yaml_config(config_path: str | Path) -> dict | None:
    try:
        import yaml

        with open(config_path, encoding='utf-8') as file:
            return yaml.safe_load(file) or {}
    except Exception:
        return None


def read_run_info(config_path: str | Path) -> tuple[str | None, str | None]:
    config = load_yaml_config(config_path) or {}
    return get_dataset_and_model(config)


def build_output_folder_name(
    *,
    custom_name: str | None = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
    config_hash: str | None = None,
    suffix: str | None = None,
    fallback_name: str = 'all',
    timestamp: str | None = None,
) -> str:

    parts = []

    if custom_name:
        parts.append(slugify(custom_name))
    else:
        if dataset_name:
            parts.append(slugify(dataset_name))
        if model_name:
            parts.append(slugify(model_name))
        if config_hash:
            parts.append(slugify(config_hash))
        if suffix:
            parts.append(slugify(suffix))

    if not parts:
        parts.append(slugify(fallback_name))

    parts.append(timestamp or get_timestamp())

    return '_'.join([part for part in parts if part])
