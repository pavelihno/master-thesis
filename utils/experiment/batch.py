from pathlib import Path

from utils.experiment.core import load_config


def find_config_files(
    base_dir: str,
    model_name: str | None = None,
    dataset_name: str | None = None,
    exclude_hyperparam_search: bool = False,
) -> list[Path]:
    """Find config files in base_dir with optional filtering by model/dataset"""
    base_path = Path(base_dir)

    if not base_path.exists():
        return []

    config_files = sorted(
        file_path
        for file_path in base_path.rglob('*.yaml')
        if not any(part.startswith('_') for part in file_path.parts)
    )

    if model_name or dataset_name:
        model_filter = model_name.lower() if model_name else None
        dataset_filter = dataset_name.lower() if dataset_name else None

        filtered_files: list[Path] = []
        for file_path in config_files:
            config = load_config(file_path) or {}
            config_model = str(config.get('model', {}).get('type', '')).lower()
            config_dataset = str(
                config.get('dataset', {}).get('dataset_name', '')
            ).lower()

            if model_filter and model_filter not in config_model:
                continue
            if dataset_filter and dataset_filter not in config_dataset:
                continue
            filtered_files.append(file_path)

        config_files = filtered_files

    if exclude_hyperparam_search:
        filtered_files = []
        for file_path in config_files:
            config = load_config(file_path) or {}
            if config.get('hyperparam_search', False):
                continue
            filtered_files.append(file_path)
        config_files = filtered_files

    return config_files
