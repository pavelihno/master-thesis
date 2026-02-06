import functools

import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def log_experiment(experiment_name: str):
    """
    Decorator to wrap a training function with MLflow tracking.
    Handles nested runs (Optuna trials) and config logging.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(cfg: DictConfig, *args, **kwargs):
            mlflow.set_experiment(experiment_name)

            # Determine if this is a Hydra Sweep trial or a single run
            hydra_cfg = HydraConfig.get()
            is_sweep = hydra_cfg.mode.name == 'MULTIRUN'

            # Use the job number/name for the trial run name
            run_name = f'trial_{hydra_cfg.job.num}' if is_sweep else 'single_run'

            # Start the run (nested=is_sweep handles the Optuna hierarchy)
            with mlflow.start_run(run_name=run_name, nested=is_sweep):
                params = OmegaConf.to_container(cfg, resolve=True)

                # Log the Hydra config as a flattened dict for parameters
                mlflow.log_params(_flatten_dict(params))

                # Save YAML artifact for full reconstruction
                mlflow.log_dict(params, 'config.yaml')

                # Execute experiment code
                return func(cfg, *args, **kwargs)

        return wrapper

    return decorator


def _flatten_dict(d, parent_key='', sep='.'):
    """Helper to turn nested config into flat keys for MLflow UI."""
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
