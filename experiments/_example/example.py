import os


os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
import mlflow
from omegaconf import DictConfig

from train.pytorch import train_model
from utils.config_factories import get_dataloader, get_device
from utils.metrics import pytorch_accuracy
from utils.mlflow import log_experiment


@hydra.main(
    version_base='1.3',
    config_path='../../conf/experiments/_example',
    config_name='example',
)
@log_experiment(experiment_name='PyTorch_Example')
def main_task(cfg: DictConfig) -> float:
    """
    Example PyTorch training with Hydra and MLflow.

    This function demonstrates:
    - Loading configuration with Hydra
    - Training PyTorch model with custom loss functions
    - Logging to MLflow

    To run:
        python experiments/_example/example.py
    """
    model = train_model(cfg)
    test_loader = get_dataloader(cfg, loader_type='test')
    device = get_device(cfg)

    test_acc = pytorch_accuracy(model, test_loader, device)

    mlflow.log_metric('test_accuracy', test_acc)

    print('Training complete!')
    print(f'Test Accuracy: {test_acc:.4f}')

    return test_acc


if __name__ == '__main__':
    main_task()
