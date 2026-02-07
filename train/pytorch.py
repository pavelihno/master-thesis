import os

import mlflow
import torch
from omegaconf import DictConfig

from utils.config_factories import (
    get_dataloader,
    get_device,
    get_loss_fn,
    get_model,
    get_optimizer,
)


def train_model(cfg: DictConfig):
    """
    Train PyTorch model with Hydra configuration, MLflow logging.

    Args:
        cfg: Hydra configuration
    Returns:
        Trained model
    """
    model = get_model(cfg)
    device = get_device(cfg)
    train_loader = get_dataloader(cfg, loader_type='train')
    loss_fn = get_loss_fn(cfg)
    optimizer = get_optimizer(cfg, model)

    # Training parameters
    epochs = cfg.training.get('epochs', 20)
    self_supervised = cfg.training.get('self_supervised', False)
    checkpoint_freq = cfg.training.get('checkpoint_freq', 50)
    checkpoint_dir = cfg.training.get('checkpoint_dir', None)
    checkpoint_path = cfg.training.get('checkpoint_path', None)
    model_dir = cfg.training.get('model_dir', None)
    model_name = cfg.training.get('model_name', None)

    model.to(device)

    start_epoch = 0

    print('\nStarting training...')

    # Resume from checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f'Resumed from epoch {start_epoch}, Loss: {loss:.4f}')

        mlflow.log_param('resumed_from_checkpoint', checkpoint_path)
        mlflow.log_param('resumed_epoch', start_epoch)

    mlflow.log_param('device', str(device))

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()

        running_loss = 0.0
        n_samples = 0

        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device) if not self_supervised else data.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)

            running_loss += loss.item()
            n_samples += data.size(0)

            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / n_samples

        mlflow.log_metric('train_loss', epoch_loss, step=epoch)

        # Print progress
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'{model_name}_check_{epoch + 1}.pth'
                if model_name
                else f'checkpoint_{epoch + 1}.pth',
            )
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                },
                checkpoint_path,
            )
            print(f'Checkpoint saved: {checkpoint_path}')

    # Save final model
    if model_name is not None:
        os.makedirs(model_dir, exist_ok=True)
        file_path = (
            os.path.join(model_dir, f'{model_name}.pth')
            if model_dir
            else f'{model_name}.pth'
        )

        torch.save(model.state_dict(), file_path)
        mlflow.log_artifact(file_path)

        print(f'Model weights saved to "{file_path}"')

    return model
