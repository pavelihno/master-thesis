import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    model = instantiate(cfg.model)
    return model


def get_dataloader(cfg: DictConfig, loader_type='train'):
    loader = instantiate(
        cfg.train_dataloader if loader_type == 'train' else cfg.test_dataloader
    )
    return loader


def get_optimizer(cfg: DictConfig, model):
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    return optimizer


def get_loss_fn(cfg: DictConfig):
    loss_fn = instantiate(cfg.loss_fn)
    return loss_fn


def get_device(cfg: DictConfig):
    if hasattr(cfg, 'device'):
        device = torch.device(cfg.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device
