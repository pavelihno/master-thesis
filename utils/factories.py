from collections.abc import Callable

import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from models.lstm import LSTM as LSTMModel
from models.wrapper import SklearnModelWrapper, TorchModelWrapper
from utils.base.bucketers import NoBucketer, PrefixLengthBucketer
from utils.base.datasets import NextActivityDataset, OutcomeDataset
from utils.base.transformers import (
    AggregateTransformer,
    IndexBasedTransformer,
    LastStateTransformer,
)
from utils.constants import (
    ACTIVITY_COL,
    CASE_ID_COL,
    CASE_PREFIX_COL,
    RESOURCE_COL,
    TIME_COL,
)
from utils.loss_functions import LogCoshLoss


def create_dataset(config):
    """Create a dataset instance from config."""
    dataset_type = config.get('type', 'outcome')
    dataset_path = config.pop('dataset_path', None)
    label_path = config.pop('label_path', None)

    case_id_col = config.pop('case_id_col', CASE_ID_COL)
    time_col = config.pop('time_col', TIME_COL)
    activity_col = config.pop('activity_col', ACTIVITY_COL)
    resource_col = config.pop('resource_col', RESOURCE_COL)

    train_ratio = config.get('train_ratio', 0.8)
    min_prefix = config.get('min_prefix', None)
    max_prefix = config.get('max_prefix', None)

    if dataset_type == 'outcome':
        return OutcomeDataset(
            dataset_path=dataset_path,
            label_path=label_path,
            train_ratio=train_ratio,
            min_prefix=min_prefix,
            max_prefix=max_prefix,
            case_id_col=case_id_col,
            time_col=time_col,
            activity_col=activity_col,
            resource_col=resource_col,
        )
    elif dataset_type == 'next_activity':
        return NextActivityDataset(
            dataset_path=dataset_path,
            label_path=label_path,
            train_ratio=train_ratio,
            min_prefix=min_prefix,
            max_prefix=max_prefix,
            case_id_col=case_id_col,
            time_col=time_col,
            activity_col=activity_col,
            resource_col=resource_col,
        )
    elif dataset_type == 'remaining_time':
        raise NotImplementedError(
            'Remaining time prediction dataset not implemented yet'
        )
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')


def create_bucketer(config):
    """Create a bucketer instance from config."""
    bucketer_type = config.get('type', 'prefix_length')
    group_col = config.get('group_col', CASE_PREFIX_COL)

    if bucketer_type == 'none':
        return NoBucketer(group_col=group_col)
    elif bucketer_type == 'prefix_length':
        return PrefixLengthBucketer(group_col=group_col)
    elif bucketer_type == 'cluster_based':
        raise NotImplementedError(
            'ClusterBasedBucketer not implemented yet. '
            'Will cluster cases based on feature similarity.'
        )
    elif bucketer_type == 'state_based':
        raise NotImplementedError(
            'StateBasedBucketer not implemented yet. '
            'Will bucket cases based on their current state/activity sequence.'
        )
    else:
        raise ValueError(f'Unknown bucketer type: {bucketer_type}')


def create_transformer(config):
    """Create a transformer instance from config."""
    transformer_type = config['type']

    if transformer_type == 'aggregate':
        return AggregateTransformer(
            group_col=config.get('group_col', CASE_PREFIX_COL),
            cat_cols=config.get('cat_cols', []),
            num_cols=config.get('num_cols', []),
            boolean=config.get('boolean', False),
            fillna=config.get('fillna', True),
        )
    elif transformer_type == 'last_state':
        return LastStateTransformer(
            group_col=config.get('group_col', CASE_PREFIX_COL),
            cat_cols=config.get('cat_cols', []),
            num_cols=config.get('num_cols', []),
        )
    elif transformer_type == 'index_based':
        return IndexBasedTransformer(
            group_col=config.get('group_col', CASE_PREFIX_COL),
            cat_cols=config.get('cat_cols', []),
            num_cols=config.get('num_cols', []),
            max_events=config.get('max_len'),
        )
    else:
        raise ValueError(f'Unknown transformer type: {transformer_type}')


def create_model(config, device=None):
    """Create a model instance from config."""
    model_type = config['type']
    params = config.get('params', {})

    if model_type == 'logistic_regression':
        return SklearnModelWrapper(LogisticRegression(**params))
    elif model_type == 'random_forest':
        return SklearnModelWrapper(RandomForestClassifier(**params))
    elif model_type == 'svm':
        return SklearnModelWrapper(SVC(**params))
    elif model_type == 'lstm':
        optimizer_cls = create_optimizer_cls(params.pop('optimizer', {'type': 'adam'}))
        loss_fn = create_loss_fn(params.pop('loss_fn', {'type': 'cross_entropy'}))
        epochs = params.pop('epochs', 20)
        batch_size = params.pop('batch_size', 32)

        model = LSTMModel(**params)

        return TorchModelWrapper(
            model,
            device=device,
            optimizer_cls=optimizer_cls,
            loss_fn=loss_fn,
            epochs=epochs,
            batch_size=batch_size,
        )

    else:
        raise ValueError(f'Unknown model type: {model_type}')


def create_optimizer_cls(config: dict) -> Callable:
    """Return an optimizer callable from a config dict."""
    config = dict(config)
    optimizer_type = config.pop('type', 'adam').lower()

    if optimizer_type == 'adam':
        return lambda p: optim.Adam(p, **config)
    elif optimizer_type == 'sgd':
        return lambda p: optim.SGD(p, **config)
    elif optimizer_type == 'adamw':
        return lambda p: optim.AdamW(p, **config)
    elif optimizer_type == 'rmsprop':
        return lambda p: optim.RMSprop(p, **config)
    elif optimizer_type == 'nadam':
        return lambda p: optim.NAdam(p, **config)
    else:
        raise ValueError(f"Unknown optimizer type: '{optimizer_type}'.")


def create_loss_fn(config: dict):
    """Return a loss instance from a config dict."""

    config = dict(config)
    criterion_type = config.pop('type', 'cross_entropy').lower()

    if criterion_type == 'cross_entropy':
        return nn.CrossEntropyLoss(**config)
    elif criterion_type == 'mse':
        return nn.MSELoss(**config)
    elif criterion_type == 'mae':
        return nn.L1Loss(**config)
    elif criterion_type == 'log_cosh':
        return LogCoshLoss()
    else:
        raise ValueError(f"Unknown criterion type: '{criterion_type}'.")
