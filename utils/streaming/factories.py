from collections.abc import Callable

import torch.nn as nn
import torch.optim as optim
from river.drift import ADWIN
from river.ensemble import SRPClassifier
from river.forest import ARFClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier

from models.darwin import DARWINClassifier
from utils.streaming.drift_detector import NoDriftDetector
from utils.streaming.pipelines import (
    NextActivityPredictionPipeline,
)
from utils.streaming.transformers import (
    ControlFlowTransformer,
    DARWINTransformer,
    DataTransformer,
    DimensionTransformer,
    IndexBasedTransformer,
)


def create_transformer(config: dict):
    """Create a streaming transformer from a config dict."""
    transformer_type = config['type']
    max_events = config.get('max_events', None)
    include_prefix_len = config.get('include_prefix_len', True)

    if transformer_type == 'cf':
        return ControlFlowTransformer(include_prefix_len=include_prefix_len)
    elif transformer_type == 'data':
        return DataTransformer(include_prefix_len=include_prefix_len)
    elif transformer_type == 'index':
        return IndexBasedTransformer(
            max_events=max_events, include_prefix_len=include_prefix_len
        )
    elif transformer_type == 'dim':
        return DimensionTransformer(
            max_events=max_events, include_prefix_len=include_prefix_len
        )
    elif transformer_type == 'darwin':
        return DARWINTransformer()
    else:
        raise ValueError(f"Unknown transformer type: '{transformer_type}'.")


def create_model(config: dict):
    """Create a river streaming classifier from a config dict."""
    model_type = config['type']
    params = dict(config.get('params', {}))

    drift_detector = create_drift_detector(params.pop('drift_detector', {}))
    warning_detector = create_drift_detector(params.pop('warning_detector', {}))

    if model_type == 'srp':
        return SRPClassifier(
            **params, drift_detector=drift_detector, warning_detector=warning_detector
        )
    elif model_type == 'arf':
        return ARFClassifier(
            **params, drift_detector=drift_detector, warning_detector=warning_detector
        )
    elif model_type == 'aht':
        return HoeffdingAdaptiveTreeClassifier(**params, drift_detector=drift_detector)
    elif model_type == 'darwin':
        clf_optimizer_cls = create_optimizer_cls(
            params.pop('clf_optimizer', {'type': 'adam', 'lr': 0.001})
        )
        clf_criterion = create_criterion(
            params.pop('clf_criterion', {'type': 'cross_entropy'})
        )

        return DARWINClassifier(
            **params,
            clf_optimizer_cls=clf_optimizer_cls,
            clf_criterion=clf_criterion,
            drift_detector=drift_detector,
        )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'.")


def create_drift_detector(config: dict):
    """Create a drift detector from a config dict."""
    config = dict(config)
    detector_type = config.pop('type', 'none')

    if detector_type == 'adwin':
        return ADWIN(**config)
    elif detector_type == 'none':
        return NoDriftDetector()
    else:
        raise ValueError(f"Unknown drift detector type: '{detector_type}'.")


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
    else:
        raise ValueError(f"Unknown optimizer type: '{optimizer_type}'.")


def create_criterion(config: dict):
    """Return a loss instance from a config dict."""

    config = dict(config)
    criterion_type = config.pop('type', 'cross_entropy').lower()

    if criterion_type == 'cross_entropy':
        return nn.CrossEntropyLoss(**config)
    elif criterion_type == 'nll':
        return nn.NLLLoss(**config)
    elif criterion_type == 'bce':
        return nn.BCELoss(**config)
    elif criterion_type == 'bce_logits':
        return nn.BCEWithLogitsLoss(**config)
    else:
        raise ValueError(f"Unknown criterion type: '{criterion_type}'.")


def create_pipeline(config: dict, model, transformer, end_events: set | None = None):
    """Create a task pipeline from a config dict."""
    task_type = config['type']

    if task_type == 'next_activity':
        return NextActivityPredictionPipeline(
            model=model,
            transformer=transformer,
            end_events=end_events,
        )
    else:
        raise ValueError(f"Unknown task type: '{task_type}'.")
