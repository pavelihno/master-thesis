from collections.abc import Callable
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from pybeamline.sources import BEvent
from river.drift import ADWIN
from river.ensemble import SRPClassifier
from river.forest import ARFClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier

from models.darwin import DARWINClassifier
from utils.streaming.base.drift_detectors import NoDriftDetector
from utils.streaming.base.extractors import (
    BinaryOutcomeExtractor,
    MultiClassOutcomeExtractor,
)
from utils.streaming.base.pipelines import (
    NextActivityPredictionPipeline,
    OutcomePredictionPipeline,
    PipelineMode,
    SourceMode,
)
from utils.streaming.base.transformers import (
    ControlFlowTransformer,
    DARWINTransformer,
    DataTransformer,
    DimensionTransformer,
    IndexBasedTransformer,
)
from utils.streaming.experiment import load_saved_model


def create_transformer(config: dict):
    """Create a streaming transformer from a config dict."""
    transformer_type = config.get('type', None)
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
    model_type = config.get('type', None)
    params = dict(config.get('params', {}))

    drift_detector = create_drift_detector(params.pop('drift_detector', {}))
    warning_detector = create_drift_detector(params.pop('warning_detector', {}))

    if model_type == 'srp':
        model = SRPClassifier(
            **params, drift_detector=drift_detector, warning_detector=warning_detector
        )
    elif model_type == 'arf':
        model = ARFClassifier(
            **params, drift_detector=drift_detector, warning_detector=warning_detector
        )
    elif model_type == 'aht':
        model = HoeffdingAdaptiveTreeClassifier(**params, drift_detector=drift_detector)
    elif model_type == 'darwin':
        optimizer_cls = create_optimizer_cls(
            params.pop('optimizer', {'type': 'adam', 'lr': 0.001})
        )
        loss_fn = create_loss_fn(params.pop('loss_fn', {'type': 'cross_entropy'}))

        # TODO: learning rate reducer
        params.pop('lr_reducer', None)

        model = DARWINClassifier(
            **params,
            optimizer_cls=optimizer_cls,
            loss_fn=loss_fn,
            drift_detector=drift_detector,
        )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'.")

    pretrain_path = config.get('pretrain_path')
    if pretrain_path:
        pretrain_path = Path(pretrain_path)
        if not pretrain_path.exists():
            raise FileNotFoundError(f'pretrain_path not found: {pretrain_path}')
        model = load_saved_model(pretrain_path, model)

    return model


def create_dataset(config: dict) -> tuple[dict, set[str] | None]:
    source_mode = SourceMode(config.get('source', None))

    if source_mode == SourceMode.LOG:
        dataset_path = config.get('dataset_path')
        if not dataset_path:
            raise ValueError('dataset_path must be provided for LogSource')

        run_kwargs = {'dataset_path': dataset_path}
        end_events = set(config.get('end_events', []))

    elif source_mode == SourceMode.STRING:
        stream_string = config.get('trace')
        if not stream_string:
            raise ValueError('trace must be provided for StringSource')

        run_kwargs = {'string': stream_string}
        end_events = set(stream_string.split()[-1:])

    elif source_mode == SourceMode.BEVENTS:
        trace_specs = config.get('traces')
        if not trace_specs:
            raise ValueError("traces must be provided e.g. ['ABCDEF': 10000]")

        process_name = config.get('dataset_name', 'synthetic_process')

        case_id = 0
        bevents = []
        end_events = set()

        for trace, count in trace_specs:
            for _ in range(count):
                case_id += 1
                for event in trace:
                    bevents.append(BEvent(event, case_id, process_name))

            end_events.add(trace[-1])

        run_kwargs = {'bevents': bevents}

    else:
        raise ValueError(f"Unknown dataset source mode: '{source_mode}'.")

    return run_kwargs, source_mode, end_events


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
    elif criterion_type == 'nll':
        return nn.NLLLoss(**config)
    elif criterion_type == 'bce':
        return nn.BCELoss(**config)
    elif criterion_type == 'bce_logits':
        return nn.BCEWithLogitsLoss(**config)
    else:
        raise ValueError(f"Unknown criterion type: '{criterion_type}'.")


def create_outcome_extractor(config: dict):
    """Create an outcome extractor for streaming outcome prediction."""
    regime = config.get('regime', None)

    if regime == 'binary':
        positive_outcomes = set(config.get('positive_outcomes', []))
        negative_outcomes = set(config.get('negative_outcomes', []))
        return BinaryOutcomeExtractor(
            positive_outcomes=positive_outcomes,
            negative_outcomes=negative_outcomes,
        )

    elif regime == 'multiclass':
        outcome_mapping = dict(config.get('outcome_mapping', {}))
        return MultiClassOutcomeExtractor(outcome_mapping=outcome_mapping)

    raise ValueError(
        f"Unknown outcome regime: '{regime}'. Use 'binary' or 'multiclass'."
    )


def create_pipeline(
    config: dict,
    model,
    transformer,
    end_events: set | None = None,
    source_mode: SourceMode | str = SourceMode.LOG,
):
    """Create a task pipeline from a config dict."""
    task_type = config.get('type', None)

    pipeline_mode = PipelineMode(config.get('mode'))
    source_mode = SourceMode(source_mode)

    if task_type == 'next_activity':
        return NextActivityPredictionPipeline(
            model=model,
            transformer=transformer,
            end_events=end_events,
            pipeline_mode=pipeline_mode,
            source_mode=source_mode,
        )
    elif task_type == 'outcome':
        outcome_extractor = create_outcome_extractor(config)

        return OutcomePredictionPipeline(
            model=model,
            transformer=transformer,
            end_events=end_events,
            pipeline_mode=pipeline_mode,
            source_mode=source_mode,
            outcome_extractor=outcome_extractor,
        )
    else:
        raise ValueError(f"Unknown task type: '{task_type}'.")
