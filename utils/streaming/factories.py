from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from pybeamline.sources import BEvent
from river.drift import ADWIN, KSWIN, NoDrift
from river.ensemble import SRPClassifier
from river.forest import ARFClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier

from models.streaming.darwin import DARWINClassifier, DARWINRegressor
from models.streaming.ngram import NGramClassifier, NGramRegressor
from models.streaming.sequence import LSTMModel, ProcessTransformerModel
from utils.constants import ACTIVITY_COL, CASE_ID_COL, TIME_COL
from utils.loss_functions import LogCoshLoss
from utils.streaming.base.extractors import (
    BinaryOutcomeExtractor,
    MultiClassOutcomeExtractor,
)
from utils.streaming.base.pipelines import (
    NextActivityPredictionPipeline,
    OutcomePredictionPipeline,
    PipelineMode,
    RemainingTimePredictionPipeline,
    SourceMode,
)
from utils.streaming.base.sample_buffers import (
    ReservoirSampleBuffer,
    SampleBuffer,
)
from utils.streaming.base.terminators import (
    EventNameTerminator,
    OracleTerminator,
    TraceTerminator,
)
from utils.streaming.base.transformers import (
    ControlFlowTransformer,
    DARWINTimeTransformer,
    DARWINTransformer,
    DataTransformer,
    DimensionTransformer,
    IndexBasedTransformer,
)
from utils.streaming.experiment.core import load_saved_model
from utils.streaming.time import TimeTarget


def create_transformer(config: dict):
    """Create a streaming transformer from a config dict."""
    config = dict(config)
    transformer_type = config.pop('type', None)
    max_events = config.pop('max_events', None)
    include_prefix_len = config.pop('include_prefix_len', True)

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
    elif transformer_type == 'darwin_time':
        return DARWINTimeTransformer()
    else:
        raise ValueError(f"Unknown transformer type: '{transformer_type}'.")


def create_model(config: dict, device: torch.device | None = None):
    """Create a river streaming classifier from a config dict."""
    config = dict(config)
    model_type = config.pop('type', None)
    params = dict(config.pop('params', {}))

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
    elif model_type == 'ngram_classifier':
        model = NGramClassifier(**params)
    elif model_type == 'ngram_regressor':
        model = NGramRegressor(**params)
    elif model_type == 'darwin_classifier' or model_type == 'darwin_regressor':
        darwin_cls = (
            DARWINClassifier if model_type == 'darwin_classifier' else DARWINRegressor
        )

        sequence_model = create_sequence_model(
            params.pop('sequence_model', {'type': 'lstm'}),
            embedding_dim=params.get('embedding_dim', 0),
            feature_size=params.get('feature_size', 0),
            sequence_window=params.get('sequence_window', 0),
        )
        optimizer_cls = create_optimizer_cls(
            params.pop('optimizer', {'type': 'adam', 'lr': 0.001})
        )
        loss_fn = create_loss_fn(params.pop('loss_fn', {'type': 'cross_entropy'}))
        sample_buffer = create_sample_buffer(params.pop('sample_buffer', None))

        model = darwin_cls(
            **params,
            sequence_model=sequence_model,
            optimizer_cls=optimizer_cls,
            loss_fn=loss_fn,
            drift_detector=drift_detector,
            sample_buffer=sample_buffer,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'.")

    pretrain_path = config.pop('pretrain_path', None)
    if pretrain_path:
        pretrain_path = Path(pretrain_path)
        if not pretrain_path.exists():
            raise FileNotFoundError(f'pretrain_path not found: {pretrain_path}')
        model = load_saved_model(pretrain_path, model)

    return model


def create_dataset(config: dict) -> tuple[dict, SourceMode, TraceTerminator]:
    config = dict(config)
    source_mode = SourceMode(config.pop('source', None))
    terminator = create_terminator(config.pop('terminator', None))
    inject_terminal = isinstance(terminator, OracleTerminator)

    if source_mode == SourceMode.LOG:
        dataset_path = config.pop('dataset_path', None)
        if not dataset_path:
            raise ValueError('dataset_path must be provided for LogSource')

        case_id_col = config.pop('case_id_col', CASE_ID_COL)
        time_col = config.pop('time_col', TIME_COL)
        activity_col = config.pop('activity_col', ACTIVITY_COL)

        run_kwargs = {
            'dataset_path': dataset_path,
            'inject_terminal': inject_terminal,
            'case_id_col': case_id_col,
            'activity_col': activity_col,
            'time_col': time_col,
        }

    elif source_mode == SourceMode.STRING:
        stream_string = config.pop('trace', None)
        if not stream_string:
            raise ValueError('trace must be provided for StringSource')

        run_kwargs = {'string': stream_string, 'inject_terminal': inject_terminal}

    elif source_mode == SourceMode.DATA_FRAME:
        data_frame = config.pop('data_frame', None)
        if data_frame is None:
            raise ValueError('data_frame must be provided for DataFrameSource')

        case_id_col = config.pop('case_id_col', CASE_ID_COL)
        time_col = config.pop('time_col', TIME_COL)
        activity_col = config.pop('activity_col', ACTIVITY_COL)

        run_kwargs = {
            'data_frame': data_frame,
            'inject_terminal': inject_terminal,
            'case_id_col': case_id_col,
            'activity_col': activity_col,
            'time_col': time_col,
        }

    elif source_mode == SourceMode.BEVENTS:
        trace_specs = config.pop('traces', None)
        if not trace_specs:
            raise ValueError("traces must be provided e.g. ['ABCDEF': 10000]")

        process_name = config.pop('dataset_name', 'synthetic_process')
        case_id = 0
        bevents = []

        for trace, count in trace_specs:
            for _ in range(count):
                case_id += 1
                for event in trace:
                    bevents.append(BEvent(event, case_id, process_name))

        run_kwargs = {'bevents': bevents, 'inject_terminal': inject_terminal}

    else:
        raise ValueError(f"Unknown dataset source mode: '{source_mode}'.")

    return run_kwargs, source_mode, terminator


def create_terminator(config: dict | None) -> TraceTerminator:
    config = dict(config)
    terminator_type = config.pop('type', 'oracle').lower()

    if terminator_type == 'event_name':
        end_events = set(config.pop('end_events', []))
        return EventNameTerminator(end_events=end_events)

    elif terminator_type == 'oracle':
        return OracleTerminator()

    raise ValueError(f"Unknown terminator type: '{terminator_type}'.")


def create_drift_detector(config: dict):
    """Create a drift detector from a config dict."""
    config = dict(config)
    detector_type = config.pop('type', 'none').lower()

    if detector_type == 'adwin':
        return ADWIN(**config)
    elif detector_type == 'kswin':
        return KSWIN(**config)
    elif detector_type == 'none':
        return NoDrift()
    else:
        raise ValueError(f"Unknown drift detector type: '{detector_type}'.")


def create_sequence_model(
    config: dict, embedding_dim: int, feature_size: int, sequence_window: int
):
    """Create DARWIN sequence backbone from DARWIN model params."""
    config = dict(config)
    architecture_type = config.pop('type', 'lstm').lower()
    input_dim = embedding_dim + feature_size

    if architecture_type == 'lstm':
        return LSTMModel(input_dim=input_dim, **config)
    elif architecture_type == 'process_transformer':
        return ProcessTransformerModel(
            input_dim=input_dim, max_len=sequence_window, **config
        )
    else:
        raise ValueError(f"Unknown DARWIN architecture type: '{architecture_type}'.")


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


def create_sample_buffer(config: dict | None) -> SampleBuffer | None:
    """Create a sample buffer from a config dict."""
    if not config:
        return None

    config = dict(config)
    buffer_type = str(config.pop('type', None)).lower()
    if not buffer_type:
        return None

    if buffer_type == 'sample':
        return SampleBuffer(**config)
    elif buffer_type == 'reservoir':
        return ReservoirSampleBuffer(**config)
    elif buffer_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown sample buffer type: '{buffer_type}'.")


def create_outcome_extractor(config: dict):
    """Create an outcome extractor for streaming outcome prediction."""
    config = dict(config)
    regime = config.pop('regime', None)

    if regime == 'binary':
        positive_outcomes = set(config.pop('positive_outcomes', []))
        negative_outcomes = set(config.pop('negative_outcomes', []))
        return BinaryOutcomeExtractor(
            positive_outcomes=positive_outcomes,
            negative_outcomes=negative_outcomes,
        )

    elif regime == 'multiclass':
        outcome_mapping = dict(config.pop('outcome_mapping', {}))
        return MultiClassOutcomeExtractor(outcome_mapping=outcome_mapping)

    raise ValueError(
        f"Unknown outcome regime: '{regime}'. Use 'binary' or 'multiclass'."
    )


def create_pipeline(
    config: dict,
    model,
    transformer,
    terminator: TraceTerminator,
    source_mode: SourceMode | str = SourceMode.LOG,
):
    """Create a task pipeline from a config dict."""
    config = dict(config)
    task_type = config.pop('type', None)
    max_events = config.pop('max_events', None)

    pipeline_mode = PipelineMode(config.pop('mode', None))
    source_mode = SourceMode(source_mode)

    if task_type == 'next_activity':
        return NextActivityPredictionPipeline(
            model=model,
            transformer=transformer,
            terminator=terminator,
            pipeline_mode=pipeline_mode,
            source_mode=source_mode,
            max_events=max_events,
        )
    elif task_type == 'outcome':
        outcome_extractor = create_outcome_extractor(config)

        return OutcomePredictionPipeline(
            model=model,
            transformer=transformer,
            terminator=terminator,
            pipeline_mode=pipeline_mode,
            source_mode=source_mode,
            outcome_extractor=outcome_extractor,
            max_events=max_events,
        )
    elif task_type == 'remaining_time':
        target = TimeTarget(config.pop('time_target', None))

        return RemainingTimePredictionPipeline(
            model=model,
            transformer=transformer,
            terminator=terminator,
            pipeline_mode=pipeline_mode,
            source_mode=source_mode,
            target=target,
            max_events=max_events,
        )
    else:
        raise ValueError(f"Unknown task type: '{task_type}'.")
