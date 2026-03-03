from river.drift import ADWIN
from river.ensemble import SRPClassifier
from river.forest import ARFClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier

from utils.streaming.drift_detector import NoDriftDetector
from utils.streaming.transformers import (
    ControlFlowTransformer,
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
