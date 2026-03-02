from river.drift import ADWIN
from river.ensemble import SRPClassifier
from river.forest import ARFClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier

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
        raise ValueError(
            f"Unknown transformer type: '{transformer_type}'. "
            'Expected one of: cf, data, index, dim.'
        )


def create_model(config: dict):
    """Create a river streaming classifier from a config dict."""
    model_type = config['type']
    params = config.get('params', {})

    if model_type == 'srp':
        _inject_adwin(params)
        return SRPClassifier(**params)
    elif model_type == 'arf':
        _inject_adwin(params)
        return ARFClassifier(**params)
    elif model_type == 'aht':
        _inject_adwin(params)
        return HoeffdingAdaptiveTreeClassifier(**params)
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Expected one of: srp, arf, aht."
        )


def _inject_adwin(params: dict) -> None:
    """Replace ADWIN config values with ADWIN objects in-place."""
    for key in ('drift_detector', 'warning_detector'):
        val = params.get(key)
        if val is None:
            continue
        elif isinstance(val, dict):
            if not val.get('include', True):
                del params[key]
            else:
                adwin_params = {k: v for k, v in val.items() if k != 'include'}
                params[key] = ADWIN(**adwin_params)
