from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from pybeamline.bevent import BEvent


_EVENT_SKIP = {'concept:name', 'time:timestamp'}
_TRACE_SKIP = {'concept:name'}


def _event_data(event: BEvent) -> dict[str, Any]:
    """Event-level data attributes (activity name and timestamp excluded)."""
    return {k: v for k, v in event.event_attributes.items() if k not in _EVENT_SKIP}


def _trace_data(event: BEvent) -> dict[str, Any]:
    """Trace-level (static) attributes (trace id excluded)."""
    return {k: v for k, v in event.trace_attributes.items() if k not in _TRACE_SKIP}


def _encode_value(features: dict, key: str, value: Any) -> None:
    """Add a single attribute value to the feature dict."""
    if isinstance(value, (int, float)):
        features[key] = value
    else:
        features[f'{key}_{value}'] = 1


class BaseStreamingTransformer(ABC):
    """Base class for streaming feature extractors."""

    @abstractmethod
    def transform(self, events: list[BEvent]) -> dict[str, Any]:
        """Extract features from a trace prefix."""
        pass


class ControlFlowTransformer(BaseStreamingTransformer):
    """
    Control flow only encoding.

    Feature vector contains:
    1) executed activities encoded as a occurrence counts
    2) prefix length.
    """

    def transform(self, events: list[BEvent]) -> dict[str, Any]:
        counts: dict[str, int] = defaultdict(int)
        for e in events:
            counts[f'act_{e.get_event_name()}'] += 1

        features: dict[str, Any] = dict(counts)
        features['prefix_len'] = len(events)

        return features


class DataTransformer(BaseStreamingTransformer):
    """
    Data only encoding.

    Aggregation per attribute over the prefix:
    - Numeric: last observed value + mean over the prefix
    - Categorical: one-hot of the last observed value
    - Trace-level: included once (one-hot or numeric)
    """

    def transform(self, events: list[BEvent]) -> dict[str, Any]:
        features: dict[str, Any] = {}

        if not events:
            features['prefix_len'] = 0
            return features

        # Trace-level static attributes
        for k, v in _trace_data(events[0]).items():
            _encode_value(features, f'trace_{k}', v)

        # Event-level data attributes over the prefix
        numeric_vals: dict[str, list[float]] = defaultdict(list)
        categorical_last: dict[str, Any] = {}

        for e in events:
            for k, v in _event_data(e).items():
                if isinstance(v, (int, float)):
                    numeric_vals[k].append(float(v))
                else:
                    categorical_last[k] = v

        for k, vals in numeric_vals.items():
            features[f'data_{k}_last'] = vals[-1]
            features[f'data_{k}_mean'] = sum(vals) / len(vals)

        for k, v in categorical_last.items():
            features[f'data_{k}_{v}'] = 1

        features['prefix_len'] = len(events)

        return features


class IndexBasedTransformer(BaseStreamingTransformer):
    """
    Index-based encoding.

    Combines control-flow and data, reduce dimensionality by
    separating static data from dynamic:
    - Static part: trace-level attributes encoded once
    - Dynamic part: activity name one-hot encoded per position
    """

    def __init__(self, max_events: int | None = None):
        self._max_events = max_events

    def transform(self, events: list[BEvent]) -> dict[str, Any]:
        features: dict[str, Any] = {}

        if not events:
            features['prefix_len'] = 0
            return features

        # Trace-level static attributes
        for k, v in _trace_data(events[0]).items():
            _encode_value(features, f'trace_{k}', v)

        # Activity at each position (last max_events positions)
        max_events = self._max_events if self._max_events is not None else len(events)
        for i, e in enumerate(events[:-max_events-1:-1]):
            features[f'act_{i}'] = e.get_event_name()

        features['prefix_len'] = len(events)

        return features


class DimensionTransformer(BaseStreamingTransformer):
    """
    Dimension (full) encoding.

    Full combination of control-flow and data features.
    """

    def __init__(self, max_events: int | None = None):
        self._max_events = max_events

    def transform(self, events: list[BEvent]) -> dict[str, Any]:
        features: dict[str, Any] = {}

        if not events:
            features['prefix_len'] = 0
            return features

        # Trace-level static attributes
        for k, v in _trace_data(events[0]).items():
            _encode_value(features, f'trace_{k}', v)

        # Activity + data at each position (last max_events positions)
        max_events = self._max_events if self._max_events is not None else len(events)
        for i, e in enumerate(events[:-max_events-1:-1]):
            features[f'act_{i}'] = e.get_event_name()

            # Event-level data attributes at each position
            for k, v in _event_data(e).items():
                _encode_value(features, f'data_{i}_{k}', v)

        features['prefix_len'] = len(events)

        return features
