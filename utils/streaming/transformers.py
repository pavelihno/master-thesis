from abc import ABC, abstractmethod
from collections import Counter, deque
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


class StreamingTransformer(ABC):
    """Base class for incremental streaming feature extractors."""

    def __init__(self, include_prefix_len: bool = False):
        self._include_prefix_len: bool = include_prefix_len

        self._prefix_lens: dict[str, int] = {}

    @abstractmethod
    def update(self, trace_id: str, event: BEvent) -> None:
        """Incrementally update per-trace state with a new event."""
        pass

    @abstractmethod
    def get_features(self, trace_id: str) -> dict[str, Any]:
        """Extract the current feature vector for a trace prefix."""
        pass

    def prefix_len(self, trace_id: str) -> int:
        """Return the number of events seen so far for a trace."""
        return self._prefix_lens.get(trace_id, 0)

    @abstractmethod
    def clear(self, trace_id: str) -> None:
        """Free per-trace state once the trace is complete."""
        pass


class ControlFlowTransformer(StreamingTransformer):
    """
    Control-flow only encoding.

    Stores a counter of activity occurrences per trace.
    """

    def __init__(self, include_prefix_len: bool = True):
        super().__init__(include_prefix_len)

        self._counts: dict[str, Counter] = {}

    def update(self, trace_id: str, event: BEvent) -> None:
        if trace_id not in self._counts:
            self._counts[trace_id] = Counter()
            self._prefix_lens[trace_id] = 0
        self._counts[trace_id][f'act_{event.get_event_name()}'] += 1
        self._prefix_lens[trace_id] += 1

    def get_features(self, trace_id: str) -> dict[str, Any]:
        features: dict[str, Any] = dict(self._counts.get(trace_id, {}))

        if self._include_prefix_len:
            features['prefix_len'] = self.prefix_len(trace_id)

        return features

    def clear(self, trace_id: str) -> None:
        self._counts.pop(trace_id, None)
        self._prefix_lens.pop(trace_id, None)


class DataTransformer(StreamingTransformer):
    """
    Data only encoding.

    Stores running aggregates per trace:
    - Numeric: running sum + count for mean, last observed value
    - Categorical: last observed value per attribute
    - Trace-level: captured from the first event
    """

    def __init__(self, include_prefix_len: bool = True):
        super().__init__(include_prefix_len)

        self._trace_attrs: dict[str, dict] = {}
        self._numeric_sums: dict[str, dict[str, float]] = {}
        self._numeric_counts: dict[str, dict[str, int]] = {}
        self._numeric_last: dict[str, dict[str, float]] = {}
        self._categorical_last: dict[str, dict[str, Any]] = {}

    def update(self, trace_id: str, event: BEvent) -> None:
        if trace_id not in self._trace_attrs:
            self._trace_attrs[trace_id] = _trace_data(event)
            self._numeric_sums[trace_id] = {}
            self._numeric_counts[trace_id] = {}
            self._numeric_last[trace_id] = {}
            self._categorical_last[trace_id] = {}
            self._prefix_lens[trace_id] = 0

        for k, v in _event_data(event).items():
            if isinstance(v, (int, float)):
                float_v = float(v)
                self._numeric_sums[trace_id][k] = (
                    self._numeric_sums[trace_id].get(k, 0.0) + float_v
                )
                self._numeric_counts[trace_id][k] = (
                    self._numeric_counts[trace_id].get(k, 0) + 1
                )
                self._numeric_last[trace_id][k] = float_v
            else:
                self._categorical_last[trace_id][k] = v

        self._prefix_lens[trace_id] += 1

    def get_features(self, trace_id: str) -> dict[str, Any]:
        features: dict[str, Any] = {}

        for k, v in self._trace_attrs.get(trace_id, {}).items():
            _encode_value(features, f'trace_{k}', v)

        for k, last_v in self._numeric_last.get(trace_id, {}).items():
            features[f'data_{k}_last'] = last_v
            count = self._numeric_counts[trace_id][k]
            features[f'data_{k}_mean'] = self._numeric_sums[trace_id][k] / count

        for k, v in self._categorical_last.get(trace_id, {}).items():
            features[f'data_{k}_{v}'] = 1

        if self._include_prefix_len:
            features['prefix_len'] = self.prefix_len(trace_id)

        return features

    def clear(self, trace_id: str) -> None:
        self._trace_attrs.pop(trace_id, None)
        self._numeric_sums.pop(trace_id, None)
        self._numeric_counts.pop(trace_id, None)
        self._numeric_last.pop(trace_id, None)
        self._categorical_last.pop(trace_id, None)
        self._prefix_lens.pop(trace_id, None)


class IndexBasedTransformer(StreamingTransformer):
    """
    Index-based encoding.

    Stores last max_events activity names per trace in a deque.
    - Static part: trace-level attributes encoded once
    - Dynamic part: activity name per position
    """

    def __init__(self, max_events: int = 10, include_prefix_len: bool = True):
        super().__init__(include_prefix_len)

        self._max_events = max_events
        self._trace_attrs: dict[str, dict] = {}
        self._activity_deques: dict[str, deque] = {}

    def update(self, trace_id: str, event: BEvent) -> None:
        if trace_id not in self._trace_attrs:
            self._trace_attrs[trace_id] = _trace_data(event)
            self._activity_deques[trace_id] = deque(maxlen=self._max_events)
            self._prefix_lens[trace_id] = 0
        self._activity_deques[trace_id].append(event.get_event_name())
        self._prefix_lens[trace_id] += 1

    def get_features(self, trace_id: str) -> dict[str, Any]:
        features: dict[str, Any] = {}

        for k, v in self._trace_attrs.get(trace_id, {}).items():
            _encode_value(features, f'trace_{k}', v)

        for i, name in enumerate(
            reversed(self._activity_deques.get(trace_id, deque()))
        ):
            features[f'act_{i}'] = name

        if self._include_prefix_len:
            features['prefix_len'] = self.prefix_len(trace_id)

        return features

    def clear(self, trace_id: str) -> None:
        self._trace_attrs.pop(trace_id, None)
        self._activity_deques.pop(trace_id, None)
        self._prefix_lens.pop(trace_id, None)


class DimensionTransformer(StreamingTransformer):
    """
    Dimension encoding.

    Stores last max_events (activity_name, event_data) tuples per trace in a deque.
    Full combination of control-flow and data features.
    """

    def __init__(self, max_events: int = 10, include_prefix_len: bool = True):
        super().__init__(include_prefix_len)

        self._max_events = max_events
        self._trace_attrs: dict[str, dict] = {}
        self._event_deques: dict[str, deque] = {}

    def update(self, trace_id: str, event: BEvent) -> None:
        if trace_id not in self._trace_attrs:
            self._trace_attrs[trace_id] = _trace_data(event)
            self._event_deques[trace_id] = deque(maxlen=self._max_events)
            self._prefix_lens[trace_id] = 0
        self._event_deques[trace_id].append(
            (event.get_event_name(), _event_data(event))
        )
        self._prefix_lens[trace_id] += 1

    def get_features(self, trace_id: str) -> dict[str, Any]:
        features: dict[str, Any] = {}

        for k, v in self._trace_attrs.get(trace_id, {}).items():
            _encode_value(features, f'trace_{k}', v)

        for i, (name, data) in enumerate(
            reversed(self._event_deques.get(trace_id, deque()))
        ):
            features[f'act_{i}'] = name
            for k, v in data.items():
                _encode_value(features, f'data_{i}_{k}', v)

        if self._include_prefix_len:
            features['prefix_len'] = self.prefix_len(trace_id)

        return features

    def clear(self, trace_id: str) -> None:
        self._trace_attrs.pop(trace_id, None)
        self._event_deques.pop(trace_id, None)
        self._prefix_lens.pop(trace_id, None)


class DARWINTransformer(StreamingTransformer):
    """
    Transformer for DARWIN-style streaming models.

    Stores the most recent activity name per trace and emits:
    {'case_id': trace_id, 'activity': last_activity_name}
    """

    def __init__(self) -> None:
        super().__init__(include_prefix_len=False)
        self._last_activity: dict[str, str] = {}

    def update(self, trace_id: str, event: BEvent) -> None:
        if trace_id not in self._last_activity:
            self._prefix_lens[trace_id] = 0
        self._last_activity[trace_id] = event.get_event_name()
        self._prefix_lens[trace_id] += 1

    def get_features(self, trace_id: str) -> dict[str, Any]:
        return {
            'case_id': trace_id,
            'activity': self._last_activity.get(trace_id, ''),
        }

    def clear(self, trace_id: str) -> None:
        self._last_activity.pop(trace_id, None)
        self._prefix_lens.pop(trace_id, None)
