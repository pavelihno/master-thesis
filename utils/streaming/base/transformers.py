from abc import ABC, abstractmethod
from collections import Counter, deque
from datetime import datetime
from typing import Any

from pybeamline.bevent import BEvent

from utils.streaming.time import TimeTarget, convert_time, parse_time


_EVENT_SKIP = {'concept:name', 'time:timestamp'}
_TRACE_SKIP = {'concept:name'}


def _event_data(event: BEvent) -> dict[str, Any]:
    """Event-level data attributes (event name and timestamp excluded)."""
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

    Stores a counter of event occurrences per trace.
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

    Stores last max_events event names per trace in a deque.
    - Static part: trace-level attributes encoded once
    - Dynamic part: event name per position
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
        self._event_deques[trace_id].append(event.get_event_name())
        self._prefix_lens[trace_id] += 1

    def get_features(self, trace_id: str) -> dict[str, Any]:
        features: dict[str, Any] = {}

        for k, v in self._trace_attrs.get(trace_id, {}).items():
            _encode_value(features, f'trace_{k}', v)

        for i, name in enumerate(reversed(self._event_deques.get(trace_id, deque()))):
            features[f'act_{i}'] = name

        if self._include_prefix_len:
            features['prefix_len'] = self.prefix_len(trace_id)

        return features

    def clear(self, trace_id: str) -> None:
        self._trace_attrs.pop(trace_id, None)
        self._event_deques.pop(trace_id, None)
        self._prefix_lens.pop(trace_id, None)


class DimensionTransformer(StreamingTransformer):
    """
    Dimension encoding.

    Stores last max_events (event_name, event_data) tuples per trace in a deque.
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


class FrequencyBasedTransformer(StreamingTransformer):
    """
    Frequency-based encoding.

    Stores a counter of event occurrences per trace.
    Assumes the vocabulary of events (unique_events) is fixed and fully known.

    Optional capabilities:
    - last_events: captures immediate sequential context.
    - include_trace_attrs: captures static trace-level payload.
    """

    def __init__(
        self,
        unique_events: set[str],
        last_events: int = 0,
        include_trace_attrs: bool = False,
        include_prefix_len: bool = True,
    ):
        super().__init__(include_prefix_len)

        self._unique_events = unique_events
        self._last_events = last_events
        self._include_trace_attrs = include_trace_attrs

        self._event_counts: dict[str, dict[str, int]] = {}
        self._event_deques: dict[str, deque] = {}
        self._trace_attrs: dict[str, dict] = {}

    def update(self, trace_id: str, event: 'BEvent') -> None:
        event_name = event.get_event_name()

        if trace_id not in self._event_counts:
            self._event_counts[trace_id] = dict.fromkeys(self._unique_events, 0)
            self._prefix_lens[trace_id] = 0

            if self._last_events > 0:
                self._event_deques[trace_id] = deque(maxlen=self._last_events)

            if self._include_trace_attrs:
                self._trace_attrs[trace_id] = _trace_data(event)

        self._event_counts[trace_id][event_name] += 1

        if self._last_events > 0:
            self._event_deques[trace_id].append(event_name)

        self._prefix_lens[trace_id] += 1

    def get_features(self, trace_id: str) -> dict[str, Any]:
        features: dict[str, Any] = {}

        # Trace-level features
        if self._include_trace_attrs:
            for k, v in self._trace_attrs.get(trace_id, {}).items():
                _encode_value(features, f'trace_{k}', v)

        # Frequency features
        counts = self._event_counts.get(trace_id, dict.fromkeys(self._unique_events, 0))
        for event_name, count in counts.items():
            features[f'count_{event_name}'] = count

        # Local sequence features
        if self._last_events > 0:
            recent_events = self._event_deques.get(trace_id, deque())
            for i, name in enumerate(reversed(recent_events)):
                features[f'last_act_{i}'] = name

        if self._include_prefix_len:
            features['prefix_len'] = self.prefix_len(trace_id)

        return features

    def clear(self, trace_id: str) -> None:
        self._event_counts.pop(trace_id, None)
        self._event_deques.pop(trace_id, None)
        self._trace_attrs.pop(trace_id, None)
        self._prefix_lens.pop(trace_id, None)


class DARWINTransformer(StreamingTransformer):
    """
    Transformer for DARWIN-style streaming models.

    Stores the most recent event per trace.
    """

    def __init__(self) -> None:
        super().__init__(include_prefix_len=False)

        self._last_event: dict[str, tuple[str, int]] = {}
        self._event_id: int = 1

    def update(self, trace_id: str, event: BEvent) -> None:
        if trace_id not in self._last_event:
            self._prefix_lens[trace_id] = 0

        self._last_event[trace_id] = (event.get_event_name(), self._event_id)
        self._prefix_lens[trace_id] += 1
        self._event_id += 1

    def get_features(self, trace_id: str) -> dict[str, Any]:
        event_name, event_id = self._last_event.get(trace_id, ('', 0))
        if not event_name:
            return {}

        return {
            'case_id': trace_id,
            'event_id': event_id,
            'event_name': event_name,
            'features': [],
        }

    def clear(self, trace_id: str) -> None:
        self._last_event.pop(trace_id, None)
        self._prefix_lens.pop(trace_id, None)


class DARWINTimeTransformer(StreamingTransformer):
    """
    Time-based transformer for DARWIN-style streaming models.

    Stores the most recent event per trace along with timestamp-based features.
     - Numerical:
        - duration since case start
        - duration since last event
        - duration since event before last event
     - Categorical:
        - month
        - day
        - week
        - hour
    """

    def __init__(
        self,
        time_target: TimeTarget = TimeTarget.DAYS,
        timestamp_key: str = 'time:timestamp',
    ) -> None:
        super().__init__(include_prefix_len=False)

        self._timestamp_key = timestamp_key
        self._time_target = time_target

        self._last_event: dict[str, tuple[str, datetime | None, int]] = {}
        self._first_event_time: dict[str, datetime | None] = {}
        self._last_before_event_time: dict[str, datetime | None] = {}
        self._before_before_event_time: dict[str, datetime | None] = {}
        self._event_id: int = 1

    def _get_event_time(self, event: BEvent) -> datetime | None:
        time_value = event.event_attributes.get(self._timestamp_key, None)
        return parse_time(time_value)

    def update(self, trace_id: str, event: BEvent) -> None:
        event_name = event.get_event_name()
        event_time = self._get_event_time(event)

        if trace_id not in self._last_event:
            self._prefix_lens[trace_id] = 0
            self._first_event_time[trace_id] = event_time
            self._last_before_event_time[trace_id] = None
            self._before_before_event_time[trace_id] = None
        else:
            _, last_event_time, _ = self._last_event[trace_id]

            self._before_before_event_time[trace_id] = self._last_before_event_time.get(
                trace_id
            )
            self._last_before_event_time[trace_id] = last_event_time

        self._last_event[trace_id] = (event_name, event_time, self._event_id)
        self._prefix_lens[trace_id] += 1
        self._event_id += 1

    def get_features(self, trace_id: str) -> dict[str, Any]:
        event_name, current_event_time, event_id = self._last_event.get(
            trace_id, ('', None, 0)
        )
        if not event_name:
            return {}

        first_event_time = self._first_event_time.get(trace_id)
        last_before_event_time = self._last_before_event_time.get(trace_id)
        before_before_event_time = self._before_before_event_time.get(trace_id)

        if current_event_time is None or first_event_time is None:
            start_time_delta = 0.0
        else:
            start_time_delta = convert_time(
                current_event_time - first_event_time, target=self._time_target
            )

        if current_event_time is None or last_before_event_time is None:
            last_event_time_delta = 0.0
        else:
            last_event_time_delta = convert_time(
                current_event_time - last_before_event_time, target=self._time_target
            )

        if current_event_time is None or before_before_event_time is None:
            before_last_event_time_delta = 0.0
        else:
            before_last_event_time_delta = convert_time(
                current_event_time - before_before_event_time,
                target=self._time_target,
            )

        return {
            'case_id': trace_id,
            'event_id': event_id,
            'event_name': event_name,
            'features': [
                start_time_delta,
                last_event_time_delta,
                before_last_event_time_delta,
            ],
        }

    def clear(self, trace_id: str) -> None:
        self._last_event.pop(trace_id, None)
        self._first_event_time.pop(trace_id, None)
        self._last_before_event_time.pop(trace_id, None)
        self._before_before_event_time.pop(trace_id, None)
        self._prefix_lens.pop(trace_id, None)
