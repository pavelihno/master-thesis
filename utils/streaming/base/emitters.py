from abc import abstractmethod
from typing import Any

from pybeamline.bevent import BEvent
from pybeamline.stream.base_map import BaseMap

from utils.streaming.base.extractors import OutcomeExtractor
from utils.streaming.base.maps import catch_and_reraise
from utils.streaming.base.transformers import StreamingTransformer


class EmitterMap(BaseMap):
    def __init__(
        self,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
    ):
        self._transformer = transformer
        self._end_events: set[str] = end_events or set()
        self._trace_n: int = 0
        self._event_n: int = 0
        self._trace_index: dict[str, int] = {}

    @abstractmethod
    def transform(self, event: BEvent) -> list[tuple[str, dict, Any, Any, dict]] | None:
        pass

    def _next_event_n(self) -> int:
        self._event_n += 1
        return self._event_n

    def _build_metadata(
        self,
        event: BEvent,
        trace_n: int,
        event_n: int,
        prefix_len: int,
        is_end: bool,
    ) -> dict[str, Any]:
        return {
            'trace_n': trace_n,
            'event_n': event_n,
            'prefix_len': prefix_len,
            'event_time': str(event.get_event_time()),
            'is_end': is_end,
        }


class NextActivityEmitter(EmitterMap):
    @catch_and_reraise
    def transform(self, event: BEvent) -> list[tuple[str, dict, Any, Any, dict]] | None:
        trace_id = event.get_trace_name()
        event_name = event.get_event_name()

        y_true = event_name

        is_first = trace_id not in self._trace_index
        is_end = event_name in self._end_events

        if is_first:
            self._trace_n += 1
            self._trace_index[trace_id] = self._trace_n

        self._transformer.update(trace_id, event)
        prefix_len = self._transformer.prefix_len(trace_id)
        trace_n = self._trace_index[trace_id]
        event_n = self._next_event_n()

        if is_end:
            features = {}
            self._transformer.clear(trace_id)
        else:
            features = self._transformer.get_features(trace_id)

        metadata = self._build_metadata(
            event,
            trace_n=trace_n,
            event_n=event_n,
            prefix_len=prefix_len,
            is_end=is_end,
        )

        return [(trace_id, features, y_true, None, metadata)]


class OutcomeEmitter(EmitterMap):
    def __init__(
        self,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
        outcome_extractor: OutcomeExtractor | None = None,
    ):
        super().__init__(transformer=transformer, end_events=end_events)

        self._outcome_extractor = outcome_extractor
        self._outcomes: dict[str, int | None] = {}

    @catch_and_reraise
    def transform(self, event: BEvent) -> list[tuple[str, dict, Any, Any, dict]] | None:
        trace_id = event.get_trace_name()
        event_name = event.get_event_name()

        is_first = trace_id not in self._trace_index
        is_end = event_name in self._end_events

        self._transformer.update(trace_id, event)

        if trace_id in self._outcomes:
            y_true = self._outcomes[trace_id]
            features = {}

        elif self._outcome_extractor:
            y_true = self._outcome_extractor.extract(trace_id, event)

            if y_true is not None:
                features = {}
                self._outcomes[trace_id] = y_true
            else:
                features = self._transformer.get_features(trace_id)

        else:
            raise ValueError('OutcomeEmitter requires outcomes.')

        if is_first:
            self._trace_n += 1
            self._trace_index[trace_id] = self._trace_n

        prefix_len = self._transformer.prefix_len(trace_id)
        trace_n = self._trace_index[trace_id]
        event_n = self._next_event_n()

        if is_end:
            self._transformer.clear(trace_id)
            self._outcomes.pop(trace_id, None)

        metadata = self._build_metadata(
            event,
            trace_n=trace_n,
            event_n=event_n,
            prefix_len=prefix_len,
            is_end=is_end,
        )

        return [(trace_id, features, y_true, None, metadata)]


class RemainingTimeEmitter(EmitterMap):
    @catch_and_reraise
    def transform(self, event: BEvent) -> list[tuple[str, dict, Any, Any, dict]] | None:
        trace_id = event.get_trace_name()
        event_name = event.get_event_name()

        is_first = trace_id not in self._trace_index
        is_end = event_name in self._end_events

        if is_first:
            self._trace_n += 1
            self._trace_index[trace_id] = self._trace_n

        self._transformer.update(trace_id, event)
        prefix_len = self._transformer.prefix_len(trace_id)
        trace_n = self._trace_index[trace_id]
        event_n = self._next_event_n()

        if is_end:
            y_true = event.get_event_time()
            features = {}
            self._transformer.clear(trace_id)
        else:
            y_true = None
            features = self._transformer.get_features(trace_id)

        metadata = self._build_metadata(
            event,
            trace_n=trace_n,
            event_n=event_n,
            prefix_len=prefix_len,
            is_end=is_end,
        )

        return [(trace_id, features, y_true, None, metadata)]
