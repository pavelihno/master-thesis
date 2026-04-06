import functools
import traceback
from abc import abstractmethod
from collections import defaultdict
from typing import Any

from pybeamline.bevent import BEvent
from pybeamline.stream.base_map import BaseMap

from utils.streaming.extractors import OutcomeExtractor
from utils.streaming.transformers import StreamingTransformer


def catch_and_reraise(method):
    """Decorator for catching exceptions in `transform`."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            print(f'[ERROR] {self.__class__.__name__}.{method.__name__} crashed')
            traceback.print_exc()
            raise

    return wrapper


class EmptyMap(BaseMap):
    def transform(self, item):
        return [item]


class EmitterMap(BaseMap):
    def __init__(
        self,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
    ):
        self._transformer = transformer
        self._end_events: set[str] = end_events or set()
        self._trace_n: int = 0
        self._trace_index: dict[str, int] = {}

    @abstractmethod
    def transform(self, event: BEvent) -> list[tuple[str, dict, Any, Any, dict]] | None:
        pass

    def _build_metadata(
        self,
        event: BEvent,
        trace_n: int,
        prefix_len: int,
        is_end: bool,
    ) -> dict[str, Any]:
        return {
            'trace_n': trace_n,
            'prefix_len': prefix_len,
            'event_time': str(event.get_event_time()),
            'is_end': is_end,
        }


class NextActivityEmitterMap(EmitterMap):
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

        if is_end:
            features = {}
            self._transformer.clear(trace_id)
        else:
            features = self._transformer.get_features(trace_id)

        metadata = self._build_metadata(
            event, trace_n=trace_n, prefix_len=prefix_len, is_end=is_end
        )

        return [(trace_id, features, y_true, None, metadata)]


class OutcomeEmitterMap(EmitterMap):
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
            raise ValueError('OutcomeEmitterMap requires outcomes.')

        if is_first:
            self._trace_n += 1
            self._trace_index[trace_id] = self._trace_n

        prefix_len = self._transformer.prefix_len(trace_id)
        trace_n = self._trace_index[trace_id]

        if is_end:
            self._transformer.clear(trace_id)
            self._outcomes.pop(trace_id, None)

        metadata = self._build_metadata(
            event, trace_n=trace_n, prefix_len=prefix_len, is_end=is_end
        )

        return [(trace_id, features, y_true, None, metadata)]


class PredictorMap(BaseMap):
    def __init__(self, model):
        self._model = model

    def _get_loss(self):
        return (
            self._model.loss_history[-1]
            if hasattr(self._model, 'loss_history') and self._model.loss_history
            else None
        )

    @catch_and_reraise
    def transform(
        self, item: tuple[str, dict, Any, dict]
    ) -> list[tuple[str, dict, Any, Any, dict]] | None:
        trace_id, features, y_true, _, metadata = item

        prefix_len = metadata.get('prefix_len', -1)

        # Predict only if features are provided
        if features:
            # TODO: include prefix_len by wrapping model for predicting
            y_pred = self._model.predict_one(features)
        else:
            y_pred = None

        loss = self._get_loss()

        metadata = {**metadata, 'loss': loss}

        return [(trace_id, features, y_true, y_pred, metadata)]


class LearnerMap(BaseMap):
    def __init__(self, model):
        self._model = model
        self._has_drift_detector = hasattr(model, 'drift_detector')

    def _get_loss(self):
        return (
            self._model.loss_history[-1]
            if hasattr(self._model, 'loss_history') and self._model.loss_history
            else None
        )

    def _is_drift_detected(self):
        return self._has_drift_detector and self._model.drift_detector.drift_detected

    @abstractmethod
    def transform(
        self, item: tuple[str, dict, Any, Any, dict]
    ) -> list[tuple[str, dict, Any, Any, dict]] | None:
        pass


class NextActivityLearnerMap(LearnerMap):
    def __init__(self, model):
        super().__init__(model)

        # Features from the previous event per trace
        self._pending_features: dict[str, dict] = {}

    @catch_and_reraise
    def transform(
        self, item: tuple[str, dict, Any, Any, dict]
    ) -> list[tuple[str, dict, Any, Any, dict]] | None:
        trace_id, features, y_true, y_pred, metadata = item

        drift_detected = False

        # Learn on features from previous event of the same trace
        if trace_id in self._pending_features:
            prev_features = self._pending_features[trace_id]
            self._model.learn_one(prev_features, y_true)
            drift_detected = self._is_drift_detected()

        # Store current features for the next event of the same trace
        if features:
            self._pending_features[trace_id] = features
        else:
            self._pending_features.pop(trace_id, None)

        loss = self._get_loss()

        metadata = {**metadata, 'drift_detected': drift_detected, 'loss': loss}

        return [(trace_id, features, y_true, y_pred, metadata)]


class OutcomeLearnerMap(LearnerMap):
    def __init__(self, model):
        super().__init__(model)

        # Features before knowing the outcome per trace
        self._pending_features: dict[str, list[tuple[int, dict]]] = defaultdict(list)

    @catch_and_reraise
    def transform(
        self, item: tuple[str, dict, Any, Any, dict]
    ) -> list[tuple[str, dict, Any, Any, dict]] | None:
        trace_id, features, y_true, y_pred, metadata = item

        prefix_len = metadata.get('prefix_len', 0)
        is_end = metadata.get('is_end', False)

        drift_detected = False

        # Learn once the trace outcome is known
        if y_true is not None:
            trace_features = self._pending_features.pop(trace_id, [])
            for pending_prefix_len, pending_features in trace_features:
                # TODO: include prefix_len by wrapping model for learning

                # print(
                #     f'LEARN> trace_id={trace_id}, prefix_len={pending_prefix_len}, features={pending_features}, y_true={y_true}'
                # )

                self._model.learn_one(pending_features, y_true)
                drift_detected = drift_detected or self._is_drift_detected()

        else:
            # Store features before outcome is known
            if features:
                self._pending_features[trace_id].append((prefix_len, features))

            # Cleanup for traces that end without known outcome
            if is_end:
                self._pending_features.pop(trace_id, None)

        loss = self._get_loss()

        metadata = {**metadata, 'drift_detected': drift_detected, 'loss': loss}

        return [(trace_id, features, y_true, y_pred, metadata)]
