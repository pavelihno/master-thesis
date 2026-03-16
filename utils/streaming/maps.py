import functools
import traceback
from abc import abstractmethod
from typing import Any

from pybeamline.bevent import BEvent
from pybeamline.stream.base_map import BaseMap
from river import metrics as river_metrics

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


class EmptyOperator(BaseMap):
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
    def _get_target(self, event: BEvent) -> Any:
        """Extract task target from an event."""
        pass

    def _build_metadata(
        self,
        event: BEvent,
        trace_n: int,
        prefix_len: int,
    ) -> dict[str, Any]:
        return {
            'trace_n': trace_n,
            'prefix_len': prefix_len,
            'event_time': str(event.get_event_time()),
        }

    @catch_and_reraise
    def transform(self, event: BEvent) -> list[tuple[str, dict, Any, dict]] | None:
        trace_id = event.get_trace_name()
        y_true = self._get_target(event)

        is_first = trace_id not in self._trace_index
        is_end = y_true in self._end_events

        if is_first:
            self._trace_n += 1
            self._trace_index[trace_id] = self._trace_n

        self._transformer.update(trace_id, event)
        prefix_len = self._transformer.prefix_len(trace_id)
        trace_n = self._trace_index[trace_id]

        if not is_end:
            features = self._transformer.get_features(trace_id)
        else:
            features = {}

        if is_end:
            self._transformer.clear(trace_id)

        metadata = self._build_metadata(event, trace_n=trace_n, prefix_len=prefix_len)

        return [(trace_id, features, y_true, metadata)]


class NextActivityEmitterMap(EmitterMap):
    def _get_target(self, event: BEvent) -> str:
        return event.get_event_name()


class PredictorMap(BaseMap):
    def __init__(self, model):
        self._model = model

    @catch_and_reraise
    def transform(
        self, item: tuple[str, dict, any, dict]
    ) -> list[tuple[str, dict, any, any, dict]] | None:
        trace_id, features, y_true, metadata = item

        if features:
            y_pred = self._model.predict_one(features)
        else:
            # No prediction if no features
            y_pred = None

        return [(trace_id, features, y_true, y_pred, metadata)]


class LearnerMap(BaseMap):
    def __init__(self, model):
        self._model = model
        self._has_drift_detector = hasattr(model, 'drift_detector')

        # Features from the previous event per trace
        self._pending_features: dict[str, dict] = {}

    @catch_and_reraise
    def transform(
        self, item: tuple[str, dict, any, any, dict]
    ) -> list[tuple[str, dict, any, any, dict]] | None:
        trace_id, features, y_true, y_pred, metadata = item

        drift_detected = False

        # Learn on features from previous event of the same trace
        if trace_id in self._pending_features:
            prev_features = self._pending_features[trace_id]
            self._model.learn_one(prev_features, y_true)
            drift_detected = (
                self._has_drift_detector and self._model.drift_detector.drift_detected
            )

        # Store current features for the next event of the same trace
        if features:
            self._pending_features[trace_id] = features
        else:
            self._pending_features.pop(trace_id, None)

        return [
            (
                trace_id,
                features,
                y_true,
                y_pred,
                {**metadata, 'drift_detected': drift_detected},
            )
        ]


class PrequentialClassifierMap(BaseMap):
    """
    Prequential (test-then-train) evaluation map.

    Receives (features, y_true, metadata), predicts, updates metrics,
    then trains on the sample.
    """

    def __init__(self, model, model_name: str = ''):
        self._model = model
        self._name = model_name
        self._acc = river_metrics.Accuracy()
        self._f1 = river_metrics.MacroF1()
        self._n_pred = 0
        self._n_drifts = 0
        self._has_drift_detector = hasattr(model, 'drift_detector')

    @catch_and_reraise
    def transform(self, item: tuple[str, dict, str, dict]) -> list[dict] | None:
        trace_id, features, y_true, metadata = item

        y_pred = self._model.predict_one(features)
        if y_pred is not None:
            self._acc.update(y_true, y_pred)
            self._f1.update(y_true, y_pred)
            self._n_pred += 1

        self._model.learn_one(features, y_true)

        if self._has_drift_detector and self._model.drift_detector.drift_detected:
            self._n_drifts += 1

        return [
            {
                'n_pred': self._n_pred,
                'y_true': y_true,
                'y_pred': y_pred,
                'accuracy': self._acc.get(),
                'macro_f1': self._f1.get(),
                'n_drifts': self._n_drifts,
                **metadata,
            }
        ]
