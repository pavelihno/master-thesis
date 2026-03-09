import functools
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
from pybeamline.bevent import BEvent
from pybeamline.sources import xes_log_source_from_file
from pybeamline.stream.base_map import BaseMap
from pybeamline.stream.base_sink import BaseSink
from river import metrics as river_metrics
from river.base.estimator import Estimator

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


class NextActivityEmitterMap(BaseMap):
    """
    Emits (features, next_activity, metadata) tuples.
    """

    def __init__(
        self,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
    ):
        self._transformer = transformer
        self._end_events: set[str] = end_events or set()
        self._trace_n: int = 0
        self._trace_index: dict[str, int] = {}

    @catch_and_reraise
    def transform(self, event: BEvent) -> list[tuple[dict, str, dict]] | None:
        trace_id = event.get_trace_name()

        is_first = trace_id not in self._trace_index
        if is_first:
            self._trace_n += 1
            self._trace_index[trace_id] = self._trace_n
            self._transformer.update(trace_id, event)
            return None

        features = self._transformer.get_features(trace_id)
        label = event.get_event_name()
        metadata = {
            'trace_id': trace_id,
            'trace_n': self._trace_index[trace_id],
            'prefix_len': self._transformer.prefix_len(trace_id),
            'event_time': str(event.get_event_time()),
            'y_true': label,
        }
        self._transformer.update(trace_id, event)

        if label in self._end_events:
            self._transformer.clear(trace_id)
            # del self._trace_index[trace_id]

        return [(features, label, metadata)]


class PredictorMap(BaseMap):
    def __init__(self, model):
        self._model = model

    @catch_and_reraise
    def transform(
        self, item: tuple[dict, any, dict]
    ) -> list[tuple[dict, any, any, dict]] | None:
        features, y_true, metadata = item
        y_pred = self._model.predict_one(features)
        return [(features, y_true, y_pred, metadata)]


class LearnerMap(BaseMap):
    def __init__(self, model):
        self._model = model
        self._has_drift_detector = hasattr(model, 'drift_detector')

    @catch_and_reraise
    def transform(
        self, item: tuple[dict, any, any, dict]
    ) -> list[tuple[dict, any, any, dict]] | None:
        features, y_true, y_pred, metadata = item

        self._model.learn_one(features, y_true)
        drift_detected = (
            self._has_drift_detector and self._model.drift_detector.drift_detected
        )

        return [
            (features, y_true, y_pred, {**metadata, 'drift_detected': drift_detected})
        ]


class CollectorSink(BaseSink):
    """Accumulates all emitted records into an in-memory list."""

    def __init__(self):
        self.records: list[dict] = []

    def consume(self, item: dict) -> None:
        self.records.append(item)

    def close(self) -> None:
        pass

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


class EvaluatorSink(CollectorSink):
    """Sink that evaluates predictions in a prequential manner."""

    def __init__(self):
        super().__init__()
        self._n_pred: int = 0
        self._n_drifts: int = 0
        self._pending: dict[str, tuple[any, dict]] = {}
        self._metrics: dict = self._make_metrics()

    def _make_metrics(self) -> dict:
        raise NotImplementedError

    def _update_metrics(self, y_true, y_pred) -> dict:
        raise NotImplementedError

    def _current_metric_values(self) -> dict:
        return {name: m.get() for name, m in self._metrics.items()}

    def consume(self, item: tuple[dict, any, any, dict]) -> None:
        features, y_true, y_pred, metadata = item
        trace_id = metadata['trace_id']

        drift_detected = metadata.pop('drift_detected', False)
        if drift_detected:
            self._n_drifts += 1

        if trace_id in self._pending:
            prev_y_pred, prev_metadata = self._pending[trace_id]
            prev_y_true = prev_metadata['y_true']

            if prev_y_pred is not None:
                metric_vals = self._update_metrics(prev_y_true, prev_y_pred)
                self._n_pred += 1
            else:
                metric_vals = self._current_metric_values()

            self.records.append(
                {
                    'n_pred': self._n_pred,
                    'y_true': prev_y_true,
                    'y_pred': prev_y_pred,
                    'n_drifts': self._n_drifts,
                    **metric_vals,
                    **prev_metadata,
                }
            )

        self._pending[trace_id] = (y_pred, metadata)

    def close(self) -> None:
        """Flush remaining pending predictions after the stream ends."""
        for trace_id, (y_pred, metadata) in self._pending.items():
            self.records.append(
                {
                    'n_pred': self._n_pred,
                    'y_true': None,
                    'y_pred': y_pred,
                    'n_drifts': self._n_drifts,
                    **self._current_metric_values(),
                    **metadata,
                }
            )
        self._pending.clear()


class ClassificationEvaluatorSink(EvaluatorSink):
    def _make_metrics(self) -> dict:
        return {
            'accuracy': river_metrics.Accuracy(),
            'macro_f1': river_metrics.MacroF1(),
        }

    def _update_metrics(self, y_true, y_pred) -> dict:
        for m in self._metrics.values():
            m.update(y_true, y_pred)
        return self._current_metric_values()


class RegressionEvaluatorSink(EvaluatorSink):
    def _make_metrics(self) -> dict:
        return {
            'mae': river_metrics.MAE(),
            'rmse': river_metrics.RMSE(),
        }

    def _update_metrics(self, y_true, y_pred) -> dict:
        for m in self._metrics.values():
            m.update(y_true, y_pred)
        return self._current_metric_values()


class TraceCollectorSink(BaseSink):
    """Groups BEvents by trace_id into a dict."""

    def __init__(self):
        self.traces: dict[str, list[BEvent]] = defaultdict(list)

    def consume(self, event: BEvent) -> None:
        self.traces[event.get_trace_name()].append(event)

    def close(self) -> None:
        pass

    def to_dict(self) -> dict[str, list[BEvent]]:
        return dict(self.traces)


class TaskPipeline(ABC):
    def __init__(
        self,
        model: Estimator,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
    ):
        self.model = model
        self.transformer = transformer
        self.end_events = end_events

    @abstractmethod
    def run(self, dataset_path: str) -> tuple[pd.DataFrame, object]:
        pass


class NextActivityPredictionPipeline(TaskPipeline):
    def __init__(
        self,
        model,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
    ):
        super().__init__(model, transformer, end_events)

    def run(self, dataset_path: str) -> tuple[pd.DataFrame, object]:
        sink = ClassificationEvaluatorSink()

        start_time = time.perf_counter()

        xes_log_source_from_file(dataset_path).pipe(
            NextActivityEmitterMap(self.transformer, end_events=self.end_events),
            PredictorMap(self.model),
            LearnerMap(self.model),
        ).sink(sink)

        elapsed_time = time.perf_counter() - start_time

        df = sink.to_dataframe()
        df['time_s'] = elapsed_time

        return df, self.model
