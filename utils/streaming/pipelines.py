import functools
import time
import traceback
from collections import defaultdict

import pandas as pd
from pybeamline.bevent import BEvent
from pybeamline.sources import xes_log_source_from_file
from pybeamline.stream.base_map import BaseMap
from pybeamline.stream.base_sink import BaseSink
from river import metrics as river_metrics
from sklearn.metrics import accuracy_score, f1_score

from utils.streaming.transformers import BaseStreamingTransformer


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

    def __init__(self, transformer: BaseStreamingTransformer):
        self._transformer = transformer
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
        }
        self._transformer.update(trace_id, event)

        # TODO: detect end-of-trace and clear up memory
        #   self._transformer.clear(trace_id)
        #   del self._trace_index[trace_id]

        return [(features, label, metadata)]


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
    def transform(self, item: tuple[dict, str, dict]) -> list[dict] | None:
        features, y_true, metadata = item

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


class PrequentialPipeline:
    """End-to-end prequential streaming pipeline."""

    def __init__(
        self,
        model,
        transformer: BaseStreamingTransformer,
        model_name: str = '',
        rolling_pct: float = 0.2,
    ):
        self.model = model
        self.transformer = transformer
        self.model_name = model_name
        self.rolling_pct = rolling_pct

    def run(self, dataset_path: str) -> tuple[pd.DataFrame, object]:
        sink = CollectorSink()

        start_time = time.perf_counter()
        xes_log_source_from_file(dataset_path).pipe(
            NextActivityEmitterMap(self.transformer),
            PrequentialClassifierMap(self.model, self.model_name),
        ).sink(sink)
        elapsed = time.perf_counter() - start_time

        df = sink.to_dataframe()
        df['time_s'] = elapsed

        predicted = df[df['y_pred'].notna()]
        n_tail = max(1, int(len(predicted) * self.rolling_pct))
        tail = predicted.iloc[-n_tail:]
        df['rolling_accuracy'] = accuracy_score(tail['y_true'], tail['y_pred'])
        df['rolling_macro_f1'] = f1_score(
            tail['y_true'], tail['y_pred'], average='macro', zero_division=0
        )

        return df, self.model
