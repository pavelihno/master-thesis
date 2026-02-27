import time
from collections import defaultdict

import pandas as pd
from pybeamline.bevent import BEvent
from pybeamline.sources import xes_log_source_from_file
from pybeamline.stream.base_map import BaseMap
from pybeamline.stream.base_sink import BaseSink
from river import metrics as river_metrics

from utils.streaming.transformers import BaseStreamingTransformer


class NextActivityEmitterMap(BaseMap):
    """
    Buffers events per trace and emits (features, next_activity, metadata) tuples.
    """

    def __init__(self, transformer: BaseStreamingTransformer):
        self._transformer = transformer
        self._buffers: dict[str, list[BEvent]] = defaultdict(list)
        self._trace_n: int = 0
        self._trace_index: dict[str, int] = {}

    def transform(self, event: BEvent) -> list[tuple[dict, str, dict]] | None:
        trace_id = event.get_trace_name()

        if trace_id not in self._trace_index:
            self._trace_n += 1
            self._trace_index[trace_id] = self._trace_n

        buf = self._buffers[trace_id]

        if len(buf) == 0:
            buf.append(event)
            return None

        features = self._transformer.transform(buf)
        label = event.get_event_name()
        metadata = {
            'trace_id': trace_id,
            'trace_n': self._trace_index[trace_id],
            'prefix_len': len(buf),
            'event_time': str(event.get_event_time()),
        }
        buf.append(event)
        return [(features, label, metadata)]


class PrequentialClassifierMap(BaseMap):
    """
    Prequential (test-then-train) evaluation map.

    Receives (features, y_true, metadata), predicts, updates running metrics,
    then trains on the sample.
    """

    def __init__(self, model, model_name: str = ''):
        self._model = model
        self._name = model_name
        self._acc = river_metrics.Accuracy()
        self._f1 = river_metrics.MacroF1()
        self._n_pred = 0

    def transform(self, item: tuple[dict, str, dict]) -> list[dict] | None:
        features, y_true, metadata = item

        y_pred = self._model.predict_one(features)
        correct = None
        if y_pred is not None:
            self._acc.update(y_true, y_pred)
            self._f1.update(y_true, y_pred)
            self._n_pred += 1
            correct = y_pred == y_true

        self._model.learn_one(features, y_true)

        return [
            {
                'n_pred': self._n_pred,
                'y_true': y_true,
                'y_pred': y_pred,
                'correct': correct,
                'accuracy': self._acc.get(),
                'macro_f1': self._f1.get(),
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


class PrequentialPipeline:
    """End-to-end prequential streaming pipeline."""

    def __init__(
        self,
        model,
        transformer: BaseStreamingTransformer,
        model_name: str = '',
    ):
        self.model = model
        self.transformer = transformer
        self.model_name = model_name

    def run(self, dataset_path: str) -> pd.DataFrame:
        sink = CollectorSink()

        start_time = time.perf_counter()
        xes_log_source_from_file(dataset_path).pipe(
            NextActivityEmitterMap(self.transformer),
            PrequentialClassifierMap(self.model, self.model_name),
        ).sink(sink)
        elapsed = time.perf_counter() - start_time

        df = sink.to_dataframe()
        df['time_s'] = elapsed
        return df
