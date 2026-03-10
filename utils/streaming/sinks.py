from collections import defaultdict

import pandas as pd
from pybeamline.bevent import BEvent
from pybeamline.stream.base_sink import BaseSink
from river import metrics as river_metrics


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
