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
    def __init__(self):
        super().__init__()

        self._n_pred: int = 0
        self._n_drifts: int = 0

        # Predictions from the previous event per trace
        self._pending_predictions: dict[str, any] = {}
        self._metrics: dict = self._make_metrics()

    def _make_metrics(self) -> dict:
        raise NotImplementedError

    def _update_metrics(self, y_true, y_pred) -> dict:
        raise NotImplementedError

    def _current_metric_values(self) -> dict:
        return {name: m.get() for name, m in self._metrics.items()}

    def consume(self, item: tuple[dict, any, any, dict]) -> None:
        trace_id, features, y_true, y_pred, metadata = item

        drift_detected = metadata.pop('drift_detected', False)
        trace_n = metadata.get('trace_n', -1)
        prefix_len = metadata.get('prefix_len', -1)

        if drift_detected:
            self._n_drifts += 1

        # Evaluate on prediction from the previous event of the same trace
        if trace_id in self._pending_predictions:
            self._n_pred += 1
            prev_y_pred = self._pending_predictions[trace_id]
            if prev_y_pred is not None:
                metric_vals = self._update_metrics(y_true, prev_y_pred)
            else:
                metric_vals = self._current_metric_values()
        else:
            prev_y_pred = None
            metric_vals = self._current_metric_values()

        # Store current prediction for the next event of the same trace
        # if features:
        if y_pred is not None:
            self._pending_predictions[trace_id] = y_pred
        else:
            self._pending_predictions.pop(trace_id, None)

        self.records.append(
            {
                'trace_id': trace_id,
                'y_true': y_true,
                'y_pred': prev_y_pred,
                'n_pred': self._n_pred,
                'n_drifts': self._n_drifts,
                'drift_detected': drift_detected,
                'trace_n': trace_n,
                'prefix_len': prefix_len,
                **metric_vals,
            }
        )

    def close(self) -> None:
        """Flush remaining pending predictions that were never evaluated."""
        self._pending_predictions.clear()


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
