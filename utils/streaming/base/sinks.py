from abc import abstractmethod
from collections import defaultdict
from typing import Any

import pandas as pd
from pybeamline.bevent import BEvent
from pybeamline.stream.base_sink import BaseSink
from river import metrics as river_metrics


class EmptySink(BaseSink):
    def consume(self, item) -> None:
        pass

    def close(self) -> None:
        pass


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
        self._metrics: dict = self._make_metrics()

    @abstractmethod
    def _make_metrics(self) -> dict:
        pass

    @abstractmethod
    def _update_metrics(self, y_true, y_pred) -> dict:
        pass

    def _current_metric_values(self) -> dict:
        return {name: m.get() for name, m in self._metrics.items()}

    def _update_metrics(self, y_true, y_pred) -> dict:
        for m in self._metrics.values():
            m.update(y_true, y_pred)
        return self._current_metric_values()

    def _current_metric_values(self) -> dict:
        return {name: m.get() for name, m in self._metrics.items()}

    @abstractmethod
    def consume(self, item: tuple[str, dict, Any, Any, dict]) -> None:
        pass

    def _get_metadata(self, metadata: dict) -> dict[str, Any]:
        return {
            'drift_detected': metadata.pop('drift_detected', False),
            'trace_n': metadata.pop('trace_n', -1),
            'prefix_len': metadata.pop('prefix_len', -1),
            'loss': metadata.pop('loss', None),
            'is_end': metadata.pop('is_end', False),
        }

    @abstractmethod
    def close(self) -> None:
        pass


class ClassificationEvaluator(EvaluatorSink):
    def _make_metrics(self) -> dict:
        return {
            'accuracy': river_metrics.Accuracy(),
            'macro_f1': river_metrics.MacroF1(),
        }


class RegressionEvaluator(EvaluatorSink):
    def _make_metrics(self) -> dict:
        return {
            'mae': river_metrics.MAE(),
            'rmse': river_metrics.RMSE(),
        }


class NextActivityEvaluator(ClassificationEvaluator):
    def __init__(self):
        super().__init__()

        self._pending_predictions: dict[str, Any] = {}

    def consume(self, item: tuple[str, dict, Any, Any, dict]) -> None:
        trace_id, features, y_true, y_pred, metadata = item

        metadata = self._get_metadata(metadata)
        drift_detected = metadata['drift_detected']
        trace_n = metadata['trace_n']
        prefix_len = metadata['prefix_len']
        loss = metadata['loss']

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
                'loss': loss,
                **metric_vals,
            }
        )

    def close(self) -> None:
        self._pending_predictions.clear()


class OutcomeEvaluator(ClassificationEvaluator):
    def __init__(self):
        super().__init__()

        self._pending_predictions: dict[str, list[tuple[int, Any]]] = defaultdict(list)

    def _get_record(
        self, trace_id, y_true, y_pred, drift_detected, trace_n, prefix_len, loss
    ):
        return {
            'trace_id': trace_id,
            'y_true': y_true,
            'y_pred': y_pred,
            'n_pred': self._n_pred,
            'n_drifts': self._n_drifts,
            'drift_detected': drift_detected,
            'trace_n': trace_n,
            'prefix_len': prefix_len,
            'loss': loss,
            **self._current_metric_values(),
        }

    def consume(self, item: tuple[str, dict, Any, Any, dict]) -> None:
        trace_id, features, y_true, y_pred, metadata = item

        metadata = self._get_metadata(metadata)
        prefix_len = metadata['prefix_len']
        is_end = metadata['is_end']
        trace_n = metadata['trace_n']
        loss = metadata['loss']
        drift_detected = metadata['drift_detected']

        if y_true is not None:
            # Evaluate all pending predictions for the same trace
            for _prefix_len, _y_pred in self._pending_predictions[trace_id]:
                if _y_pred is not None:
                    self._update_metrics(y_true, _y_pred)

                self.records.append(
                    self._get_record(
                        trace_id=trace_id,
                        y_true=y_true,
                        y_pred=_y_pred,
                        drift_detected=drift_detected,
                        trace_n=trace_n,
                        prefix_len=_prefix_len,
                        loss=loss,
                    )
                )

            self._pending_predictions[trace_id].clear()

        else:
            # Store prediction for later evaluation
            self._pending_predictions[trace_id].append((prefix_len, y_pred))

        if is_end:
            for _prefix_len, _y_pred in self._pending_predictions[trace_id]:
                self.records.append(
                    self._get_record(
                        trace_id=trace_id,
                        y_true=y_true,
                        y_pred=_y_pred,
                        drift_detected=drift_detected,
                        trace_n=trace_n,
                        prefix_len=_prefix_len,
                        loss=loss,
                    )
                )

            self._pending_predictions.pop(trace_id, None)

    def close(self) -> None:
        self._pending_predictions.clear()


class TraceCollector(BaseSink):
    """Groups BEvents by trace_id into a dict."""

    def __init__(self):
        self.traces: dict[str, list[BEvent]] = defaultdict(list)

    def consume(self, event: BEvent) -> None:
        self.traces[event.get_trace_name()].append(event)

    def close(self) -> None:
        pass

    def to_dict(self) -> dict[str, list[BEvent]]:
        return dict(self.traces)
