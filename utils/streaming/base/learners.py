from abc import abstractmethod
from collections import defaultdict
from typing import Any

from pybeamline.stream.base_map import BaseMap

from utils.streaming.base.maps import catch_and_reraise
from utils.streaming.time import TimeTarget, convert_time, parse_time


class LearnerMap(BaseMap):
    def __init__(self, model):
        self._model = model
        self._has_drift_detector = (
            hasattr(model, 'drift_detector') and model.drift_detector is not None
        )

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


class NextActivityLearner(LearnerMap):
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


class OutcomeLearner(LearnerMap):
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


class RemainingTimeLearner(LearnerMap):
    def __init__(self, model, target: TimeTarget = TimeTarget.SECONDS):
        super().__init__(model)

        self._target = target

        # Features and event times before target is available
        self._pending_features: dict[str, list[tuple[int, dict, Any]]] = defaultdict(
            list
        )

    @catch_and_reraise
    def transform(
        self, item: tuple[str, dict, Any, Any, dict]
    ) -> list[tuple[str, dict, Any, Any, dict]] | None:
        trace_id, features, y_true, y_pred, metadata = item

        prefix_len = metadata.get('prefix_len', 0)
        is_end = metadata.get('is_end', False)
        event_time = parse_time(metadata.get('event_time'))

        drift_detected = False

        # Store prefix features and event timestamps
        if features and event_time is not None:
            self._pending_features[trace_id].append((prefix_len, features, event_time))

        # Learn once the trace has ended
        if is_end and event_time is not None:
            end_time = event_time
            trace_features = self._pending_features.pop(trace_id, [])

            for pending_prefix_len, pending_features, prefix_time in trace_features:
                remaining_time = convert_time(
                    end_time - prefix_time,
                    target=self._target,
                )

                self._model.learn_one(pending_features, remaining_time)
                drift_detected = drift_detected or self._is_drift_detected()

        elif is_end:
            print(f'Learner warning: Trace {trace_id} ended without valid event time')
            self._pending_features.pop(trace_id, None)

        loss = self._get_loss()
        metadata = {**metadata, 'drift_detected': drift_detected, 'loss': loss}

        return [(trace_id, features, y_true, y_pred, metadata)]
