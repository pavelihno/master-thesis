from typing import Any

from pybeamline.stream.base_map import BaseMap

from utils.streaming.base.maps import catch_and_reraise


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
            y_pred = self._model.predict_one(features)
        else:
            y_pred = None

        loss = self._get_loss()

        metadata = {**metadata, 'loss': loss}

        return [(trace_id, features, y_true, y_pred, metadata)]
