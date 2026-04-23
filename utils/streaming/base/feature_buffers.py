from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any

from utils.streaming.base.sample_buffers import PrefixTreeNode


class FeatureBuffer:
    def __init__(self, sequence_window: int) -> None:

        self._sequence_window = sequence_window
        self._features: defaultdict[Any, OrderedDict[PrefixTreeNode, list[float]]] = (
            defaultdict(OrderedDict)
        )

    def add_features(
        self,
        prefix_node: PrefixTreeNode,
        case_id: Any,
        features: list[float],
    ) -> None:
        """Store features for one case and prefix node."""
        case_features = self._features[case_id]
        feature_values = [float(v) for v in features]

        if prefix_node not in case_features:
            if len(case_features) == self._sequence_window:
                case_features.popitem(last=False)

        case_features[prefix_node] = feature_values

    def get_features(
        self, prefix_node: PrefixTreeNode, case_id: Any
    ) -> list[float] | None:
        """Return features for the given case and prefix node."""
        case_features = self._features.get(case_id)
        if case_features is None:
            return None

        values = case_features.get(prefix_node)
        if values is None:
            return None
        return list(values)

    @property
    def size(self) -> int:
        """Return total number of buffered features."""
        return sum(len(case_features) for case_features in self._features.values())

    def clear(self, case_id: Any) -> None:
        """Clear buffered features for the given case."""
        self._features.pop(case_id, None)
