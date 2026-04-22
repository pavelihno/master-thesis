from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from utils.streaming.base.sample_buffers import PrefixTreeNode


class FeatureBuffer:
    def __init__(self, max_size: int | None = None) -> None:
        self._max_size = max_size
        self._feature_map: dict[tuple[Any, PrefixTreeNode], list[float]] = {}
        self._case_index: dict[Any, set[PrefixTreeNode]] = defaultdict(set)
        self._insertion_order: deque[tuple[Any, PrefixTreeNode]] = deque()

    def add_features(
        self,
        prefix_node: PrefixTreeNode,
        case_id: Any,
        features: list[float],
    ) -> None:
        """Add one training sample."""
        key = (case_id, prefix_node)
        feature_values = [float(v) for v in features]

        if key not in self._feature_map:
            self._insertion_order.append(key)
            self._case_index[case_id].add(prefix_node)

        self._feature_map[key] = feature_values

        if self._max_size is not None and self._max_size > 0:
            while len(self._feature_map) > self._max_size:
                oldest = self._insertion_order.popleft()
                if oldest not in self._feature_map:
                    continue

                old_case_id, old_node = oldest
                del self._feature_map[oldest]

                case_nodes = self._case_index.get(old_case_id)
                if case_nodes is not None:
                    case_nodes.discard(old_node)
                    if not case_nodes:
                        self._case_index.pop(old_case_id, None)

    def get_features(
        self, prefix_node: PrefixTreeNode, case_id: Any
    ) -> list[float] | None:
        """Return features for the given case and prefix node."""
        values = self._feature_map.get((case_id, prefix_node))
        if values is None:
            return None
        return list(values)

    @property
    def size(self) -> int:
        """Return total number of buffered features."""
        return len(self._feature_map)

    def clear(self, case_id: Any) -> None:
        """Clear all buffered features for the given case."""
        nodes = self._case_index.pop(case_id, set())
        if not nodes:
            return

        for node in nodes:
            self._feature_map.pop((case_id, node), None)

        self._insertion_order = deque(
            key for key in self._insertion_order if key[0] != case_id
        )
