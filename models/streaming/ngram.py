from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from river import base


@dataclass(slots=True)
class HistoryNode:
    event_name: str | None
    parent: HistoryNode | None
    children: dict[str, HistoryNode] = field(default_factory=dict)

    def __repr__(self):
        if self.event_name is None:
            return '[ROOT]'
        return f'{self.parent} -> {self.event_name}'


@dataclass(slots=True)
class PrefixStatisticsNode:
    event_name: str | None
    parent: PrefixStatisticsNode | None
    children: dict[str, PrefixStatisticsNode] = field(default_factory=dict)
    target_counts: Counter[Any] = field(default_factory=Counter)
    target_sum: float = 0.0
    target_count: int = 0
    last_target: Any | None = None

    def update_classification(self, target: Any) -> None:
        self.last_target = target
        self.target_count += 1
        self.target_counts[target] += 1

    def update_regression(self, target: Any) -> None:
        value = float(target)
        self.last_target = value
        self.target_count += 1
        self.target_sum += value

    @property
    def majority_target(self) -> Any | None:
        if not self.target_counts:
            return None

        most_common = self.target_counts.most_common()
        best_count = most_common[0][1]
        tied_targets = [target for target, count in most_common if count == best_count]

        if self.last_target in tied_targets:
            return self.last_target

        return tied_targets[0]

    @property
    def average_target(self) -> float | None:
        if self.target_count == 0:
            return None
        return self.target_sum / self.target_count


class NGramBase:
    def __init__(self, sequence_window: int = 3) -> None:
        self.sequence_window = sequence_window

        self.prediction_history_root = HistoryNode(event_name=None, parent=None)
        self.prediction_history_table: dict[str, HistoryNode] = {}

        self.learning_history_root = HistoryNode(event_name=None, parent=None)
        self.learning_history_table: dict[str, HistoryNode] = {}

        self.prediction_seen_events: dict[str, set[int]] = defaultdict(set)
        self.learning_seen_events: dict[str, set[int]] = defaultdict(set)

        self.prefix_statistics_root = PrefixStatisticsNode(
            event_name=None,
            parent=None,
        )

    def clear_case(self, case_id: str) -> None:
        self.prediction_history_table.pop(case_id, None)
        self.learning_history_table.pop(case_id, None)
        self.prediction_seen_events.pop(case_id, None)
        self.learning_seen_events.pop(case_id, None)

    def _build_history_node(
        self,
        root: HistoryNode,
        table: dict[str, HistoryNode],
        seen_events: dict[str, set[int]],
        x: dict,
    ) -> HistoryNode:
        case_id, event_name, event_id = x['case_id'], x['event_name'], x['event_id']

        if event_id in seen_events[case_id]:
            return table.get(case_id, root)

        current_node = table.get(case_id, root)
        next_node = current_node.children.get(event_name)

        if next_node is None:
            next_node = HistoryNode(event_name=event_name, parent=current_node)
            current_node.children[event_name] = next_node

        table[case_id] = next_node
        seen_events[case_id].add(event_id)
        return next_node

    def _get_history_prefix(self, node: HistoryNode) -> list[str]:
        path: list[str] = []
        current = node
        while current.parent is not None:
            if current.event_name is not None:
                path.append(current.event_name)
            current = current.parent
        path.reverse()
        return path[-self.sequence_window :]

    def _get_prefix_node(self, prefix: list[str]) -> PrefixStatisticsNode:
        current = self.prefix_statistics_root
        for event_name in prefix:
            child = current.children.get(event_name)
            if child is None:
                child = PrefixStatisticsNode(event_name=event_name, parent=current)
                current.children[event_name] = child
            current = child
        return current

    def _find_prefix_node(self, prefix: list[str]) -> PrefixStatisticsNode:
        current = self.prefix_statistics_root
        best_node = current if current.target_count > 0 else current

        for event_name in prefix:
            child = current.children.get(event_name)
            if child is None:
                break
            current = child
            if current.target_count > 0:
                best_node = current

        return best_node


class NGramClassifier(base.Classifier, NGramBase):
    def __init__(self, sequence_window: int = 3) -> None:
        NGramBase.__init__(self, sequence_window=sequence_window)

    def learn_one(self, x: dict, y: Any):
        if y is None:
            return self

        self.prefix_statistics_root.update_classification(y)

        history_node = self._build_history_node(
            self.learning_history_root,
            self.learning_history_table,
            self.learning_seen_events,
            x,
        )
        prefix = self._get_history_prefix(history_node)
        self._get_prefix_node(prefix).update_classification(y)
        return self

    def predict_one(self, x: dict) -> Any | None:
        history_node = self._build_history_node(
            self.prediction_history_root,
            self.prediction_history_table,
            self.prediction_seen_events,
            x,
        )
        prefix = self._get_history_prefix(history_node)
        node = self._find_prefix_node(prefix)
        return node.majority_target

    def predict_proba_one(self, x: dict) -> dict[Any, float]:
        history_node = self._build_history_node(
            self.prediction_history_root,
            self.prediction_history_table,
            self.prediction_seen_events,
            x,
        )
        prefix = self._get_history_prefix(history_node)
        node = self._find_prefix_node(prefix)

        if node.target_count == 0:
            return {}

        total = sum(node.target_counts.values())
        return {target: count / total for target, count in node.target_counts.items()}


class NGramRegressor(base.Regressor, NGramBase):
    def __init__(self, sequence_window: int = 3) -> None:
        NGramBase.__init__(self, sequence_window=sequence_window)

    def learn_one(self, x: dict, y: Any):
        if y is None:
            return self

        self.prefix_statistics_root.update_regression(y)

        history_node = self._build_history_node(
            self.learning_history_root,
            self.learning_history_table,
            self.learning_seen_events,
            x,
        )
        prefix = self._get_history_prefix(history_node)
        self._get_prefix_node(prefix).update_regression(y)
        return self

    def predict_one(self, x: dict) -> float | None:
        history_node = self._build_history_node(
            self.prediction_history_root,
            self.prediction_history_table,
            self.prediction_seen_events,
            x,
        )
        prefix = self._get_history_prefix(history_node)
        node = self._find_prefix_node(prefix)
        return node.average_target
