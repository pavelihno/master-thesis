from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from utils.streaming.base.feature_aggregators import FeatureAggregator


@dataclass
class PrefixTreeNode:
    """A node in the Prefix Tree (T) representing a process event."""

    event_name: str | None
    parent: PrefixTreeNode | None
    children: dict[str, PrefixTreeNode] = field(default_factory=dict)
    feature_aggs: list[FeatureAggregator] = field(default_factory=list)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        if self.event_name is None:
            return '[ROOT]'
        return f'{self.parent} -> {self.event_name}'


@dataclass(frozen=True, slots=True)
class NodeTargetKey:
    """Aggregation key for classification windows."""

    prefix_node: PrefixTreeNode
    target: Any


@dataclass(frozen=True, slots=True)
class NodeBinKey:
    """Aggregation key for regression windows."""

    prefix_node: PrefixTreeNode
    bin_value: Any


@dataclass(slots=True)
class BinStats:
    """Statistics collected for a window bin."""

    count: int = 0
    target_sum: float = 0.0


class SampleBuffer(ABC):
    @abstractmethod
    def add_sample(self, prefix_node: PrefixTreeNode, target: Any) -> None:
        """Add one training sample."""

    @abstractmethod
    def get_samples(self) -> list[dict[str, Any]]:
        """Return flattened samples."""

    @abstractmethod
    def get_unique_nodes(self) -> set[PrefixTreeNode]:
        """Return unique prefix nodes referenced by the buffer."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Return total number of buffered samples."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all buffered data."""


class SimpleBuffer(SampleBuffer):
    """Store samples as-is without aggregation."""

    def __init__(self) -> None:
        self._samples: list[NodeTargetKey] = []

    def add_sample(self, prefix_node: PrefixTreeNode, target: Any) -> None:
        self._samples.append(NodeTargetKey(prefix_node=prefix_node, target=target))

    def get_samples(self) -> list[dict[str, Any]]:
        return [
            {'prefix_node': sample.prefix_node, 'target': sample.target}
            for sample in self._samples
        ]

    def get_unique_nodes(self) -> set[PrefixTreeNode]:
        return {sample.prefix_node for sample in self._samples}

    @property
    def size(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()


class CountBuffer(SampleBuffer):
    """Aggregate classification samples by (prefix_node, class_target)."""

    def __init__(self) -> None:
        self._counts: dict[NodeTargetKey, int] = defaultdict(int)
        self._size: int = 0

    def add_sample(self, prefix_node: PrefixTreeNode, target: Any) -> None:
        key = NodeTargetKey(prefix_node=prefix_node, target=target)
        self._counts[key] += 1
        self._size += 1

    def get_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for key, count in self._counts.items():
            for _ in range(count):
                samples.append({'prefix_node': key.prefix_node, 'target': key.target})
        return samples

    def get_unique_nodes(self) -> set[PrefixTreeNode]:
        return {key.prefix_node for key in self._counts}

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._counts.clear()
        self._size = 0


class BinBuffer(SampleBuffer):
    """Aggregate regression samples by (prefix_node, bin_value)."""

    def __init__(
        self,
        bin_width: float = 1.0,
    ) -> None:
        if bin_width <= 0:
            raise ValueError('bin_width must be > 0')

        self.bin_width = float(bin_width)
        self._stats: dict[NodeBinKey, BinStats] = {}
        self._size: int = 0

    def _to_bin(self, value: float) -> int:
        """Map numeric target value to an integer bin id."""
        return int(math.floor(value / self.bin_width))

    def add_sample(self, prefix_node: PrefixTreeNode, target: Any) -> None:
        y = float(target)
        key = NodeBinKey(
            prefix_node=prefix_node,
            bin_value=self._to_bin(y),
        )
        stats = self._stats.get(key)
        if stats is None:
            stats = BinStats()
            self._stats[key] = stats

        stats.count += 1
        stats.target_sum += y
        self._size += 1

    def get_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for key, stats in self._stats.items():
            if stats.count <= 0:
                continue

            mean_target = stats.target_sum / stats.count
            for _ in range(stats.count):
                samples.append(
                    {
                        'prefix_node': key.prefix_node,
                        'target': mean_target,
                    }
                )
        return samples

    def get_unique_nodes(self) -> set[PrefixTreeNode]:
        return {key.prefix_node for key in self._stats}

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._stats.clear()
        self._size = 0
