from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrefixTreeNode:
    """A node in the Prefix Tree (T) representing a process event."""

    event_name: str | None
    parent: PrefixTreeNode | None
    children: dict[str, PrefixTreeNode] = field(default_factory=dict)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        if self.event_name is None:
            return '[ROOT]'
        return f'{self.parent} -> {self.event_name}'


@dataclass(frozen=True, slots=True)
class SampleBufferElement:
    prefix_node: PrefixTreeNode
    case_id: Any
    target: Any


class SampleBuffer:
    def __init__(self, max_size: int | None = None) -> None:
        self._samples: list[SampleBufferElement] = []
        self._max_size = max_size

    def add_sample(
        self, prefix_node: PrefixTreeNode, case_id: Any, target: Any
    ) -> None:
        """Add one training sample."""
        self._samples.append(
            SampleBufferElement(prefix_node=prefix_node, case_id=case_id, target=target)
        )

        if self._max_size is not None and self._max_size > 0:
            if len(self._samples) > self._max_size:
                self._samples = self._samples[-self._max_size :]

    def get_samples(self) -> list[SampleBufferElement]:
        """Return all stored samples."""
        return list(self._samples)

    @property
    def size(self) -> int:
        """Return total number of buffered samples."""
        return len(self._samples)

    def clear(self) -> None:
        """Clear all buffered samples."""
        self._samples.clear()


class ReservoirSampleBuffer(SampleBuffer):
    """Stores samples using reservoir sampling.

    Maintains a representative subset of the data.
    """

    def __init__(self, max_size: int | None = None) -> None:
        super().__init__(max_size=max_size)
        self._seen = 0

    def add_sample(
        self, prefix_node: PrefixTreeNode, case_id: Any, target: Any
    ) -> None:
        element = SampleBufferElement(
            prefix_node=prefix_node,
            case_id=case_id,
            target=target,
        )

        if self._max_size is None:
            self._samples.append(element)
            return

        if self._max_size <= 0:
            return

        self._seen += 1

        if len(self._samples) < self._max_size:
            self._samples.append(element)
            return

        idx = random.randint(0, self._seen - 1)
        if idx < self._max_size:
            self._samples[idx] = element

    def clear(self) -> None:
        super().clear()
        self._seen = 0
