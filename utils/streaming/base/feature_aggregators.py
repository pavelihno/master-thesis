from abc import ABC, abstractmethod
from typing import TypeAlias


FeatureValue: TypeAlias = float | int | str | bool | None


class FeatureAggregator(ABC):
    """Base class for feature aggregators."""

    def __init__(self, output_size: int = 1) -> None:
        self.output_size = output_size

    @abstractmethod
    def update(self, value: FeatureValue) -> None:
        pass

    @abstractmethod
    def get_features(self) -> list[float]:
        pass

    @abstractmethod
    def reset(self, count: int) -> None:
        """Reset the aggregator state."""
        pass


class NumericalFeatureAggregator(FeatureAggregator, ABC):
    """Aggregator for numerical features."""

    @abstractmethod
    def update(self, value: float | int | None) -> None:
        pass


class AverageAggregator(NumericalFeatureAggregator):
    """Aggregator that computes the average of a numerical feature."""

    def __init__(self) -> None:
        super().__init__(output_size=1)

        self._sum: float = 0.0
        self._count: int = 0

    def update(self, value: float | int | None) -> None:
        if value is None:
            return

        self._sum += float(value)
        self._count += 1

    @property
    def _average(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    def get_features(self) -> list[float]:
        return [self._average]

    def reset(self, count: int) -> None:
        if count <= 0:
            return

        _old_average = self._average

        self._count = max(0, self._count - count)

        if self._count == 0:
            self._sum = 0.0
            return

        self._sum = _old_average * self._count


class CategoricalFeatureAggregator(FeatureAggregator, ABC):
    """Aggregator for categorical features."""

    @abstractmethod
    def update(self, value: FeatureValue) -> None:
        pass
