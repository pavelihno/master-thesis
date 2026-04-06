from abc import ABC, abstractmethod
from typing import Any

from river import base


class Bucketer(ABC):
    def __init__(self, model: base.Estimator, max_buckets: int | None = None):
        self.model = model
        self.max_buckets = max_buckets
        self.buckets: dict[Any, base.Estimator] = {}

    @abstractmethod
    def get_bucket(self, x: dict) -> Any:
        pass

    def _create_model(self) -> base.Base:
        return self.model.clone()

    def _get_model(self, x: dict) -> base.Base:
        bucket = self.get_bucket(x)

        if bucket not in self.buckets:
            if self.max_buckets is not None and len(self.buckets) >= self.max_buckets:
                raise ValueError(
                    'max_buckets='
                    f'{self.max_buckets} reached; cannot create bucket {bucket!r}'
                )
            self.buckets[bucket] = self._create_model()

        return self.buckets[bucket]

    def learn_one(self, x: dict, y: Any):
        self._get_model(x).learn_one(x, y)
        return self

    def predict_one(self, x: dict) -> Any:
        return self._get_model(x).predict_one(x)

    def predict_proba_one(self, x: dict) -> dict[Any, float]:
        model = self._get_model(x)

        if hasattr(model, 'predict_proba_one'):
            return model.predict_proba_one(x)

        prediction = model.predict_one(x)

        return {} if prediction is None else {prediction: 1.0}


class NoBucketer(Bucketer):
    def __init__(self, model: base.Estimator):
        super().__init__(model=model, max_buckets=1)

    def get_bucket(self, x: dict) -> Any:
        return 0


class PrefixLengthBucketer(Bucketer):
    def __init__(
        self,
        model: base.Estimator,
        max_buckets: int | None = None,
        prefix_len_key: str = 'prefix_len',
    ):
        super().__init__(model=model, max_buckets=max_buckets)
        self.prefix_len_key = prefix_len_key

    def get_bucket(self, x: dict) -> Any:
        if self.prefix_len_key not in x:
            raise KeyError(
                f"Missing '{self.prefix_len_key}' in features; "
                'cannot determine prefix-length bucket'
            )

        return int(x[self.prefix_len_key])
