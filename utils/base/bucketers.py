from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BaseBucketer(BaseEstimator, ABC):
    """Base class for all bucketing strategies."""

    def __init__(self):
        self.n_states = 0

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the bucketer on the data."""
        pass

    @abstractmethod
    def predict(self, X, y=None):
        """Assign cases to buckets."""
        pass

    def fit_predict(self, X, y=None):
        """Fit and predict in one step."""
        self.fit(X, y)
        return self.predict(X, y)


class NoBucketer(BaseBucketer):
    """Assigns all cases to a single bucket (no bucketing)."""

    def __init__(self, case_id_col):
        super().__init__()
        self.case_id_col = case_id_col
        self.n_states = 1

    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        n_cases = X[self.case_id_col].nunique()
        return np.ones(n_cases, dtype=int)


class PrefixLengthBucketer(BaseBucketer):
    """Assigns cases to buckets based on their prefix length."""

    def __init__(self, case_id_col):
        super().__init__()
        self.case_id_col = case_id_col

    def fit(self, X, y=None):
        prefix_lengths = X.groupby(self.case_id_col).size()
        self.n_states = prefix_lengths.nunique()
        return self

    def predict(self, X, y=None):
        prefix_lengths = X.groupby(self.case_id_col).size().values
        return prefix_lengths


class ClusterBasedBucketer(BaseBucketer):
    """Assigns cases to buckets using clustering on encoded features."""

    def __init__(self, encoder, clustering):
        super().__init__()
        self.encoder = encoder
        self.clustering = clustering

    def fit(self, X, y=None):
        encoded_data = self.encoder.fit_transform(X)
        self.clustering.fit(encoded_data)
        return self

    def predict(self, X, y=None):
        encoded_data = self.encoder.transform(X)
        cluster_ids = self.clustering.predict(encoded_data)
        return cluster_ids


class StateBasedBucketer(BaseBucketer):
    """Assigns cases to buckets based on unique encoded states."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.state_mapping = None
        self.n_states = 0

    def fit(self, X, y=None):
        encoded_data = self.encoder.fit_transform(X)

        # Find unique states and assign IDs
        unique_states = encoded_data.drop_duplicates()
        self.state_mapping = unique_states.assign(state=range(len(unique_states)))
        self.n_states = len(self.state_mapping)

        return self

    def predict(self, X, y=None):
        encoded_data = self.encoder.transform(X)

        # Map encoded features to state IDs
        data_with_states = pd.merge(encoded_data, self.state_mapping, how='left')
        state_ids = data_with_states['state'].fillna(-1).astype(int).values

        return state_ids
