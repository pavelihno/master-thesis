import numpy as np
import pandas as pd


class NoBucketer:
    """Assigns all cases to a single bucket (no bucketing)."""

    def __init__(self, case_id_col):
        self.n_states = 1
        self.case_id_col = case_id_col

    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        n_cases = X[self.case_id_col].nunique()
        return np.ones(n_cases, dtype=int)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


class PrefixLengthBucketer:
    """Assigns cases to buckets based on their prefix length."""

    def __init__(self, case_id_col):
        self.n_states = 0
        self.case_id_col = case_id_col

    def fit(self, X, y=None):
        prefix_lengths = X.groupby(self.case_id_col).size()
        self.n_states = prefix_lengths.nunique()
        return self

    def predict(self, X, y=None):
        prefix_lengths = X.groupby(self.case_id_col).size().values
        return prefix_lengths

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


class ClusterBasedBucketer:
    """Assigns cases to buckets using clustering on encoded features."""

    def __init__(self, encoder, clustering):
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

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


class StateBasedBucketer:
    """Assigns cases to buckets based on unique encoded states."""

    def __init__(self, encoder):
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

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)
