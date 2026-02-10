import numpy as np
import pandas as pd


class ClusterBasedBucketer:
    def __init__(self, encoder, clustering):
        self.encoder = encoder
        self.clustering = clustering

    def fit(self, X, y=None):

        dt_encoded = self.encoder.fit_transform(X)

        self.clustering.fit(dt_encoded)

        return self

    def predict(self, X, y=None):

        dt_encoded = self.encoder.transform(X)

        return self.clustering.predict(dt_encoded)

    def fit_predict(self, X, y=None):

        self.fit(X)
        return self.predict(X)


class NoBucketer:
    def __init__(self, case_id_col):
        self.n_states = 1
        self.case_id_col = case_id_col

    def fit(self, X, y=None):

        return self

    def predict(self, X, y=None):

        return np.ones(len(X[self.case_id_col].unique()), dtype=np.int)

    def fit_predict(self, X, y=None):

        self.fit(X)
        return self.predict(X)


class PrefixLengthBucketer:
    def __init__(self, case_id_col):
        self.n_states = 0
        self.case_id_col = case_id_col

    def fit(self, X, y=None):

        sizes = X.groupby(self.case_id_col).size()
        self.n_states = sizes.unique()

        return self

    def predict(self, X, y=None):

        return X.groupby(self.case_id_col).size().as_matrix()

    def fit_predict(self, X, y=None):

        self.fit(X)
        return self.predict(X)


class StateBasedBucketer:
    def __init__(self, encoder):
        self.encoder = encoder

        self.dt_states = None
        self.n_states = 0

    def fit(self, X, y=None):

        dt_encoded = self.encoder.fit_transform(X)

        self.dt_states = dt_encoded.drop_duplicates()
        self.dt_states = self.dt_states.assign(state=range(len(self.dt_states)))

        self.n_states = len(self.dt_states)

        return self

    def predict(self, X, y=None):

        dt_encoded = self.encoder.transform(X)

        dt_transformed = pd.merge(dt_encoded, self.dt_states, how='left')
        dt_transformed.fillna(-1, inplace=True)

        return dt_transformed['state'].astype(int).as_matrix()

    def fit_predict(self, X, y=None):

        self.fit(X)
        return self.predict(X)
