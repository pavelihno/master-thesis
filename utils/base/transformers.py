from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from utils.constants import CASE_PREFIX_COL, TIME_COL


class BaseTransformer(BaseEstimator, TransformerMixin, ABC):
    """Base class for all feature transformers."""

    def __init__(self):
        self.columns = None

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the transformer on the data."""
        pass

    @abstractmethod
    def transform(self, X, y=None):
        """Transform the data into features."""
        pass

    def _ensure_consistent_columns(self, result):
        """Ensure consistent columns across fit/transform calls."""
        if self.columns is None:
            self.columns = result.columns
        else:
            # Add missing columns with zeros
            missing_cols = [col for col in self.columns if col not in result.columns]
            for col in missing_cols:
                result[col] = 0
            result = result[self.columns]
        return result


class AggregateTransformer(BaseTransformer):
    """Aggregates features per case using statistics."""

    def __init__(
        self,
        group_col=CASE_PREFIX_COL,
        cat_cols=None,
        num_cols=None,
        boolean=False,
        fillna=True,
    ):
        super().__init__()
        self.group_col = group_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.boolean = boolean
        self.fillna = fillna

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cat_cols = self.cat_cols or []
        num_cols = self.num_cols or []

        # Aggregate numerical features
        if len(num_cols) > 0:
            numeric_agg = X.groupby(self.group_col)[num_cols].agg(
                ['mean', 'max', 'min', 'sum', 'std']
            )
            numeric_agg.columns = [
                '_'.join(col).strip() for col in numeric_agg.columns.values
            ]

        # One-hot encode and aggregate categorical features
        categorical_encoded = pd.get_dummies(X[cat_cols])
        categorical_encoded[self.group_col] = X[self.group_col]

        if self.boolean:
            # Max aggregation: presence/absence of category
            categorical_agg = categorical_encoded.groupby(self.group_col).max()
        else:
            # Sum aggregation: count occurrences
            categorical_agg = categorical_encoded.groupby(self.group_col).sum()

        # Combine numerical and categorical aggregations
        if len(num_cols) > 0:
            result = pd.concat([categorical_agg, numeric_agg], axis=1)
        else:
            result = categorical_agg

        # Fill missing values
        if self.fillna:
            result = result.fillna(0)

        # Ensure consistent columns across fit/transform calls
        return self._ensure_consistent_columns(result)


class IndexBasedTransformer(BaseTransformer):
    """Encodes events by position (index) within each case."""

    def __init__(
        self,
        group_col=CASE_PREFIX_COL,
        cat_cols=None,
        num_cols=None,
        max_events=None,
        fillna=True,
        create_dummies=True,
    ):
        super().__init__()
        self.group_col = group_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_events = max_events
        self.fillna = fillna
        self.create_dummies = create_dummies

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cat_cols = self.cat_cols or []
        num_cols = self.num_cols or []
        grouped = X.groupby(self.group_col, as_index=False)

        # Determine max_events if not specified
        if self.max_events is None:
            self.max_events = grouped.size()['size'].max()

        # Start with case IDs
        result = pd.DataFrame(grouped.apply(lambda x: x.name), columns=[self.group_col])

        # Extract features for each event position
        for i in range(self.max_events):
            event_at_position = grouped.nth(i)[[self.group_col] + cat_cols + num_cols]
            event_at_position.columns = (
                [self.group_col]
                + [f'{col}_{i}' for col in cat_cols]
                + [f'{col}_{i}' for col in num_cols]
            )
            result = pd.merge(result, event_at_position, on=self.group_col, how='left')

        result.index = result[self.group_col]

        # One-hot encode categorical columns
        if self.create_dummies:
            cat_col_names = [
                f'{col}_{i}' for col in cat_cols for i in range(self.max_events)
            ]
            result = pd.get_dummies(result, columns=cat_col_names).drop(
                self.group_col, axis=1
            )

        # Fill missing values
        if self.fillna:
            result = result.fillna(0)

        # Ensure consistent columns across fit/transform calls
        return self._ensure_consistent_columns(result)


class LastStateTransformer(BaseTransformer):
    """Extracts features from the last event of each case."""

    def __init__(
        self,
        group_col=CASE_PREFIX_COL,
        cat_cols=None,
        num_cols=None,
        fillna=True,
    ):
        super().__init__()
        self.group_col = group_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cat_cols = self.cat_cols or []
        num_cols = self.num_cols or []

        last_events = X.groupby(self.group_col).last()

        # Extract numerical features
        result = last_events[num_cols]

        # One-hot encode categorical features
        if len(cat_cols) > 0:
            categorical_encoded = pd.get_dummies(last_events[cat_cols])
            result = pd.concat([result, categorical_encoded], axis=1)

        # Fill missing values
        if self.fillna:
            result = result.fillna(0)

        # Ensure consistent columns across fit/transform calls
        return self._ensure_consistent_columns(result)


class PreviousStateTransformer(BaseTransformer):
    """Extracts features from the second-to-last event of each case."""

    def __init__(
        self,
        group_col=CASE_PREFIX_COL,
        cat_cols=None,
        num_cols=None,
        fillna=True,
    ):
        super().__init__()
        self.group_col = group_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cat_cols = self.cat_cols or []
        num_cols = self.num_cols or []
        # Get second-to-last event
        penultimate_events = X.groupby(self.group_col).nth(-2)

        # Extract numerical features
        result = penultimate_events[num_cols]

        # One-hot encode categorical features
        if len(cat_cols) > 0:
            categorical_encoded = pd.get_dummies(penultimate_events[cat_cols])
            result = pd.concat([result, categorical_encoded], axis=1)

        # Reindex to include all cases (fill with 0 for cases with <2 events)
        all_case_ids = X.groupby(self.group_col).first().index
        result = result.reindex(all_case_ids, fill_value=0)

        # Fill missing values
        if self.fillna:
            result = result.fillna(0)

        # Ensure consistent columns across fit/transform calls
        return self._ensure_consistent_columns(result)


class LSTMTransformer(BaseTransformer):
    """Prepares sequences of events for LSTM model."""

    def __init__(
        self,
        group_col=CASE_PREFIX_COL,
        time_col=TIME_COL,
        cat_cols=None,
        num_cols=None,
        max_events=None,
    ):
        self.group_col = group_col
        self.time_col = time_col
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.max_events = max_events

        self.vocab = {}
        self.vocab_sizes = {}

        self.numeric_mean = {}
        self.numeric_std = {}

    def fit(self, X, y=None):
        X = X.sort_values([self.group_col, self.time_col])

        if self.max_events is None:
            self.max_events = int(X.groupby(self.group_col).size().max())

        # Build Vocabularies
        for col in self.cat_cols:
            vocab = {'<PAD>': 0, '<UNK>': 1}
            for idx, val in enumerate(X[col].dropna().unique(), start=2):
                vocab[val] = idx
            self.vocab[col] = vocab
            self.vocab_sizes[col] = len(vocab)

        # Calculate standard scaling parameters
        for col in self.num_cols:
            self.numeric_mean[col] = float(X[col].mean())
            std = float(X[col].std())
            self.numeric_std[col] = 1.0 if pd.isna(std) or std == 0 else std

        return self

    def transform(self, X):
        if not self.vocab and self.cat_cols:
            raise ValueError('Transformer not fitted.')

        X = X.sort_values([self.group_col, self.time_col]).copy()

        # Scale numeric vectors
        for col in self.num_cols:
            mean = self.numeric_mean[col]
            std = self.numeric_std[col]
            X[col] = (X[col].fillna(mean) - mean) / std

        # Map categoricals to token IDs
        for col in self.cat_cols:
            unk_idx = self.vocab[col]['<UNK>']
            X[col] = X[col].map(self.vocab[col]).fillna(unk_idx).astype(np.int64)

        # Build sequence index constraints
        X['__event_idx'] = X.groupby(self.group_col).cumcount()
        X = X[X['__event_idx'] < self.max_events]

        # Extract uniquely identified case sequences
        case_ids = X[self.group_col].unique()
        n_cases = len(case_ids)

        case_to_row = {cid: idx for idx, cid in enumerate(case_ids)}
        row_indices = X[self.group_col].map(case_to_row).values
        col_indices = X['__event_idx'].values

        # Build categorical structure matrix
        cat_features = np.zeros(
            (n_cases, self.max_events, len(self.cat_cols)), dtype=np.int64
        )
        if self.cat_cols:
            cat_features[row_indices, col_indices] = X[self.cat_cols].values

        # Build numerical structure matrix
        num_features = np.zeros(
            (n_cases, self.max_events, len(self.num_cols)), dtype=np.float32
        )
        if self.num_cols:
            num_features[row_indices, col_indices] = X[self.num_cols].values

        return {
            'cat_features': cat_features,
            'num_features': num_features,
        }
