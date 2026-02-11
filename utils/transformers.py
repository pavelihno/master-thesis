import pandas as pd
from sklearn.base import TransformerMixin


class AggregateTransformer(TransformerMixin):
    """Aggregates features per case using statistics."""

    def __init__(self, case_id_col, cat_cols, num_cols, boolean=False, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.boolean = boolean
        self.fillna = fillna
        self.columns = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Aggregate numerical features
        if len(self.num_cols) > 0:
            numeric_agg = X.groupby(self.case_id_col)[self.num_cols].agg(
                ['mean', 'max', 'min', 'sum', 'std']
            )
            numeric_agg.columns = [
                '_'.join(col).strip() for col in numeric_agg.columns.values
            ]

        # One-hot encode and aggregate categorical features
        categorical_encoded = pd.get_dummies(X[self.cat_cols])
        categorical_encoded[self.case_id_col] = X[self.case_id_col]

        if self.boolean:
            # Max aggregation: presence/absence of category
            categorical_agg = categorical_encoded.groupby(self.case_id_col).max()
        else:
            # Sum aggregation: count occurrences
            categorical_agg = categorical_encoded.groupby(self.case_id_col).sum()

        # Combine numerical and categorical aggregations
        if len(self.num_cols) > 0:
            result = pd.concat([categorical_agg, numeric_agg], axis=1)
        else:
            result = categorical_agg

        # Fill missing values
        if self.fillna:
            result = result.fillna(0)

        # Ensure consistent columns across fit/transform calls
        if self.columns is None:
            self.columns = result.columns
        else:
            # Add missing columns with zeros
            missing_cols = [col for col in self.columns if col not in result.columns]
            for col in missing_cols:
                result[col] = 0
            result = result[self.columns]

        return result


class IndexBasedExtractor(TransformerMixin):
    """Extracts only index-based encoded features up to max_events."""

    def __init__(self, cat_cols, num_cols, max_events, fillna=True):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_events = max_events
        self.fillna = fillna
        self.columns = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Determine relevant columns on first call
        if self.columns is None:
            # Numerical columns: col_0, col_1, ..., col_{max_events-1}
            num_col_names = [
                f'{col}_{i}'
                for col in self.num_cols
                for i in range(self.max_events)
            ]

            # Categorical columns: col_0_value, col_1_value, etc.
            cat_col_prefixes = tuple([
                f'{col}_{i}_'
                for col in self.cat_cols
                for i in range(self.max_events)
            ])
            cat_col_names = [
                col for col in X.columns
                if col.startswith(cat_col_prefixes)
            ]

            self.columns = cat_col_names + num_col_names
        else:
            # Add missing columns with zeros
            missing_cols = [col for col in self.columns if col not in X.columns]
            for col in missing_cols:
                X[col] = 0

        return X[self.columns]


class IndexBasedTransformer(TransformerMixin):
    """Encodes events by position (index) within each case."""

    def __init__(
        self,
        case_id_col,
        cat_cols,
        num_cols,
        max_events=None,
        fillna=True,
        create_dummies=True,
    ):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_events = max_events
        self.fillna = fillna
        self.create_dummies = create_dummies
        self.columns = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        grouped = X.groupby(self.case_id_col, as_index=False)

        # Determine max_events if not specified
        if self.max_events is None:
            self.max_events = grouped.size()['size'].max()

        # Start with case IDs
        result = pd.DataFrame(
            grouped.apply(lambda x: x.name), columns=[self.case_id_col]
        )

        # Extract features for each event position
        for i in range(self.max_events):
            event_at_position = grouped.nth(i)[
                [self.case_id_col] + self.cat_cols + self.num_cols
            ]
            event_at_position.columns = (
                [self.case_id_col]
                + [f'{col}_{i}' for col in self.cat_cols]
                + [f'{col}_{i}' for col in self.num_cols]
            )
            result = pd.merge(
                result, event_at_position, on=self.case_id_col, how='left'
            )

        result.index = result[self.case_id_col]

        # One-hot encode categorical columns
        if self.create_dummies:
            cat_col_names = [
                f'{col}_{i}' for col in self.cat_cols for i in range(self.max_events)
            ]
            result = pd.get_dummies(result, columns=cat_col_names).drop(
                self.case_id_col, axis=1
            )

        # Fill missing values
        if self.fillna:
            result = result.fillna(0)

        # Ensure consistent columns across fit/transform calls
        if self.columns is None:
            self.columns = result.columns
        else:
            # Add missing columns with zeros
            missing_cols = [col for col in self.columns if col not in result.columns]
            for col in missing_cols:
                result[col] = 0
            result = result[self.columns]

        return result


class LastStateTransformer(TransformerMixin):
    """Extracts features from the last event of each case."""

    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        self.columns = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        last_events = X.groupby(self.case_id_col).last()

        # Extract numerical features
        result = last_events[self.num_cols]

        # One-hot encode categorical features
        if len(self.cat_cols) > 0:
            categorical_encoded = pd.get_dummies(last_events[self.cat_cols])
            result = pd.concat([result, categorical_encoded], axis=1)

        # Fill missing values
        if self.fillna:
            result = result.fillna(0)

        # Ensure consistent columns across fit/transform calls
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in result.columns]
            for col in missing_cols:
                result[col] = 0
            result = result[self.columns]
        else:
            self.columns = result.columns

        return result


class PreviousStateTransformer(TransformerMixin):
    """Extracts features from the second-to-last event of each case."""

    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        self.columns = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Get second-to-last event
        penultimate_events = X.groupby(self.case_id_col).nth(-2)

        # Extract numerical features
        result = penultimate_events[self.num_cols]

        # One-hot encode categorical features
        if len(self.cat_cols) > 0:
            categorical_encoded = pd.get_dummies(penultimate_events[self.cat_cols])
            result = pd.concat([result, categorical_encoded], axis=1)

        # Reindex to include all cases (fill with 0 for cases with <2 events)
        all_case_ids = X.groupby(self.case_id_col).first().index
        result = result.reindex(all_case_ids, fill_value=0)

        # Fill missing values
        if self.fillna:
            result = result.fillna(0)

        # Ensure consistent columns across fit/transform calls
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in result.columns]
            for col in missing_cols:
                result[col] = 0
            result = result[self.columns]
        else:
            self.columns = result.columns

        return result


class StaticTransformer(TransformerMixin):
    """Extracts features from the first event of each case."""

    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        self.columns = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        first_events = X.groupby(self.case_id_col).first()

        # Extract numerical features
        result = first_events[self.num_cols]

        # One-hot encode categorical features
        if len(self.cat_cols) > 0:
            categorical_encoded = pd.get_dummies(first_events[self.cat_cols])
            result = pd.concat([result, categorical_encoded], axis=1)

        # Fill missing values
        if self.fillna:
            result = result.fillna(0)

        # Ensure consistent columns across fit/transform calls
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in result.columns]
            for col in missing_cols:
                result[col] = 0
            result = result[self.columns]
        else:
            self.columns = result.columns

        return result
