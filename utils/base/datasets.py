from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pm4py
from sklearn.preprocessing import LabelEncoder

from utils.experiment.temporal_splitting import (
    create_unbiased_temporal_split,
    print_split_summary,
)


class BaseLogDataset(ABC):
    def __init__(
        self,
        dataset_path,
        train_ratio=0.8,
        min_prefix=None,
        max_prefix=None,
        case_id_col='case:concept:name',
        time_col='time:timestamp',
        activity_col='concept:name',
        resource_col='org:resource',
    ):
        self.dataset_path = dataset_path

        self.train_ratio = train_ratio
        self.min_prefix = min_prefix if min_prefix is not None else 1
        self.max_prefix = max_prefix

        self.case_id_col = case_id_col
        self.time_col = time_col
        self.activity_col = activity_col
        self.resource_col = resource_col

        self.raw_df = None

        print(f'Loading {self.dataset_path}...')

        self.raw_df = pm4py.read_xes(self.dataset_path)
        self.raw_df[self.case_id_col] = self.raw_df[self.case_id_col].astype(str)
        self.raw_df[self.time_col] = pd.to_datetime(
            self.raw_df[self.time_col], utc=True
        )
        self.raw_df = self.raw_df.sort_values([self.case_id_col, self.time_col])

        return self

    def _extract_features(self, df):
        """Extract temporal, calendar, and process context features."""
        # Avoid modifying the original
        df = df.copy()

        grouped = df.groupby(self.case_id_col)[self.time_col]

        # Temporal features (in days)
        df['elapsed_time'] = (
            grouped.transform(lambda x: x - x.min()).dt.total_seconds() / 86400
        )
        df['time_since_last_event'] = (
            grouped.diff().dt.total_seconds() / 86400
        ).fillna(0)

        # Trace duration and remaining time
        trace_durations = (
            grouped.transform(lambda x: x.max() - x.min()).dt.total_seconds() / 86400
        )
        df['trace_duration'] = trace_durations
        df['remaining_time'] = trace_durations - df['elapsed_time']

        # Event position in trace
        df['event_position'] = df.groupby(self.case_id_col).cumcount() + 1

        # Process context features
        df['executed_events_count'] = df.groupby(self.time_col)[
            self.case_id_col
        ].transform('count')
        df['new_traces_count'] = df.groupby(
            df.groupby(self.case_id_col)[self.time_col].transform('min')
        )[self.case_id_col].transform('count')

        # Resources used at each timestamp
        if self.resource_col in df.columns:
            df['resources_used_count'] = df.groupby(self.time_col)[
                self.resource_col
            ].transform('nunique')

        # Calendar Features
        df['day_of_week'] = df[self.time_col].dt.dayofweek
        hour = df[self.time_col].dt.hour
        df['hour_of_day'] = df[self.time_col].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        return df

    def get_prefixes(self, df):
        prefixes = []
        for case_id, group in df.groupby(self.case_id_col):
            events = group.to_dict('records')
            limit = (
                min(len(events), self.max_prefix) if self.max_prefix else len(events)
            )

            for i in range(self.min_prefix, limit + 1):
                prefix_slice = [dict(e) for e in events[:i]]
                for e in prefix_slice:
                    e['prefix_id'] = f'{case_id}_prefix_{i}'
                    e['prefix_len'] = i
                prefixes.extend(prefix_slice)
        return pd.DataFrame(prefixes)

    def train_test_split(
        self,
        start_date=None,
        end_date=None,
        max_days=None,
        verbose=False,
    ):
        """
        Create train/test split with temporal debiasing (Weytjens methodology).

        Addresses data leakage and bias in standard chronological splits by:
        - Ensuring strict temporal separation (train cases end before test cases start)
        - Debiasing test set by limiting maximum case duration
        - Removing chronological outliers
        """
        if self.raw_df is None:
            raise ValueError('Dataset not loaded. Call load_and_preprocess() first.')

        train_df, test_df, split_info = create_unbiased_temporal_split(
            self.raw_df,
            start_date=start_date,
            end_date=end_date,
            max_days=max_days,
            test_len_share=(1 - self.train_ratio),
            time_col=self.time_col,
            case_id_col=self.case_id_col,
        )

        # Extract features
        train_df = self._extract_features(train_df)
        test_df = self._extract_features(test_df)

        # Print summary if verbose
        if verbose:
            print_split_summary(split_info)

        return train_df, test_df

    @abstractmethod
    def get_labels(self, **kwargs):
        """Task-specific label generation (Outcome, Next Activity, Time)."""
        pass


class OutcomeDataset(BaseLogDataset):
    def __init__(self, label_path, **kwargs):
        super().__init__(**kwargs)

        self.label_path = label_path
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.available_case_ids = None

    def filter_by_labels(self):
        """Filter dataframe to keep only cases that have labels."""
        if self.raw_df is None:
            raise ValueError(
                'Must call load_and_preprocess() before filtering by labels'
            )

        labels_path = self.label_path
        labels_df = pd.read_csv(labels_path)

        # Available case IDs from labels
        self.available_case_ids = {str(case_id) for case_id in labels_df.iloc[:, 0]}

        original_cases = self.raw_df[self.case_id_col].nunique()
        self.raw_df = self.raw_df[
            self.raw_df[self.case_id_col].isin(self.available_case_ids)
        ]
        filtered_cases = self.raw_df[self.case_id_col].nunique()

        dropped = original_cases - filtered_cases
        if dropped > 0:
            print(f'Filtered out {dropped} unlabeled cases.')
            print(f'({original_cases} -> {filtered_cases})')

        return self

    def get_labels(self, df, encode=True):
        """Map outcome labels to dataframe using case IDs."""
        labels_df = pd.read_csv(self.label_path)

        # Assuming CSV has columns: [case_id, outcome]
        label_map = dict(
            zip(labels_df.iloc[:, 0].astype(str), labels_df.iloc[:, 1], strict=True)
        )

        labels = df[self.case_id_col].map(label_map)

        # Check for missing labels
        missing_mask = labels.isna()
        if missing_mask.any():
            n_missing = missing_mask.sum()
            missing_cases = df.loc[missing_mask, self.case_id_col].unique()
            print(f'{n_missing} rows from {len(missing_cases)} cases are not labeled')
            print('Consider calling filter_by_labels() after load_and_preprocess()')

            # Filter out rows with missing labels
            df = df[~missing_mask]
            labels = labels[~missing_mask]

        if encode:
            if self.classes_ is None:
                labels_encoded = self.label_encoder.fit_transform(labels)
                self.classes_ = self.label_encoder.classes_
                print(f'Label encoding: {dict(enumerate(self.classes_))}')
            else:
                labels_encoded = self.label_encoder.transform(labels)
            return pd.Series(labels_encoded, index=labels.index)

        return labels

    def decode_labels(self, encoded_labels):
        """Decode numeric labels back to original string labels."""
        if self.label_encoder is None or self.classes_ is None:
            raise ValueError('Label encoder not fitted. Call prepare_labels first.')
        return self.label_encoder.inverse_transform(encoded_labels)


class NextActivityDataset(BaseLogDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def get_labels(self, df, encode=True):
        if df is None:
            raise ValueError('Dataframe is required.')

        df = df.sort_values([self.case_id_col, self.time_col]).copy()

        next_activity = df.groupby(self.case_id_col)[self.activity_col].shift(-1)
        mask = next_activity.notna()

        labels = next_activity[mask]
        df = df[mask]

        if encode:
            if self.classes_ is None:
                encoded = self.label_encoder.fit_transform(labels)
                self.classes_ = self.label_encoder.classes_
            else:
                encoded = self.label_encoder.transform(labels)
            return pd.Series(encoded, index=df.index)

        return labels

    def decode_labels(self, encoded_labels):
        if self.classes_ is None:
            raise ValueError('Label encoder not fitted.')
        return self.label_encoder.inverse_transform(encoded_labels)
