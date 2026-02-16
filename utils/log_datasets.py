from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pm4py
from sklearn.preprocessing import LabelEncoder


class BaseLogDataset(ABC):
    def __init__(
        self,
        dataset_name,
        dataset_folder,
        labels_folder=None,
        train_ratio=0.8,
        min_prefix=3,
        max_prefix=None,
        case_id_col='case:concept:name',
        time_col='time:timestamp',
        activity_col='concept:name',
        resource_col='org:resource',
    ):
        # Path constants
        self.dataset_folder = dataset_folder
        self.labels_folder = labels_folder

        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.min_prefix = min_prefix
        self.max_prefix = max_prefix

        # Column constants
        self.case_id_col = case_id_col
        self.time_col = time_col
        self.activity_col = activity_col
        self.resource_col = resource_col

        self.raw_df = None

    def load_and_preprocess(self):
        """Loads XES and extracts basic temporal features."""
        print(f'Loading {self.dataset_name}...')

        dataset_path = f'{self.dataset_folder}/{self.dataset_name}.xes'

        self.raw_df = pm4py.read_xes(dataset_path)
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
        """Optimized Pandas-based prefix generation."""
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

    def train_test_split(self):
        """Extract features and split into train/test based on chronological order."""
        # Chronological split by case start time
        case_starts = (
            self.raw_df.groupby(self.case_id_col)[self.time_col]
            .min()
            .sort_values()
            .index.tolist()
        )

        split_idx = int(len(case_starts) * self.train_ratio)
        train_ids = set(case_starts[:split_idx])
        test_ids = set(case_starts[split_idx:])

        train_df = self.raw_df[self.raw_df[self.case_id_col].isin(train_ids)]
        test_df = self.raw_df[self.raw_df[self.case_id_col].isin(test_ids)]

        # Extract features
        train_df = self._extract_features(train_df)
        test_df = self._extract_features(test_df)

        print(f'Train cases: {len(train_ids)}, Test cases: {len(test_ids)}')
        return train_df, test_df

    @abstractmethod
    def prepare_labels(self, **kwargs):
        """Task-specific label generation (Outcome, Next Activity, Time)."""
        pass


class OutcomeDataset(BaseLogDataset):
    def __init__(self, dataset_name, dataset_folder, labels_folder, **kwargs):
        super().__init__(dataset_name, dataset_folder, labels_folder, **kwargs)
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.available_case_ids = None

    def filter_by_labels(self):
        """Filter dataframe to keep only cases that have labels."""
        if self.raw_df is None:
            raise ValueError(
                'Must call load_and_preprocess() before filtering by labels'
            )

        labels_path = f'{self.labels_folder}/{self.dataset_name}.csv'
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

    def prepare_labels(self, df, encode=True):
        """Map outcome labels to dataframe using case IDs."""
        labels_path = f'{self.labels_folder}/{self.dataset_name}.csv'
        labels_df = pd.read_csv(labels_path)

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
            labels = pd.Series(labels_encoded, index=labels.index)

        return labels if not encode else pd.Series(labels, index=df.index)

    def decode_labels(self, encoded_labels):
        """Decode numeric labels back to original string labels."""
        if self.label_encoder is None or self.classes_ is None:
            raise ValueError('Label encoder not fitted. Call prepare_labels first.')
        return self.label_encoder.inverse_transform(encoded_labels)
