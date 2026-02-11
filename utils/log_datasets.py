from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pm4py


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
        feature_names=None,
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
        self.feature_names = feature_names

        # Dataset stats
        self.num_cases = None
        self.num_events = None

        self.raw_df = None

    def load_and_preprocess(self):
        """Loads XES and extracts basic temporal features."""
        print(f'Loading {self.dataset_name}...')
        dataset_path = f'{self.dataset_folder}/{self.dataset_name}.xes'
        self.raw_df = pm4py.read_xes(dataset_path)

        # Basic cleanup and sorting
        self.raw_df[self.time_col] = pd.to_datetime(
            self.raw_df[self.time_col], utc=True
        )
        self.raw_df = self.raw_df.sort_values([self.case_id_col, self.time_col])

        self.num_cases = self.raw_df[self.case_id_col].nunique()
        self.num_events = len(self.raw_df)

        return self

    def _extract_features(self, df):
        """Extract temporal and calendar features, then filter by feature_names."""
        # Avoid modifying the original
        df = df.copy()

        grouped = df.groupby(self.case_id_col)[self.time_col]

        df['time_since_start'] = (
            grouped.transform(lambda x: x - x.min()).dt.total_seconds() / 86400
        )
        df['time_since_last_event'] = (
            grouped.diff().dt.total_seconds() / 86400
        ).fillna(0)
        df['event_index'] = df.groupby(self.case_id_col).cumcount() + 1

        # Calendar Features
        df['day_of_week'] = df[self.time_col].dt.dayofweek
        hour = df[self.time_col].dt.hour
        df['hour_of_day'] = df[self.time_col].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Filter features: keep all columns if features are not specified
        if self.feature_names is None:
            return df

        # Columns that must always be kept
        essential_cols = [self.case_id_col, self.time_col, self.activity_col]

        cols_to_keep = list(set(essential_cols + self.feature_names))
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]

        return df[cols_to_keep]

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

    def prepare_labels(self, df):
        """Map outcome labels to dataframe using case IDs."""
        labels_path = f'{self.labels_folder}/{self.dataset_name}.csv'
        labels_df = pd.read_csv(labels_path)

        # Assuming CSV has columns: [case_id, outcome]
        label_map = dict(zip(labels_df.iloc[:, 0], labels_df.iloc[:, 1], strict=True))

        labels = df[self.case_id_col].map(label_map)

        return labels
