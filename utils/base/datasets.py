from abc import ABC, abstractmethod

import pandas as pd
import pm4py

from utils.constants import CASE_PREFIX_COL
from utils.experiment.temporal_splitting import (
    create_unbiased_temporal_split,
    print_split_summary,
)


class BaseLogDataset(ABC):
    def __init__(
        self,
        dataset_path=None,
        raw_df=None,
        train_ratio=0.8,
        min_prefix=None,
        max_prefix=None,
        feature_extractors=None,
        case_id_col='case:concept:name',
        time_col='time:timestamp',
        activity_col='concept:name',
        resource_col='org:resource',
    ):
        self.raw_df = None
        self.dataset_path = dataset_path

        if raw_df is not None:
            self.raw_df = raw_df
        elif self.dataset_path is not None:
            self.raw_df = pm4py.read_xes(self.dataset_path)
        else:
            raise ValueError('Either dataset_path or raw_df must be provided.')

        self.feature_extractors = (
            feature_extractors if feature_extractors is not None else []
        )

        self.train_ratio = train_ratio
        self.min_prefix = min_prefix if min_prefix is not None else 1
        self.max_prefix = max_prefix

        self.case_id_col = case_id_col
        self.time_col = time_col
        self.activity_col = activity_col
        self.resource_col = resource_col

        self.raw_df[self.case_id_col] = self.raw_df[self.case_id_col].astype(str)
        self.raw_df[self.time_col] = pd.to_datetime(
            self.raw_df[self.time_col], utc=True
        )
        self.raw_df = self.raw_df.sort_values([self.time_col, self.case_id_col])

    @abstractmethod
    def get_prefixes_and_labels(self, df):
        """Get prefixes and labels."""
        pass

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
            raise ValueError('Dataset not loaded.')

        # Fit and transform features
        features_df = self.raw_df.copy()
        for feature_extractor in self.feature_extractors:
            features_df = feature_extractor.fit_transform(features_df)

        train_df, test_df, split_info = create_unbiased_temporal_split(
            features_df,
            start_date=start_date,
            end_date=end_date,
            max_days=max_days,
            test_len_share=(1 - self.train_ratio),
            time_col=self.time_col,
            case_id_col=self.case_id_col,
        )

        # Print summary if verbose
        if verbose:
            print_split_summary(split_info)

        return train_df, test_df


class OutcomeDataset(BaseLogDataset):
    def __init__(self, label_path=None, labels_df=None, **kwargs):
        super().__init__(**kwargs)

        self.label_path = label_path
        self.labels_df = labels_df

    def get_prefixes_and_labels(self, df):
        """Generate all prefixes and their outcome labels."""

        prefixes = []
        for case_id, group in df.groupby(self.case_id_col):
            events = group.to_dict('records')
            limit = (
                min(len(events), self.max_prefix) if self.max_prefix else len(events)
            )

            for i in range(self.min_prefix, limit + 1):
                prefix_slice = [dict(e) for e in events[:i]]
                for e in prefix_slice:
                    e[CASE_PREFIX_COL] = f'{case_id}_prefix_{i}'
                    e['prefix_len'] = i
                prefixes.extend(prefix_slice)

        prefixes_df = pd.DataFrame(prefixes)

        if self.labels_df is None:
            if self.label_path is not None:
                self.labels_df = pd.read_csv(self.label_path)
            else:
                raise ValueError('Either labels_df or label_path must be provided.')

        # Assuming CSV has columns: [case_id, outcome]
        label_map = dict(
            zip(
                self.labels_df.iloc[:, 0].astype(str),
                self.labels_df.iloc[:, 1],
                strict=True,
            )
        )

        case_ids = prefixes_df[self.case_id_col]
        labels = case_ids.map(label_map)

        # Check for missing labels
        missing_mask = labels.isna()
        if missing_mask.any():
            n_missing = missing_mask.sum()
            print(f'WARNING: {n_missing} prefix rows have missing labels')
            prefixes_df = prefixes_df[~missing_mask].reset_index(drop=True)
            labels = labels[~missing_mask].reset_index(drop=True)

        return prefixes_df, labels


class NextActivityDataset(BaseLogDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_prefixes_and_labels(self, df):
        """Generate prefixes and their next activity labels.

        For each prefix, the label is the activity that comes after it.
        Last events in traces are excluded (no next activity to predict).
        """
        # Generate prefixes excluding last event of each trace
        prefixes = []
        for case_id, group in df.groupby(self.case_id_col):
            events = group.to_dict('records')
            limit = (
                min(len(events), self.max_prefix) if self.max_prefix else len(events)
            )

            for i in range(self.min_prefix, limit):
                next_activity = events[i][self.activity_col]
                prefix_slice = [dict(e) for e in events[:i]]
                for e in prefix_slice:
                    e[CASE_PREFIX_COL] = f'{case_id}_prefix_{i}'
                    e['prefix_len'] = i
                    e['next_activity'] = next_activity
                prefixes.extend(prefix_slice)

        prefixes_df = pd.DataFrame(prefixes)

        labels_df = prefixes_df['next_activity']

        return prefixes_df, labels_df
