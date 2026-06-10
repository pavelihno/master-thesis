from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class FeatureExtractor(ABC):
    """Base class for all feature extractors."""

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """Fit the feature extractor on the data."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data into features."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


class TemporalFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        case_id_col='case:concept:name',
        time_col='time:timestamp',
    ):
        self.case_id_col = case_id_col
        self.time_col = time_col

    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        grouped = df.groupby(self.case_id_col)[self.time_col]

        df['elapsed_time'] = (
            grouped.transform(lambda x: x - x.min()).dt.total_seconds() / 86400
        )
        df['time_since_last_event'] = (
            grouped.diff().dt.total_seconds() / 86400
        ).fillna(0)

        trace_durations = (
            grouped.transform(lambda x: x.max() - x.min()).dt.total_seconds() / 86400
        )
        df['trace_duration'] = trace_durations
        df['remaining_time'] = trace_durations - df['elapsed_time']

        return df


class CalendarFeatureExtractor(FeatureExtractor):
    def __init__(self, time_col='time:timestamp'):
        self.time_col = time_col

    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['day_of_week'] = df[self.time_col].dt.dayofweek
        hour = df[self.time_col].dt.hour
        df['hour_of_day'] = df[self.time_col].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        return df


class ProcessContextFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        case_id_col='case:concept:name',
        time_col='time:timestamp',
        resource_col='org:resource',
    ):
        self.case_id_col = case_id_col
        self.time_col = time_col
        self.resource_col = resource_col

    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['event_position'] = df.groupby(self.case_id_col).cumcount() + 1
        df['executed_events_count'] = df.groupby(self.time_col)[
            self.case_id_col
        ].transform('count')
        df['new_traces_count'] = df.groupby(
            df.groupby(self.case_id_col)[self.time_col].transform('min')
        )[self.case_id_col].transform('count')

        if self.resource_col in df.columns:
            df['resources_used_count'] = df.groupby(self.time_col)[
                self.resource_col
            ].transform('nunique')
        return df
