import numpy as np
import pandas as pd


def start_from_date(dataset, start_date):
    """Remove cases starting before start_date from dataset."""
    dataset = dataset.copy()
    case_starts_df = pd.DataFrame(
        dataset.groupby('case:concept:name')['time:timestamp'].min().reset_index()
    )
    case_starts_df['date'] = case_starts_df['time:timestamp'].dt.to_period('M')
    cases_after = case_starts_df[case_starts_df['date'].astype('str') >= start_date][
        'case:concept:name'
    ].values
    dataset = dataset[dataset['case:concept:name'].isin(cases_after)]
    return dataset


def end_before_date(dataset, end_date):
    """Remove cases ending after end_date from dataset."""
    dataset = dataset.copy()
    case_stops_df = pd.DataFrame(
        dataset.groupby('case:concept:name')['time:timestamp'].max().reset_index()
    )
    case_stops_df['date'] = case_stops_df['time:timestamp'].dt.to_period('M')
    cases_before = case_stops_df[case_stops_df['date'].astype('str') <= end_date][
        'case:concept:name'
    ].values
    dataset = dataset[dataset['case:concept:name'].isin(cases_before)]
    return dataset


def remove_chronological_outliers(dataset, start_date=None, end_date=None):
    """Remove cases starting before start_date or ending after end_date."""
    dataset = dataset.copy()
    dataset['time:timestamp'] = pd.to_datetime(dataset['time:timestamp'], utc=True)

    if start_date:
        dataset = start_from_date(dataset, start_date)

    if end_date:
        dataset = end_before_date(dataset, end_date)

    return dataset


def limit_case_duration(dataset, max_duration):
    """Limit dataset to cases shorter than max_duration and debias the end."""
    dataset = dataset.copy()
    dataset['time:timestamp'] = pd.to_datetime(dataset['time:timestamp'], utc=True)

    # Compute each case's duration
    agg_dict = {'time:timestamp': ['min', 'max']}
    duration_df = pd.DataFrame(
        dataset.groupby('case:concept:name').agg(agg_dict)
    ).reset_index()

    duration_df['duration'] = (
        duration_df[('time:timestamp', 'max')] - duration_df[('time:timestamp', 'min')]
    ).dt.total_seconds() / (24 * 60 * 60)

    # Condition 1: Cases are shorter than max_duration
    condition_1 = duration_df['duration'] <= max_duration * 1.00000000001
    cases_retained = duration_df[condition_1]['case:concept:name'].values
    dataset = dataset[dataset['case:concept:name'].isin(cases_retained)].reset_index(
        drop=True
    )

    # Condition 2: Drop cases starting after (last_timestamp - max_duration)
    # This debiases the end of the dataset
    latest_start = dataset['time:timestamp'].max() - pd.Timedelta(
        max_duration, unit='D'
    )
    condition_2 = duration_df[('time:timestamp', 'min')] <= latest_start
    cases_retained = duration_df[condition_2]['case:concept:name'].values
    dataset = dataset[dataset['case:concept:name'].isin(cases_retained)].reset_index(
        drop=True
    )

    return dataset, latest_start


def train_test_split(df, test_len, latest_start, targets):
    """
    Split dataset with strict temporal separation and test set debiasing.

    This is the core splitting function that ensures training cases complete
    BEFORE test cases start, preventing data leakage.
    """
    df = df.copy()

    # Get case start and end times
    case_starts_df = df.groupby('case:concept:name')['time:timestamp'].min()
    case_nr_list_start = case_starts_df.sort_values().index.array
    case_stops_df = df.groupby('case:concept:name')['time:timestamp'].max().to_frame()

    # Determine temporal separation point
    first_test_case_nr = int(len(case_nr_list_start) * (1 - test_len))
    first_test_start_time = np.sort(case_starts_df.values)[first_test_case_nr]

    ### TEST SET ###
    # Include cases that END after separation time
    test_case_nrs = case_stops_df[
        case_stops_df['time:timestamp'].values >= first_test_start_time
    ].index.array
    df_test_all = df[df['case:concept:name'].isin(test_case_nrs)].reset_index(drop=True)

    # Drop events past latest_start (debiasing)
    df_test = df_test_all[df_test_all['time:timestamp'] <= latest_start]

    # Mark boundary events as NaN (events before separation in test cases)
    df_test.loc[df_test['time:timestamp'].values < first_test_start_time, targets] = (
        np.nan
    )

    ### TRAINING SET ###
    # Include only cases that COMPLETE before separation time
    train_case_nrs = case_stops_df[
        case_stops_df['time:timestamp'].values < first_test_start_time
    ].index.array
    df_train = df[df['case:concept:name'].isin(train_case_nrs)].reset_index(drop=True)

    return df_train, df_test


def temporal_train_test_split(
    dataset, test_len_share=0.2, latest_start=None, mark_boundary_events=True
):
    """
    Split dataset with strict temporal separation and optional boundary marking.

    This function ensures training cases complete BEFORE test cases start,
    preventing data leakage. Optionally marks events in boundary cases.

    """
    dataset = dataset.copy()
    dataset['time:timestamp'] = pd.to_datetime(dataset['time:timestamp'], utc=True)

    # Get case start and end times
    case_starts_df = dataset.groupby('case:concept:name')['time:timestamp'].min()
    case_nr_list_start = case_starts_df.sort_values().index.array
    case_stops_df = (
        dataset.groupby('case:concept:name')['time:timestamp'].max().to_frame()
    )

    # Determine temporal separation point
    first_test_case_nr = int(len(case_nr_list_start) * (1 - test_len_share))
    first_test_start_time = np.sort(case_starts_df.values)[first_test_case_nr]

    ### TEST SET ###
    # Include cases that END after separation time
    test_case_nrs = case_stops_df[
        case_stops_df['time:timestamp'].values >= first_test_start_time
    ].index.array
    df_test_all = dataset[dataset['case:concept:name'].isin(test_case_nrs)].reset_index(
        drop=True
    )

    # Drop events past latest_start if specified (debiasing)
    if latest_start is not None:
        df_test = df_test_all[df_test_all['time:timestamp'] <= latest_start]
    else:
        df_test = df_test_all

    # Mark boundary events (events before separation time in test cases)
    if mark_boundary_events:
        df_test['_is_boundary_event'] = (
            df_test['time:timestamp'].values < first_test_start_time
        )

    ### TRAINING SET ###
    # Include only cases that COMPLETE before separation time
    train_case_nrs = case_stops_df[
        case_stops_df['time:timestamp'].values < first_test_start_time
    ].index.array
    df_train = dataset[dataset['case:concept:name'].isin(train_case_nrs)].reset_index(
        drop=True
    )

    return df_train, df_test


def create_unbiased_temporal_split(
    dataset,
    start_date=None,
    end_date=None,
    max_days=None,
    test_len_share=0.2,
    mark_boundary_events=True,
    time_col='time:timestamp',
    case_id_col='case:concept:name',
):
    """
    Create train/test split with temporal debiasing (Weytjens methodology).

    This function implements the complete preprocessing and splitting pipeline:
    1. Remove chronological outliers (start_date, end_date filters)
    2. Remove duplicate events
    3. Limit maximum case duration and debias dataset end
    4. Perform strict temporal train/test split
    """
    dataset = dataset.copy()
    dataset[time_col] = pd.to_datetime(dataset[time_col], utc=True)

    # Step 1: Remove chronological outliers
    dataset = remove_chronological_outliers(dataset, start_date, end_date)

    # Step 2: Remove duplicates
    dataset.drop_duplicates(inplace=True)
    original_cases = dataset[case_id_col].nunique()

    # Step 3: Limit duration and debias
    if max_days:
        dataset, latest_start = limit_case_duration(dataset, max_days)
        filtered_cases = dataset[case_id_col].nunique()
    else:
        latest_start = dataset[time_col].max()
        filtered_cases = original_cases

    # Step 4: Temporal split
    train_df, test_df = temporal_train_test_split(
        dataset,
        test_len_share=test_len_share,
        latest_start=latest_start,
        mark_boundary_events=mark_boundary_events,
    )

    # Gather metadata
    metadata = {
        'original_cases': original_cases,
        'filtered_cases': filtered_cases,
        'train_cases': train_df[case_id_col].nunique(),
        'test_cases': test_df[case_id_col].nunique(),
        'train_events': len(train_df),
        'test_events': len(test_df),
        'latest_start': latest_start,
        'test_share': test_len_share,
        'max_days': max_days,
        'start_date': start_date,
        'end_date': end_date,
    }

    return train_df, test_df, metadata


def print_split_summary(metadata):
    """Print a summary of the train/test split."""
    print('\n' + '=' * 60)
    print('TRAIN/TEST SPLIT SUMMARY')
    print('=' * 60)
    print(f'Original cases: {metadata["original_cases"]}')
    print(f'After filtering: {metadata["filtered_cases"]}')
    print('\nTraining set:')
    print(f'  Cases: {metadata["train_cases"]}')
    print(f'  Events: {metadata["train_events"]}')
    print('\nTest set:')
    print(f'  Cases: {metadata["test_cases"]}')
    print(f'  Events: {metadata["test_events"]}')
    print('\nParameters:')
    print(f'  Test share: {metadata["test_share"]:.1%}')
    print(f'  Max duration: {metadata["max_days"]} days')
    print(f'  Start date filter: {metadata["start_date"]}')
    print(f'  End date filter: {metadata["end_date"]}')
    print(f'  Latest start (debiased): {metadata["latest_start"]}')
    print('=' * 60 + '\n')
