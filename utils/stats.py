import pandas as pd


def print_class_balance(labels, label_col='outcome', dataset_names=None):
    """Calculate class balance as percentage distribution of each class."""
    if isinstance(labels, pd.Series):
        labels_series = labels
    elif isinstance(labels, pd.DataFrame):
        labels_series = labels[label_col]
    else:
        raise TypeError('labels must be a pandas Series or DataFrame')

    total = len(labels_series)
    if total == 0:
        return {}

    counts = labels_series.value_counts().sort_index()
    percentages = labels_series.value_counts(normalize=True).sort_index() * 100

    balance_df = pd.DataFrame(
        {
            'Class': counts.index,
            'Count': counts.values,
            'Percentage': percentages.values,
        }
    )

    # Header
    dataset_names_str = ', '.join(dataset_names) if dataset_names else ''
    title = f'Class Balance{f" - {dataset_names_str}" if dataset_names else ""}'
    print(f'\n{"=" * 60}')
    print(f'{title:^60}')
    print(f'{"=" * 60}')

    # Table header
    print(f'{"Class":<30} {"Count":>12} {"Percentage":>12}')
    print(f'{"-" * 60}')

    # Table rows
    for cls, count, pct in zip(
        counts.index, counts.values, percentages.values, strict=False
    ):
        print(f'{str(cls):<30} {count:>12,} {pct:>11.2f}%')

    # Footer
    print(f'{"-" * 60}')
    print(f'{"Total":<30} {total:>12,} {"100.00%":>12}')
    print(f'{"=" * 60}\n')
