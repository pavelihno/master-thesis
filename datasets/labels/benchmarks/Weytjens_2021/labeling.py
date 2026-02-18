import os

import pandas as pd
from pm4py.objects.conversion.log import converter
from pm4py.objects.log.importer.xes import importer as xes_importer


def determine_outcome_by_priority(
    activities, keywords_dict, priority_order, outcome_labels
):
    """Determine case outcome based on keyword matching with priority rules."""
    for outcome_type in priority_order:
        if any(kw in activities for kw in keywords_dict.get(outcome_type, [])):
            return outcome_labels.get(outcome_type, outcome_type.capitalize())
    return 'Other'


def extract_labels(
    dataset_path,
    output_path,
    keywords_dict,
    priority_order,
    outcome_labels,
    activity_column='concept:name',
    use_lifecycle=False,
    subprocess_filter=None,
):
    """Generic label extraction for classification tasks."""
    log = xes_importer.apply(dataset_path)
    dataset = converter.apply(log, variant=converter.Variants.TO_DATA_FRAME)

    # Filter by subprocess if specified
    if subprocess_filter:
        if 'subprocess' in dataset.columns:
            original_len = len(dataset)
            dataset = dataset[dataset['subprocess'] == subprocess_filter]
            print(
                f'Filtered to {subprocess_filter}: {len(dataset)}/{original_len} events'
            )
        else:
            print('Warning: subprocess_filter specified but no subprocess column found')

    # Create activity column (with or without lifecycle)
    if use_lifecycle:
        if 'lifecycle:transition' in dataset.columns:
            dataset['activity'] = (
                dataset[activity_column] + '_' + dataset['lifecycle:transition']
            )
            activity_col = 'activity'
        else:
            print('Warning: use_lifecycle=True but no lifecycle:transition column')
            activity_col = activity_column
    else:
        activity_col = activity_column

    # Extract labels for each case
    labels = []
    for case_id, group in dataset.groupby('case:concept:name'):
        activities = set(group[activity_col].values)
        outcome = determine_outcome_by_priority(
            activities, keywords_dict, priority_order, outcome_labels
        )
        labels.append({'case_id': case_id, 'outcome': outcome})

    # Save labels
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(output_path, index=False)
    print(f'Saved {len(labels_df)} labels to {output_path}')
    print(f'Outcome distribution:\n{labels_df["outcome"].value_counts()}')

    return labels_df


def extract_bpic12_labels(
    dataset_path, output_path, variant='A', subprocess_filter=None
):
    """Extract BPIC 2012 labels."""
    # Keywords for classification targets (from Weytjens paper/notebook)
    keywords_dict = {
        'approved': [
            'A_REGISTERED_COMPLETE',
            'A_APPROVED_COMPLETE',
            'O_ACCEPTED_COMPLETE',
            'A_ACTIVATED_COMPLETE',
        ],
        'declined': ['A_DECLINED_COMPLETE', 'O_DECLINED_COMPLETE'],
        'canceled': ['A_CANCELLED_COMPLETE', 'O_CANCELLED_COMPLETE'],
    }

    # Variant-specific priority and labels
    variant_configs = {
        'A': {
            'priority': ['declined', 'approved', 'canceled'],
            'labels': {
                'declined': 'Declined',
                'approved': 'Approved',
                'canceled': 'Cancelled',
            },
        },
        'O': {
            'priority': ['canceled', 'approved', 'declined'],
            'labels': {
                'canceled': 'Cancelled',
                'approved': 'Accepted',
                'declined': 'Declined',
            },
        },
        'W': {
            # For W variant, same keywords apply but different priority/labels
            'priority': ['declined', 'approved', 'canceled'],
            'labels': {
                'declined': 'Declined',
                'approved': 'Approved',
                'canceled': 'Cancelled',
            },
        },
    }

    config = variant_configs.get(variant, variant_configs['A'])

    return extract_labels(
        dataset_path,
        output_path,
        keywords_dict,
        config['priority'],
        config['labels'],
        activity_column='concept:name',
        use_lifecycle=True,
        subprocess_filter=subprocess_filter,
    )


def extract_bpic17_labels(dataset_path, output_path):
    """Extract BPIC 2017 labels."""
    # Keywords for classification targets (from Weytjens paper)
    keywords_dict = {
        'approved': ['O_Accepted'],
        'declined': ['O_Refused'],
        'canceled': ['O_Cancelled'],
    }

    # Priority: canceled > declined > approved (as per notebook)
    priority_order = ['canceled', 'declined', 'approved']
    outcome_labels = {
        'canceled': 'Cancelled',
        'declined': 'Declined',
        'approved': 'Accepted',
    }

    return extract_labels(
        dataset_path,
        output_path,
        keywords_dict,
        priority_order,
        outcome_labels,
        activity_column='concept:name',
        use_lifecycle=False,
    )


def main():
    raw_folder = 'datasets/raw'
    output_folder = 'datasets/labels/benchmarks/Weytjens_2021'

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # BPIC 2012 dataset configurations
    bpic12_configs = [
        ('A', 'Application', 'Application (A)'),
        ('O', 'Offer', 'Offer (O)'),
        ('W', 'Work', 'Work Items (W)'),
    ]

    # Check if we have full BPIC_12 or pre-split files
    has_full_bpic12 = os.path.exists(f'{raw_folder}/BPIC_12.xes')
    has_split_bpic12 = os.path.exists(f'{raw_folder}/BPIC_12_A.xes')

    if has_split_bpic12:
        print('\n' + '=' * 60)
        print('BPIC 2012 - Using pre-split dataset files')
        print('=' * 60)

        for variant, _, display_name in bpic12_configs:
            print(f'\n{display_name}...')
            extract_bpic12_labels(
                f'{raw_folder}/BPIC_12_{variant}.xes',
                f'{output_folder}/BPIC_12_{variant}.csv',
                variant=variant,
            )
    elif has_full_bpic12:
        print('\n' + '=' * 60)
        print('BPIC 2012 - Using full dataset with subprocess filtering')
        print('=' * 60)

        for variant, subprocess, display_name in bpic12_configs:
            print(f'\n{display_name}...')
            extract_bpic12_labels(
                f'{raw_folder}/BPIC_12.xes',
                f'{output_folder}/BPIC_12_{variant}.csv',
                variant=variant,
                subprocess_filter=subprocess,
            )
    else:
        print('\nWarning: No BPIC_12 files found, skipping...')

    # BPIC 2017 with classification outcomes
    print('\n' + '=' * 60)
    print('BPIC 2017')
    print('=' * 60)
    if os.path.exists(f'{raw_folder}/BPIC_17.xes'):
        extract_bpic17_labels(
            f'{raw_folder}/BPIC_17.xes',
            f'{output_folder}/BPIC_17.csv',
        )
    else:
        print('BPIC_17.xes not found, skipping...')

    print('\n' + '=' * 60)
    print('Label extraction complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
