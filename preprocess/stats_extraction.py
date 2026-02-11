import json
import os

import pm4py


def load_links(json_path):
    """Load dataset links from JSON file."""
    with open(json_path) as file:
        return json.load(file)


def get_dataset_stats(
    log_df, case_id_col='case:concept:name', time_col='time:timestamp'
):
    """Calculate statistics for a dataset."""
    # Case durations in days
    case_durations = log_df.groupby(case_id_col)[time_col].agg(
        lambda x: (x.max() - x.min()).total_seconds() / 86400
    )
    mean_cycle = case_durations.mean()
    normalization_factor = case_durations.max()

    return {
        'mean_cycle': float(mean_cycle),
        'normalization_factor': float(normalization_factor),
    }


def extract_stats(
    json_path='datasets/links.json',
    input_dir='datasets/raw',
    output_path='datasets/stats.json',
):
    """Extract statistics for all datasets and save to JSON.

    Args:
        json_path: Path to JSON file containing dataset links
        input_dir: Directory where XES files are located
        output_path: Path to save statistics JSON file
    """
    links = load_links(json_path)

    all_stats = {}

    # Process each dataset
    for dataset_key in links.keys():
        print(f'\n{"=" * 60}')
        print(f'Processing: {dataset_key}')
        print(f'{"=" * 60}')

        # Construct path to XES file
        xes_path = os.path.join(input_dir, f'{dataset_key}.xes')

        # Check if file exists
        if not os.path.exists(xes_path):
            print(f'Warning: {xes_path} not found, skipping...')
            continue

        try:
            log_df = pm4py.read_xes(xes_path)

            stats = get_dataset_stats(log_df)
            all_stats[dataset_key] = stats

            print(f'Mean cycle time: {stats["mean_cycle"]:.2f} days')
            print(f'Normalization factor: {stats["normalization_factor"]:.2f} days')

        except Exception as e:
            print(f'Error processing {dataset_key}: {str(e)}')
            continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=4)

    print(f'\n{"=" * 60}')
    print(f'Statistics saved to: {output_path}')
    print(f'Total datasets processed: {len(all_stats)}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    extract_stats()
