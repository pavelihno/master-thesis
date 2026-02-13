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
    input_dir='datasets/raw',
    output_path='datasets/stats.json',
):
    """Extract statistics for all datasets and save to JSON."""
    all_stats = {}

    # Get all .xes files in the input directory
    if not os.path.exists(input_dir):
        print(f'Error: Input directory {input_dir} does not exist')
        return

    xes_files = [f for f in os.listdir(input_dir) if f.endswith('.xes')]

    if not xes_files:
        print(f'Warning: No .xes files found in {input_dir}')
        return

    print(f'Found {len(xes_files)} XES files to process\n')

    # Process each dataset
    for xes_file in sorted(xes_files):
        dataset_key = os.path.splitext(xes_file)[0]

        print(f'\n{"=" * 60}')
        print(f'Processing: {dataset_key}')
        print(f'{"=" * 60}')

        xes_path = os.path.join(input_dir, xes_file)

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
