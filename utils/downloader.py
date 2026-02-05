import gzip
import json
import os
import urllib.request
import zipfile

import pandas as pd
import pm4py


def load_links(json_path):
    """Load dataset links from JSON file."""
    with open(json_path) as file:
        return json.load(file)


def download_file(url, output_path):
    """Download a file from URL to output path."""
    urllib.request.urlretrieve(url, output_path)
    print(f'Downloaded: {output_path}')


def unzip_file(zip_path, extract_dir):
    """Extract contents of a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f'Extracted: {zip_path}')


def filter_files(directory, dataset_key):
    """Keep only .xes, .xes.gz, and .csv files, delete all others."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(('.xes', '.xes.gz', '.csv')):
                continue
            else:
                os.remove(file_path)
                print(f'Removed unwanted file: {file_path}')


def decompress_gz(gz_path, dataset_key):
    """Decompress .xes.gz file to .xes format."""
    output_filename = f'{dataset_key}.xes'
    output_path = os.path.join(os.path.dirname(gz_path), output_filename)

    with gzip.open(gz_path, 'rt', encoding='utf-8') as gz_file:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(gz_file.read())

    os.remove(gz_path)
    print(f'Decompressed: {gz_path} -> {output_path}')
    return output_path


def convert_csv_to_xes(csv_path, dataset_key):
    """Convert CSV file to XES format."""
    df = pd.read_csv(csv_path, sep=',')

    # Define possible column mappings to try
    column_mappings = [
        # Standard XES format
        {
            'case_id': 'case:concept:name',
            'activity_key': 'concept:name',
            'timestamp_key': 'time:timestamp',
        },
        # Other common formats
        {
            'case_id': 'Case ID',
            'activity_key': 'Activity',
            'timestamp_key': 'Complete Timestamp',
        },
        {
            'case_id': 'case_id',
            'activity_key': 'activity',
            'timestamp_key': 'timestamp',
        },
        {
            'case_id': 'CaseID',
            'activity_key': 'Activity',
            'timestamp_key': 'Timestamp'
        },
    ]

    # Try each mapping until one works
    formatted_df = None
    for mapping in column_mappings:
        try:
            formatted_df = pm4py.format_dataframe(
                df,
                case_id=mapping['case_id'],
                activity_key=mapping['activity_key'],
                timestamp_key=mapping['timestamp_key'],
            )
            print(f'Successfully formatted with mapping: {mapping}')
            break
        except Exception:
            continue

    if formatted_df is None:
        print(f'Error: Could not format dataframe for {csv_path}')
        print(f'Available columns: {list(df.columns)}')
        raise ValueError(f'No valid column mapping found for {csv_path}')

    # Generate output filename from dataset key
    output_filename = f'{dataset_key}.xes'

    # Convert to event log and write XES
    event_log = pm4py.convert_to_event_log(formatted_df)
    output_path = os.path.join(os.path.dirname(csv_path), output_filename)
    pm4py.write_xes(event_log, output_path)

    os.remove(csv_path)
    print(f'Converted CSV to XES: {csv_path} -> {output_path}')
    return output_path


def process_files(directory, dataset_key):
    """Process all files in directory: decompress .gz files and convert CSV to XES."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith('.xes.gz'):
                decompress_gz(file_path, dataset_key)
            elif file.endswith('.csv'):
                convert_csv_to_xes(file_path, dataset_key)


def download_datasets(json_path='datasets/links.json', output_dir='datasets/raw'):
    """Main function to download and process all datasets.

    Args:
        json_path: Path to JSON file containing dataset links
        output_dir: Directory where datasets will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset links
    links = load_links(json_path)

    # Process each dataset
    for dataset_key, url in links.items():
        print(f'\n{"=" * 60}')
        print(f'Processing: {dataset_key}')
        print(f'{"=" * 60}')

        # Download zip file
        zip_path = os.path.join(output_dir, f'{dataset_key}.zip')
        download_file(url, zip_path)

        # Extract zip file
        unzip_file(zip_path, output_dir)

        # Remove zip file
        os.remove(zip_path)
        print(f'Removed zip: {zip_path}')

        # Filter files (keep only relevant formats)
        filter_files(output_dir, dataset_key)

        # Process files (decompress .gz, convert CSV)
        process_files(output_dir, dataset_key)

        print(f'Completed: {dataset_key}')

    print(f'\n{"=" * 60}')
    print('All datasets downloaded and processed successfully!')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    download_datasets()
