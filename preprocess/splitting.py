import os
from abc import ABC, abstractmethod

import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import EventLog, Trace


class DatasetSplitter(ABC):
    """Base class for dataset splitting operations."""

    def __init__(
        self,
        dataset_name,
        dataset_folder='datasets/raw',
        output_folder='datasets/raw',
    ):
        self.dataset_name = dataset_name
        self.dataset_path = f'{dataset_folder}/{dataset_name}.xes'
        self.output_folder = output_folder

    def run(self):
        """Main method to execute the splitting process."""
        print(f'\n=== Splitting {self.dataset_name} ===')

        # Read the original event log
        log_df = pm4py.read_xes(self.dataset_path)
        log = log_converter.apply(log_df)

        # Apply splitting logic
        subdatasets = self.apply_splitting_rules(log, log_df)

        # Save each subdataset
        for subdataset_name, subdataset_log in subdatasets.items():
            self._save_subdataset(subdataset_name, subdataset_log)

        print(
            f'Successfully split {self.dataset_name} '
            f'into {len(subdatasets)} subdatasets\n'
        )
        return subdatasets

    @abstractmethod
    def apply_splitting_rules(self, log, log_df):
        """Apply splitting logic to the event log."""
        pass

    def _filter_trace(self, trace, prefix):
        """Filter a trace to keep only events with the given prefix."""

        filtered_trace = Trace()

        # Copy trace attributes
        for attr_key, attr_value in trace.attributes.items():
            filtered_trace.attributes[attr_key] = attr_value

        # Filter events
        for event in trace:
            activity = event.get('concept:name', '')
            if activity.startswith(prefix):
                filtered_trace.append(event)

        # Return None if no events remain
        return filtered_trace if len(filtered_trace) > 0 else None

    def _save_subdataset(self, subdataset_name, subdataset_log):
        """Save a subdataset to an XES file."""
        output_path = os.path.join(self.output_folder, f'{subdataset_name}.xes')

        xes_exporter.apply(subdataset_log, output_path)
        print(
            f'  - Saved {subdataset_name}: '
            f'{len(subdataset_log)} traces -> {output_path}'
        )


class BPIC12Splitter(DatasetSplitter):
    """
    Split BPIC_12 dataset into three subdatasets based on subprocess prefixes.

    The BPIC 2012 event log contains three interleaved subprocesses:
    - A_*: Application subprocess
    - O_*: Offer subprocess
    - W_*: Work subprocess
    """

    def apply_splitting_rules(self, log, log_df):
        """Split BPIC_12 based on activity name prefixes."""
        # Initialize subdataset logs
        log_A = EventLog()
        log_O = EventLog()
        log_W = EventLog()

        # Copy log attributes
        for attr_key, attr_value in log.attributes.items():
            log_A.attributes[attr_key] = attr_value
            log_O.attributes[attr_key] = attr_value
            log_W.attributes[attr_key] = attr_value

        # Extract activities from each subprocess for every trace
        for trace in log:
            # Filter and add to subprocess A if it has A_ activities
            filtered_trace_A = self._filter_trace(trace, 'A_')
            if filtered_trace_A:
                log_A.append(filtered_trace_A)

            # Filter and add to subprocess O if it has O_ activities
            filtered_trace_O = self._filter_trace(trace, 'O_')
            if filtered_trace_O:
                log_O.append(filtered_trace_O)

            # Filter and add to subprocess W if it has W_ activities
            filtered_trace_W = self._filter_trace(trace, 'W_')
            if filtered_trace_W:
                log_W.append(filtered_trace_W)

        return {
            'BPIC_12_A': log_A,
            'BPIC_12_O': log_O,
            'BPIC_12_W': log_W,
        }


class BPIC18Splitter(DatasetSplitter):
    """
    Split BPIC_18 dataset into 3 subdatasets based on event attribute doctype:
    - Geo parcel document
    - Parcel document
    - Control summary
    """

    def apply_splitting_rules(self, log, log_df):
        log_G = EventLog()
        log_P = EventLog()
        log_C = EventLog()

        for attr_key, attr_value in log.attributes.items():
            log_G.attributes[attr_key] = attr_value
            log_P.attributes[attr_key] = attr_value
            log_C.attributes[attr_key] = attr_value

        def _get_doctype(event):
            return (
                event.get('doctype')
                or event.get('doc_type')
                or event.get('docType')
                or ''
            )

        for trace in log:
            filtered_G = Trace()
            filtered_P = Trace()
            filtered_C = Trace()

            for attr_key, attr_value in trace.attributes.items():
                filtered_G.attributes[attr_key] = attr_value
                filtered_P.attributes[attr_key] = attr_value
                filtered_C.attributes[attr_key] = attr_value

            for event in trace:
                doctype = _get_doctype(event).lower()
                if doctype == 'geo parcel document':
                    filtered_G.append(event)
                elif doctype == 'parcel document':
                    filtered_P.append(event)
                elif doctype == 'control summary':
                    filtered_C.append(event)

            if len(filtered_G) > 0:
                log_G.append(filtered_G)
            if len(filtered_P) > 0:
                log_P.append(filtered_P)
            if len(filtered_C) > 0:
                log_C.append(filtered_C)

        return {
            'BPIC_18_G': log_G,
            'BPIC_18_P': log_P,
            'BPIC_18_C': log_C,
        }


def split_datasets():
    """Main function to split all configured datasets."""
    print('=' * 60)
    print('Dataset Splitting')
    print('=' * 60)

    splitters = [
        ('BPIC_12', BPIC12Splitter),
        ('BPIC_18', BPIC18Splitter),
    ]

    for dataset_name, splitter_class in splitters:
        try:
            splitter = splitter_class(dataset_name)
            splitter.run()
        except FileNotFoundError:
            print(f'Warning: {dataset_name}.xes not found, skipping...\n')
        except Exception as e:
            print(f'Failed to split {dataset_name}: {e}\n')

    print('=' * 60)
    print('Dataset splitting complete!')
    print('=' * 60)


if __name__ == '__main__':
    split_datasets()
