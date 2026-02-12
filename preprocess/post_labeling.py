import pandas as pd
from labeling import DatasetLabeler


class DatasetPostLabeler(DatasetLabeler):
    def __init__(
        self,
        dataset_name,
        label_folder,
        original_label_folder='datasets/labels',
    ):
        self.dataset_name = dataset_name
        self.original_label_path = f'{original_label_folder}/{dataset_name}.csv'
        self.label_path = f'{label_folder}/{dataset_name}.csv'

    def run(self):
        print(f'\n=== Post labeling {self.dataset_name} ===')

        original_labels_df = pd.read_csv(self.original_label_path)
        new_labels = self.apply_labeling_rules(original_labels_df)

        self._print_class_balance(new_labels)

        return self._save_labels(new_labels)


class FilterPostLabeler(DatasetPostLabeler):
    def __init__(
        self,
        keep_labels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if keep_labels is None:
            self.keep_labels = None
        elif isinstance(keep_labels, list):
            self.keep_labels = keep_labels if keep_labels else None
        else:
            self.keep_labels = [keep_labels]

    def apply_labeling_rules(self, labels_df):
        # Keep only specified labels if provided
        if self.keep_labels is None:
            return labels_df.copy()
        filtered_df = labels_df[labels_df['outcome'].isin(self.keep_labels)].copy()
        return filtered_df


class BinaryDatasetPostLabeler(FilterPostLabeler):
    def __init__(
        self,
        positive_label,
        keep_labels=None,
        label_folder='datasets/labels/binary',
        **kwargs,
    ):
        super().__init__(keep_labels=keep_labels, label_folder=label_folder, **kwargs)
        self.positive_label = positive_label

    def apply_labeling_rules(self, labels_df):
        filtered_df = super().apply_labeling_rules(labels_df)

        filtered_df['outcome'] = filtered_df['outcome'].apply(
            lambda x: 1 if x == self.positive_label else 0
        )
        return filtered_df


class BPIC12PostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Accepted',
            # keep_labels=['Accepted', 'Cancelled', 'Declined'],
            **kwargs,
        )


class BPIC13IPostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Resolved',
            # keep_labels=['Cancelled', 'In_Call', 'Other', 'Resolved'],
            **kwargs,
        )


class BPIC17PostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Accepted',
            # keep_labels=['Accepted', 'Cancelled', 'Other', 'Refused'],
            **kwargs,
        )


class BPIC20DDPostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Payment Handled',
            # keep_labels=['Other', 'Payment Handled', 'Rejected'],
            **kwargs,
        )


class BPIC20PTCPostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Payment Handled',
            # keep_labels=['Other', 'Payment Handled', 'Rejected'],
            **kwargs,
        )


class BPIC20RFPPostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Payment Handled',
            # keep_labels=['Other', 'Payment Handled', 'Rejected'],
            **kwargs,
        )


class HelpdeskPostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Closed',
            # keep_labels=['Closed', 'Other'],
            **kwargs,
        )


class HospitalBillingPostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Closed',
            # keep_labels=['Closed', 'Not Closed', 'Reopened'],
            **kwargs,
        )


class SepsisPostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Release A',
            # keep_labels=['Deviant Release', 'No Release', 'Release A'],
            **kwargs,
        )


class TrafficFinesPostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Repaid',
            # keep_labels=['Credit Collection', 'Incomplete', 'Repaid'],
            **kwargs,
        )


class BPIC15PostLabeler(BinaryDatasetPostLabeler):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            positive_label='Violation',
            keep_labels=['Violation', 'No violation'],
            **kwargs,
        )


if __name__ == '__main__':
    labelers = [
        ('BPIC_12', BPIC12PostLabeler),
        ('BPIC_13_I', BPIC13IPostLabeler),
        ('BPIC_17', BPIC17PostLabeler),
        ('BPIC_20_DD', BPIC20DDPostLabeler),
        ('BPIC_20_PTC', BPIC20PTCPostLabeler),
        ('BPIC_20_RFP', BPIC20RFPPostLabeler),
        ('Helpdesk', HelpdeskPostLabeler),
        ('Hospital_Billing', HospitalBillingPostLabeler),
        ('Sepsis', SepsisPostLabeler),
        ('Traffic_Fines', TrafficFinesPostLabeler),
    ]

    for i in range(1, 6):
        labelers.append((f'BPIC_15_{i}', BPIC15PostLabeler))
        labelers.append((f'Env_Permits_{i}', BPIC15PostLabeler))

    for dataset_name, labeler_class in labelers:
        try:
            labeler_class(dataset_name).run()
        except Exception as e:
            print(f'Failed to process {dataset_name}: {e}\n')
