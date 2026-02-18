from abc import abstractmethod

import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter

from utils.stats import print_class_balance


class DatasetLabeler:
    def __init__(
        self,
        dataset_name,
        dataset_folder='datasets/raw',
        label_folder='datasets/labels',
        label_filename=None,
    ):
        self.dataset_name = dataset_name
        self.dataset_path = f'{dataset_folder}/{dataset_name}.xes'
        self.label_filename = label_filename if label_filename else dataset_name
        self.label_path = f'{label_folder}/{self.label_filename}.csv'

    def run(self):
        print(f'\n=== Labeling {self.dataset_name} ===')

        log_df = pm4py.read_xes(self.dataset_path)
        log = log_converter.apply(log_df)
        labels = self.apply_labeling_rules(log)

        labels_df = pd.DataFrame(labels)
        print_class_balance(labels_df, dataset_names=[self.dataset_name, 'Labeled'])

        return self._save_labels(labels_df)

    @abstractmethod
    def apply_labeling_rules(self, log):
        """
        Returns a list of dictionaries with case_id and outcome, e.g.:
        [{'case_id': '1', 'outcome': 0}, ...]
        """
        pass

    def _save_labels(self, labels_df):
        labels_df.to_csv(self.label_path, index=False)
        print(f'Successfully saved {len(labels_df)} labels to {self.label_path}\n')
        return labels_df


class BPIC12ALabeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []

        # Keywords with lifecycle transitions (matching Weytjens methodology)
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

        # Priority: declined > approved > canceled
        priority_order = ['declined', 'approved', 'canceled']
        outcome_labels = {
            'declined': 'Declined',
            'approved': 'Approved',
            'canceled': 'Cancelled',
        }

        for trace in log:
            case_id = trace.attributes.get('concept:name')

            # Create activity+lifecycle column
            activities = set()
            for ev in trace:
                concept = ev.get('concept:name', '')
                lifecycle = ev.get('lifecycle:transition', '')
                if lifecycle:
                    activities.add(f'{concept}_{lifecycle}')
                else:
                    activities.add(concept)

            # Determine outcome by priority
            outcome = 'Other'
            for outcome_type in priority_order:
                if any(kw in activities for kw in keywords_dict.get(outcome_type, [])):
                    outcome = outcome_labels.get(
                        outcome_type, outcome_type.capitalize()
                    )
                    break

            labels.append({'case_id': case_id, 'outcome': outcome})

        return labels


class BPIC12OLabeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []

        # Keywords with lifecycle transitions (matching Weytjens methodology)
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

        # Priority: canceled > approved > declined
        priority_order = ['canceled', 'approved', 'declined']
        outcome_labels = {
            'canceled': 'Cancelled',
            'approved': 'Accepted',
            'declined': 'Declined',
        }

        for trace in log:
            case_id = trace.attributes.get('concept:name')

            # Create activity+lifecycle column
            activities = set()
            for ev in trace:
                concept = ev.get('concept:name', '')
                lifecycle = ev.get('lifecycle:transition', '')
                if lifecycle:
                    activities.add(f'{concept}_{lifecycle}')
                else:
                    activities.add(concept)

            # Determine outcome by priority
            outcome = 'Other'
            for outcome_type in priority_order:
                if any(kw in activities for kw in keywords_dict.get(outcome_type, [])):
                    outcome = outcome_labels.get(
                        outcome_type, outcome_type.capitalize()
                    )
                    break

            labels.append({'case_id': case_id, 'outcome': outcome})

        return labels


class BPIC12WLabeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []

        keywords_dict = {
            'approved': [
                'W_Completeren aanvraag_COMPLETE',  # Complete application
                'W_Valideren aanvraag_COMPLETE',  # Validate application
            ],
            'declined': [],
            'canceled': [
                'W_Nabellen offertes_COMPLETE',  # Call about offers (follow-up)
            ],
        }

        # Priority: declined > approved > canceled (same as A variant)
        priority_order = ['declined', 'approved', 'canceled']
        outcome_labels = {
            'declined': 'Declined',
            'approved': 'Approved',
            'canceled': 'Cancelled',
        }

        for trace in log:
            case_id = trace.attributes.get('concept:name')

            # Create activity+lifecycle column
            activities = set()
            for ev in trace:
                concept = ev.get('concept:name', '')
                lifecycle = ev.get('lifecycle:transition', '')
                if lifecycle:
                    activities.add(f'{concept}_{lifecycle}')
                else:
                    activities.add(concept)

            # Determine outcome by priority
            outcome = 'Other'
            for outcome_type in priority_order:
                if any(kw in activities for kw in keywords_dict.get(outcome_type, [])):
                    outcome = outcome_labels.get(
                        outcome_type, outcome_type.capitalize()
                    )
                    break

            labels.append({'case_id': case_id, 'outcome': outcome})

        return labels


class BPIC13ILabeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []

        for trace in log:
            case_id = trace.attributes.get('concept:name')

            # Combination of event and lifecycle
            full_events = []
            for ev in trace:
                concept = ev.get('concept:name', '')
                lifecycle = ev.get('lifecycle:transition', '')
                full_events.append(f'{concept}+{lifecycle}' if lifecycle else concept)

            last_event = full_events[-1] if full_events else ''

            if last_event in ['Completed+Resolved', 'Completed+Closed']:
                outcome = 'Resolved'
            elif last_event == 'Completed+In Call':
                outcome = 'In_Call'
            elif last_event == 'Completed+Cancelled':
                outcome = 'Cancelled'
            else:
                outcome = 'Other'

            labels.append({'case_id': case_id, 'outcome': outcome})
        return labels


class BPIC13CLabeler(DatasetLabeler):
    """All cases are closed, so no outcome distinction."""


class BPIC15Labeler(DatasetLabeler):
    """Can be also applied to Env_Permits"""

    def apply_labeling_rules(self, log):
        labels = []

        # LTL Logic: "Every time A happens, B must happen later"
        EVENT_A = '01_HOOFD_010'  # 'send confirmation receipt'
        EVENT_B = '01_HOOFD_011'  # 'retrieve missing data'

        for trace in log:
            case_id = trace.attributes.get('concept:name')

            events = [ev['concept:name'] for ev in trace]

            indices_a = [i for i, act in enumerate(events) if act == EVENT_A]
            outcome = 'No violation'

            for idx_a in indices_a:
                event_b = any(act == EVENT_B for act in events[idx_a + 1 :])
                if not event_b:
                    outcome = 'Violation'
                    break
            labels.append({'case_id': case_id, 'outcome': outcome})
        return labels


class BPIC17Labeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []

        # Keywords for classification targets (matching Weytjens methodology)
        keywords_dict = {
            'approved': ['O_Accepted'],
            'declined': ['O_Refused'],
            'canceled': ['O_Cancelled'],
        }

        # Priority: canceled > declined > approved
        priority_order = ['canceled', 'declined', 'approved']
        outcome_labels = {
            'canceled': 'Cancelled',
            'declined': 'Declined',
            'approved': 'Accepted',
        }

        for trace in log:
            case_id = trace.attributes.get('concept:name')

            # Get all activity names (no lifecycle for BPIC17)
            activities = {ev.get('concept:name', '') for ev in trace}

            # Determine outcome by priority
            outcome = 'Other'
            for outcome_type in priority_order:
                if any(kw in activities for kw in keywords_dict.get(outcome_type, [])):
                    outcome = outcome_labels.get(
                        outcome_type, outcome_type.capitalize()
                    )
                    break

            labels.append({'case_id': case_id, 'outcome': outcome})

        return labels


class BPIC19Labeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []

        for trace in log:
            case_id = trace.attributes.get('concept:name')
            activities = [ev.get('concept:name', '') for ev in trace]

            # Check if invoice was cleared (payment completed)
            has_invoice_cleared = any(
                'Clear Invoice' in act or 'Record Invoice Receipt' in act
                for act in activities
            )

            # Check if there are rejection/deletion activities
            has_rejection = any(
                'Delete' in act or 'Reject' in act for act in activities
            )

            if has_rejection:
                outcome = 'Rejected'
            elif has_invoice_cleared:
                outcome = 'Cleared'
            else:
                outcome = 'Incomplete'

            labels.append({'case_id': case_id, 'outcome': outcome})
        return labels


class BPIC20DDLabeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []
        for trace in log:
            case_id = trace.attributes.get('concept:name')

            final_event = trace[-1]['concept:name']

            if final_event == 'Payment Handled':
                outcome = 'Payment Handled'
            elif 'Declaration REJECTED' in final_event:
                outcome = 'Rejected'
            else:
                outcome = 'Other'

            labels.append({'case_id': case_id, 'outcome': outcome})
        return labels


class BPIC20IDLabeler(DatasetLabeler):
    """Straight forward process without outcome branching."""


class BPIC20PTCLabeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []
        for trace in log:
            case_id = trace.attributes.get('concept:name')

            final_event = trace[-1]['concept:name']

            if final_event == 'Payment Handled':
                outcome = 'Payment Handled'
            elif 'Request For Payment REJECTED' in final_event:
                outcome = 'Rejected'
            else:
                outcome = 'Other'

            labels.append({'case_id': case_id, 'outcome': outcome})
        return labels


class BPIC20RFPLabeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []
        for trace in log:
            case_id = trace.attributes.get('concept:name')

            final_event = trace[-1]['concept:name']

            if final_event == 'Payment Handled':
                outcome = 'Payment Handled'
            elif 'Request For Payment REJECTED' in final_event:
                outcome = 'Rejected'
            else:
                outcome = 'Other'

            labels.append({'case_id': case_id, 'outcome': outcome})
        return labels


class BPIC20TPDLabeler(DatasetLabeler):
    """Straight forward process without outcome branching."""


class HelpdeskLabeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []
        for trace in log:
            case_id = trace.attributes.get('concept:name')

            final_event = trace[-1]['concept:name']

            if final_event == 'Closed':
                outcome = 'Closed'
            else:
                outcome = 'Other'

            labels.append({'case_id': case_id, 'outcome': outcome})
        return labels


class HospitalBillingLabeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []

        for trace in log:
            case_id = trace.attributes.get('concept:name')
            activities = [ev.get('concept:name', '') for ev in trace]

            is_not_closed = 1 if 'FIN' not in activities else 0
            is_reopened = 1 if 'REOPEN' in activities else 0

            if is_not_closed:
                if is_reopened:
                    outcome = 'Not Closed & Reopened'
                else:
                    outcome = 'Not Closed'
            elif is_reopened:
                outcome = 'Reopened'
            else:
                outcome = 'Closed'

            labels.append({'case_id': case_id, 'outcome': outcome})
        return labels


class SepsisLabeler(DatasetLabeler):
    # def apply_labeling_rules(self, log):
    #     labels = []

    #     for trace in log:
    #         case_id = trace.attributes.get('concept:name')
    #         activities = [ev.get('concept:name', '') for ev in trace]

    #         # Patient admitted to intensive care
    #         is_icu_admitted = 'Admission IC' in activities

    #         labels.append({'case_id': case_id, 'outcome': int(is_icu_admitted)})
    #     return labels

    def apply_labeling_rules(self, log):
        labels = []

        for trace in log:
            case_id = trace.attributes.get('concept:name')
            activities = [ev.get('concept:name', '') for ev in trace]

            release_events = [a for a in activities if a.startswith('Release')]

            if release_events:
                last_release = release_events[-1]

                if last_release != 'Release A':
                    outcome = 'Deviant Release'
                else:
                    outcome = 'Release A'
            else:
                outcome = 'No Release'

            labels.append({'case_id': case_id, 'outcome': outcome})
        return labels


class TrafficFinesLabeler(DatasetLabeler):
    def apply_labeling_rules(self, log):
        labels = []

        for trace in log:
            case_id = trace.attributes.get('concept:name')
            activities = [ev.get('concept:name', '') for ev in trace]

            if 'Send for Credit Collection' in activities:
                outcome = 'Credit Collection'

            elif activities[-1] == 'Send Fine':
                outcome = 'Incomplete'

            else:
                outcome = 'Repaid'

            labels.append({'case_id': case_id, 'outcome': outcome})
        return labels


if __name__ == '__main__':
    labelers = [
        ('BPIC_12_A', BPIC12ALabeler),
        ('BPIC_12_O', BPIC12OLabeler),
        ('BPIC_12_W', BPIC12WLabeler),
        ('BPIC_13_I', BPIC13ILabeler),
        ('BPIC_17', BPIC17Labeler),
        ('BPIC_19', BPIC19Labeler),
        ('BPIC_20_DD', BPIC20DDLabeler),
        ('BPIC_20_PTC', BPIC20PTCLabeler),
        ('BPIC_20_RFP', BPIC20RFPLabeler),
        ('Helpdesk', HelpdeskLabeler),
        ('Hospital_Billing', HospitalBillingLabeler),
        ('Sepsis', SepsisLabeler),
        ('Traffic_Fines', TrafficFinesLabeler),
    ]

    for i in range(1, 6):
        labelers.append((f'BPIC_15_{i}', BPIC15Labeler))
        labelers.append((f'Env_Permits_{i}', BPIC15Labeler))

    for dataset_name, labeler_class in labelers:
        try:
            labeler_class(dataset_name).run()
        except Exception as e:
            print(f'Failed to process {dataset_name}: {e}\n')
