from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils.bucketers import NoBucketer, PrefixLengthBucketer
from utils.log_datasets import OutcomeDataset
from utils.transformers import (
    AggregateTransformer,
    IndexBasedTransformer,
    LastStateTransformer,
)


def create_dataset(config):
    """Create a dataset instance from config."""
    dataset_type = config.get('type', 'outcome')

    if dataset_type == 'outcome':
        return OutcomeDataset(
            dataset_name=config['dataset_name'],
            dataset_folder=config['dataset_folder'],
            labels_folder=config['labels_folder'],
            label_filename=config.get('label_filename', None),
            train_ratio=config.get('train_ratio', 0.8),
            min_prefix=config.get('min_prefix', 3),
            max_prefix=config.get('max_prefix', None),
            case_id_col=config.get('case_id_col', 'case:concept:name'),
            time_col=config.get('time_col', 'time:timestamp'),
            activity_col=config.get('activity_col', 'concept:name'),
            resource_col=config.get('resource_col', 'org:resource'),
        )
    elif dataset_type == 'next_activity':
        # Placeholder for future implementation
        raise NotImplementedError(
            'Next activity prediction dataset not implemented yet'
        )
    elif dataset_type == 'remaining_time':
        # Placeholder for future implementation
        raise NotImplementedError(
            'Remaining time prediction dataset not implemented yet'
        )
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')


def create_bucketer(config):
    """Create a bucketer instance from config."""
    bucketer_type = config.get('type', 'prefix_length')
    case_id_col = config.get('case_id_col', 'prefix_id')

    if bucketer_type == 'none':
        return NoBucketer(case_id_col=case_id_col)
    elif bucketer_type == 'prefix_length':
        return PrefixLengthBucketer(case_id_col=case_id_col)
    elif bucketer_type == 'cluster_based':
        # Placeholder for future implementation
        raise NotImplementedError(
            'ClusterBasedBucketer not implemented yet. '
            'Will cluster cases based on feature similarity.'
        )
    elif bucketer_type == 'state_based':
        # Placeholder for future implementation
        raise NotImplementedError(
            'StateBasedBucketer not implemented yet. '
            'Will bucket cases based on their current state/activity sequence.'
        )
    else:
        raise ValueError(f'Unknown bucketer type: {bucketer_type}')


def create_transformer(config):
    """Create a transformer instance from config."""
    transformer_type = config['type']

    if transformer_type == 'aggregate':
        return AggregateTransformer(
            case_id_col=config.get('case_id_col', 'prefix_id'),
            cat_cols=config.get('cat_cols', []),
            num_cols=config.get('num_cols', []),
            boolean=config.get('boolean', False),
            fillna=config.get('fillna', True),
        )
    elif transformer_type == 'last_state':
        return LastStateTransformer(
            case_id_col=config.get('case_id_col', 'prefix_id'),
            cat_cols=config.get('cat_cols', []),
            num_cols=config.get('num_cols', []),
        )
    elif transformer_type == 'index_based':
        return IndexBasedTransformer(
            case_id_col=config.get('case_id_col', 'prefix_id'),
            cat_cols=config.get('cat_cols', []),
            num_cols=config.get('num_cols', []),
            max_len=config.get('max_len'),
        )
    else:
        raise ValueError(f'Unknown transformer type: {transformer_type}')


def create_model(config, num_classes=2):
    """Create a model instance from config."""
    model_type = config['type']
    params = config.get('params', {})

    if model_type == 'logistic_regression':
        return LogisticRegression(**params)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**params)
    elif model_type == 'xgboost':
        if num_classes == 2:
            default_params = {'objective': 'binary:logistic'}
        else:
            default_params = {'objective': 'multi:softprob', 'num_class': num_classes}
        return XGBClassifier(**{**default_params, **params})
    elif model_type == 'svm':
        return SVC(**params)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
