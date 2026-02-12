from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from utils.bucketers import (
    PrefixLengthBucketer,
)
from utils.log_datasets import OutcomeDataset
from utils.pipelines import ProcessPredictorPipeline
from utils.stats import print_class_balance
from utils.transformers import (
    AggregateTransformer,
)


dataset_name = 'Traffic_Fines'
dataset_folder = 'datasets/raw'
labels_folder = 'datasets/labels/binary'

dataset = OutcomeDataset(
    dataset_name=dataset_name,
    dataset_folder=dataset_folder,
    labels_folder=labels_folder,
    feature_names=[
        'time_since_start',
        'time_since_last_event',
        'event_index',
        'day_of_week',
        'hour_of_day',
        'hour_sin',
        'hour_cos',
    ],
    min_prefix=3,
    max_prefix=10,
)

# Load raw data
dataset.load_and_preprocess()

# Filter out cases without labels
dataset.filter_by_labels()

# Split into train/test
train_df, test_df = dataset.train_test_split()

# Generate prefixes
train_prefixes = dataset.get_prefixes(train_df)
test_prefixes = dataset.get_prefixes(test_df)

# Prepare labels
y_train = dataset.prepare_labels(train_prefixes)
y_test = dataset.prepare_labels(test_prefixes)

print_class_balance(y_train, dataset_names=[dataset_name, 'Train'])
print_class_balance(y_test, dataset_names=[dataset_name, 'Test'])

# Build the pipeline
bucketer = PrefixLengthBucketer(case_id_col='prefix_id')
transformer = AggregateTransformer(
    case_id_col='prefix_id',
    cat_cols=['concept:name'],
    num_cols=['time_since_last_event', 'time_since_start'],
)

num_classes = len(dataset.label_encoder.classes_)

if num_classes == 2:
    # Binary classification
    model = XGBClassifier(n_estimators=100, max_depth=5, objective='binary:logistic')
else:
    # Multiclass classification
    model = XGBClassifier(
        n_estimators=100, max_depth=5, objective='multi:softprob', num_class=num_classes
    )

# Build and Fit the Pipeline
pipeline = ProcessPredictorPipeline(bucketer, transformer, model)
pipeline.fit(train_prefixes, y_train)

# Evaluate
y_pred = pipeline.predict(test_prefixes)

# Decode predictions and labels for evaluation
y_pred_decoded = dataset.decode_labels(y_pred)
y_test_decoded = dataset.decode_labels(y_test)

print(classification_report(y_test_decoded, y_pred_decoded))
