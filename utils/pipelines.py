import numpy as np
import pandas as pd
from sklearn.base import clone


class ProcessPredictorPipeline:
    """Pipeline for bucketing, transforming, and predicting on process data."""

    def __init__(self, bucketer, transformer, estimator):
        self.bucketer = bucketer
        self.transformer = transformer
        self.estimator = estimator

        self.bucket_models = {}
        self.bucket_transformers = {}
        self.trained_buckets = []
        self.n_classes = None

    def _aggregate_by_prefix(self, prefixes_df, labels, prefix_col):
        """Helper to aggregate labels by prefix ID."""
        labels.index = prefixes_df.index
        aggregated = (
            pd.DataFrame(
                {
                    'prefix_id': prefixes_df[prefix_col],
                    'label': labels,
                }
            )
            .groupby('prefix_id')['label']
            .first()
        )
        return aggregated

    def fit(self, prefixes_df, labels):
        """Train separate models for each bucket."""
        # Determine number of classes
        self.n_classes = len(np.unique(labels))

        # Get bucket assignments
        bucket_ids_per_prefix = self.bucketer.fit_predict(prefixes_df)
        unique_bucket_ids = np.unique(bucket_ids_per_prefix)
        self.trained_buckets = unique_bucket_ids

        # Map prefix IDs to buckets
        unique_prefix_ids = prefixes_df[self.bucketer.case_id_col].unique()
        prefix_to_bucket = dict(
            zip(unique_prefix_ids, bucket_ids_per_prefix, strict=True)
        )
        bucket_ids = prefixes_df[self.bucketer.case_id_col].map(prefix_to_bucket).values

        # Train model for each bucket
        for bucket_id in unique_bucket_ids:
            print(f'Training bucket: {bucket_id}...')

            # Filter data for this bucket
            in_bucket = bucket_ids == bucket_id
            bucket_prefixes = prefixes_df[in_bucket]
            bucket_labels = labels[in_bucket]

            if len(bucket_labels) == 0:
                continue

            # Transform features
            bucket_transformer = clone(self.transformer)
            transformed_features = bucket_transformer.fit_transform(bucket_prefixes)

            # Aggregate labels by prefix ID
            aggregated_labels = self._aggregate_by_prefix(
                bucket_prefixes, bucket_labels, self.bucketer.case_id_col
            )

            # Train model
            bucket_model = clone(self.estimator)
            bucket_model.fit(transformed_features, aggregated_labels)

            # Store for prediction
            self.bucket_models[bucket_id] = bucket_model
            self.bucket_transformers[bucket_id] = bucket_transformer

    def predict(self, prefixes_df):
        """Generate predictions for each prefix."""
        # Get bucket assignments
        bucket_ids_per_prefix = self.bucketer.predict(prefixes_df)
        unique_prefix_ids = prefixes_df[self.bucketer.case_id_col].unique()
        prefix_to_bucket = dict(
            zip(unique_prefix_ids, bucket_ids_per_prefix, strict=True)
        )

        # Map bucket IDs to all rows
        bucket_ids = prefixes_df[self.bucketer.case_id_col].map(prefix_to_bucket).values

        # Collect predictions for each unique prefix
        predictions_dict = {}

        for bucket_id in self.trained_buckets:
            in_bucket = bucket_ids == bucket_id
            if not any(in_bucket) or bucket_id not in self.bucket_models:
                continue

            bucket_prefixes = prefixes_df[in_bucket]
            transformed_features = self.bucket_transformers[bucket_id].transform(
                bucket_prefixes
            )
            bucket_predictions = self.bucket_models[bucket_id].predict(
                transformed_features
            )

            # Map predictions to prefix IDs
            for prefix_id, pred in zip(
                bucket_prefixes[self.bucketer.case_id_col].unique(),
                bucket_predictions,
                strict=True,
            ):
                predictions_dict[prefix_id] = pred

        predictions = (
            prefixes_df[self.bucketer.case_id_col].map(predictions_dict).values
        )

        return predictions
