import time
from abc import ABC, abstractmethod

import pandas as pd
from pybeamline.sources import xes_log_source_from_file
from river.base.estimator import Estimator

from utils.streaming.maps import LearnerMap, NextActivityEmitterMap, PredictorMap
from utils.streaming.sinks import ClassificationEvaluatorSink
from utils.streaming.transformers import StreamingTransformer


class TaskPipeline(ABC):
    def __init__(
        self,
        model: Estimator,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
    ):
        self.model = model
        self.transformer = transformer
        self.end_events = end_events

    @abstractmethod
    def run(self, dataset_path: str) -> tuple[pd.DataFrame, object]:
        pass


class NextActivityPredictionPipeline(TaskPipeline):
    def __init__(
        self,
        model,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
    ):
        super().__init__(model, transformer, end_events)

    def run(self, dataset_path: str) -> tuple[pd.DataFrame, object]:
        sink = ClassificationEvaluatorSink()

        start_time = time.perf_counter()

        xes_log_source_from_file(dataset_path).pipe(
            NextActivityEmitterMap(self.transformer, end_events=self.end_events),
            PredictorMap(self.model),
            LearnerMap(self.model),
        ).sink(sink)

        elapsed_time = time.perf_counter() - start_time

        df = sink.to_dataframe()
        df['time_s'] = elapsed_time

        return df, self.model
