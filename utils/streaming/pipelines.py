import time
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
from pybeamline.sources import xes_log_source_from_file
from river.base.estimator import Estimator

from utils.streaming.maps import (
    LearnerMap,
    NextActivityEmitterMap,
    PredictorMap,
)
from utils.streaming.sinks import ClassificationEvaluatorSink
from utils.streaming.transformers import StreamingTransformer


class PipelineMode(Enum):
    PREDICT_AND_LEARN = 'predict_and_learn'
    PREDICT_ONLY = 'predict_only'


class TaskPipeline(ABC):
    def __init__(
        self,
        model: Estimator,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
        mode: PipelineMode = PipelineMode.PREDICT_AND_LEARN,
    ):
        self.model = model
        self.transformer = transformer
        self.end_events = end_events
        self.mode = mode

    @abstractmethod
    def run(self, dataset_path: str) -> tuple[pd.DataFrame, object, float]:
        pass


class NextActivityPredictionPipeline(TaskPipeline):
    def __init__(
        self,
        model,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
        mode: PipelineMode = PipelineMode.PREDICT_AND_LEARN,
    ):
        super().__init__(model, transformer, end_events, mode)

    def run(self, dataset_path: str) -> tuple[pd.DataFrame, object, float]:
        sink = ClassificationEvaluatorSink()

        start_time = time.perf_counter()

        emitter = NextActivityEmitterMap(self.transformer, end_events=self.end_events)
        predictor = PredictorMap(self.model)

        if self.mode == PipelineMode.PREDICT_AND_LEARN:
            xes_log_source_from_file(dataset_path).pipe(
                emitter,
                predictor,
                LearnerMap(self.model),
            ).sink(sink)
        elif self.mode == PipelineMode.PREDICT_ONLY:
            xes_log_source_from_file(dataset_path).pipe(
                emitter,
                predictor,
            ).sink(sink)
        else:
            raise ValueError(f'Unknown pipeline mode: {self.mode}')

        elapsed_time = time.perf_counter() - start_time

        df = sink.to_dataframe()

        return df, self.model, elapsed_time
