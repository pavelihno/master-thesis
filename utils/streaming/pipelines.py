import time
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
from pybeamline.mappers.print_operator import print_operator
from pybeamline.sources import string_test_source, xes_log_source_from_file
from river.base.estimator import Estimator

from utils.streaming.maps import (
    EmitterMap,
    EmptyOperator,
    LearnerMap,
    NextActivityEmitterMap,
    PredictorMap,
)
from utils.streaming.sinks import ClassificationEvaluatorSink, EvaluatorSink
from utils.streaming.transformers import StreamingTransformer


class PipelineMode(Enum):
    PREDICT_AND_LEARN = 'predict_and_learn'
    PREDICT_ONLY = 'predict_only'


class SourceMode(Enum):
    LOG = 'log'
    STRING = 'string'


class Source(ABC):
    @abstractmethod
    def get_source(self, **kwargs):
        pass


class LogSource(Source):
    def get_source(self, **kwargs):
        dataset_path = kwargs.get('dataset_path')
        if dataset_path is None:
            raise ValueError('dataset_path must be provided for LogSource')
        return xes_log_source_from_file(dataset_path)


class StringSource(Source):
    def get_source(self, **kwargs):
        string = kwargs.get('string')
        if string is None:
            raise ValueError('string must be provided for StringSource')
        return string_test_source([string])


class TaskPipeline(ABC):
    def __init__(
        self,
        model: Estimator,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
        mode: PipelineMode = PipelineMode.PREDICT_AND_LEARN,
        source_mode: SourceMode = SourceMode.LOG,
    ):
        self.model = model
        self.transformer = transformer
        self.end_events = end_events
        self.mode = mode
        self.source_mode = source_mode

    @abstractmethod
    def get_emitter(self) -> EmitterMap:
        pass

    def get_predictor(self) -> PredictorMap:
        return PredictorMap(self.model)

    def get_learner(self) -> LearnerMap:
        return LearnerMap(self.model)

    @abstractmethod
    def get_sink(self) -> EvaluatorSink:
        pass

    def run(self, debug=False, **kwargs) -> tuple[pd.DataFrame, object, float]:
        sink = self.get_sink()

        emitter = self.get_emitter()
        predictor = self.get_predictor()
        learner = self.get_learner()

        if self.source_mode == SourceMode.LOG:
            source_obj = LogSource()
        elif self.source_mode == SourceMode.STRING:
            source_obj = StringSource()
        else:
            raise ValueError(f'Unknown source mode: {self.source_mode}')

        source = source_obj.get_source(**kwargs)

        start_time = time.perf_counter()

        if self.mode == PipelineMode.PREDICT_AND_LEARN:
            source.pipe(
                emitter,
                print_operator('EMIT> {0}') if debug else EmptyOperator(),
                predictor,
                print_operator('PREDICT> {0}') if debug else EmptyOperator(),
                learner,
            ).sink(sink)

        elif self.mode == PipelineMode.PREDICT_ONLY:
            source.pipe(
                emitter,
                print_operator('EMIT> {0}') if debug else EmptyOperator(),
                predictor,
                print_operator('PREDICT> {0}') if debug else EmptyOperator(),
            ).sink(sink)

        else:
            raise ValueError(f'Unknown pipeline mode: {self.mode}')

        elapsed_time = time.perf_counter() - start_time

        df = sink.to_dataframe()

        return df, self.model, elapsed_time


class NextActivityPredictionPipeline(TaskPipeline):
    def __init__(
        self,
        model,
        transformer: StreamingTransformer,
        end_events: set[str] | None = None,
        mode: PipelineMode = PipelineMode.PREDICT_AND_LEARN,
        source_mode: SourceMode = SourceMode.LOG,
    ):
        super().__init__(model, transformer, end_events, mode, source_mode=source_mode)

    def get_emitter(self):
        return NextActivityEmitterMap(self.transformer, end_events=self.end_events)

    def get_sink(self):
        return ClassificationEvaluatorSink()
