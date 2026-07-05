import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum

import pandas as pd
from pybeamline.mappers.print_operator import print_operator
from river.base.estimator import Estimator

from utils.streaming.base.bucketers import Bucketer
from utils.streaming.base.emitters import (
    EmitterMap,
    NextActivityEmitter,
    NextActivityTimeEmitter,
    OutcomeEmitter,
    RemainingTimeEmitter,
)
from utils.streaming.base.extractors import OutcomeExtractor
from utils.streaming.base.learners import (
    LearnerMap,
    NextActivityLearner,
    NextActivityTimeLearner,
    OutcomeLearner,
    RemainingTimeLearner,
)
from utils.streaming.base.maps import EmptyMap
from utils.streaming.base.predictors import PredictorMap
from utils.streaming.base.sinks import (
    EvaluatorSink,
    NextActivityEvaluator,
    NextActivityTimeEvaluator,
    OutcomeEvaluator,
    RemainingTimeEvaluator,
)
from utils.streaming.base.terminators import TraceTerminator
from utils.streaming.base.transformers import StreamingTransformer
from utils.streaming.pybeamline import (
    xes_log_source_from_bevents,
    xes_log_source_from_dataframe,
    xes_log_source_from_file,
    xes_log_source_from_string,
)
from utils.streaming.time import TimeTarget


class PipelineMode(Enum):
    PREDICT_AND_LEARN = 'predict_and_learn'
    PREDICT_ONLY = 'predict_only'


class SourceMode(Enum):
    LOG = 'log'
    DATA_FRAME = 'data_frame'
    STRING = 'string'
    BEVENTS = 'bevents'


class Source(ABC):
    @abstractmethod
    def get_source(self, max_events: int | None = None, **kwargs):
        pass


class LogSource(Source):
    def get_source(
        self,
        max_events: int | None = None,
        inject_terminal: bool = False,
        case_id_col: str = 'case:concept:name',
        activity_col: str = 'concept:name',
        time_col: str = 'time:timestamp',
        **kwargs,
    ):
        dataset_path = kwargs.get('dataset_path')
        if dataset_path is None:
            raise ValueError('dataset_path must be provided for LogSource')

        return xes_log_source_from_file(
            dataset_path,
            max_events=max_events,
            inject_terminal=inject_terminal,
            case_id_col=case_id_col,
            activity_col=activity_col,
            time_col=time_col,
        )


class DataFrameSource(Source):
    def get_source(
        self,
        max_events: int | None = None,
        inject_terminal: bool = False,
        case_id_col: str = 'case:concept:name',
        activity_col: str = 'concept:name',
        time_col: str = 'time:timestamp',
        **kwargs,
    ):
        df = kwargs.get('data_frame')
        if df is None:
            raise ValueError('data_frame must be provided for DataFrameSource')

        return xes_log_source_from_dataframe(
            df,
            max_events=max_events,
            inject_terminal=inject_terminal,
            case_id_col=case_id_col,
            activity_col=activity_col,
            time_col=time_col,
        )


class StringSource(Source):
    def get_source(self, max_events: int | None = None, **kwargs):
        string = kwargs.get('string')
        if string is None:
            raise ValueError('string must be provided for StringSource')

        inject_terminal = kwargs.get('inject_terminal', False)

        return xes_log_source_from_string(
            trace_string=string,
            max_events=max_events,
            inject_terminal=inject_terminal,
        )


class BEventSource(Source):
    def get_source(self, max_events: int | None = None, **kwargs):
        bevents = kwargs.get('bevents')
        if bevents is None:
            raise ValueError('bevents must be provided for BEventsSource')
        if not isinstance(bevents, Iterable):
            raise TypeError('bevents must be an iterable of BEvent')

        inject_terminal = kwargs.get('inject_terminal', False)

        return xes_log_source_from_bevents(
            bevents=bevents,
            max_events=max_events,
            inject_terminal=inject_terminal,
        )


class TaskPipeline(ABC):
    def __init__(
        self,
        model: Estimator | Bucketer,
        transformer: StreamingTransformer,
        terminator: TraceTerminator,
        pipeline_mode: PipelineMode = PipelineMode.PREDICT_AND_LEARN,
        source_mode: SourceMode = SourceMode.LOG,
        max_events: int | None = None,
    ):
        self.model = model
        self.transformer = transformer
        self.terminator = terminator
        self.pipeline_mode = pipeline_mode
        self.source_mode = source_mode
        self.max_events = max_events

    @abstractmethod
    def get_emitter(self) -> EmitterMap:
        pass

    @abstractmethod
    def get_predictor(self) -> PredictorMap:
        pass

    @abstractmethod
    def get_learner(self) -> LearnerMap:
        pass

    @abstractmethod
    def get_sink(self) -> EvaluatorSink:
        pass

    def run(self, debug=False, **kwargs) -> tuple[pd.DataFrame, object, float]:

        emitter = self.get_emitter()
        predictor = self.get_predictor()
        learner = self.get_learner()
        sink = self.get_sink()

        if self.source_mode == SourceMode.LOG:
            source_obj = LogSource()
        elif self.source_mode == SourceMode.DATA_FRAME:
            source_obj = DataFrameSource()
        elif self.source_mode == SourceMode.STRING:
            source_obj = StringSource()
        elif self.source_mode == SourceMode.BEVENTS:
            source_obj = BEventSource()
        else:
            raise ValueError(f'Unknown source mode: {self.source_mode}')

        source = source_obj.get_source(max_events=self.max_events, **kwargs)

        start_time = time.perf_counter()

        if self.pipeline_mode == PipelineMode.PREDICT_AND_LEARN:
            source.pipe(
                emitter,
                print_operator('EMIT> {0}') if debug else EmptyMap(),
                predictor,
                print_operator('PREDICT> {0}') if debug else EmptyMap(),
                learner,
            ).sink(sink)

        elif self.pipeline_mode == PipelineMode.PREDICT_ONLY:
            source.pipe(
                emitter,
                print_operator('EMIT> {0}') if debug else EmptyMap(),
                predictor,
                print_operator('PREDICT> {0}') if debug else EmptyMap(),
            ).sink(sink)

        else:
            raise ValueError(f'Unknown pipeline task mode: {self.pipeline_mode}')

        elapsed_time = time.perf_counter() - start_time

        df = sink.to_dataframe()

        return df, self.model, elapsed_time


class NextActivityPredictionPipeline(TaskPipeline):
    def __init__(
        self,
        model: Estimator | Bucketer,
        transformer: StreamingTransformer,
        terminator: TraceTerminator,
        pipeline_mode: PipelineMode = PipelineMode.PREDICT_AND_LEARN,
        source_mode: SourceMode = SourceMode.LOG,
        max_events: int | None = None,
    ):
        super().__init__(
            model,
            transformer,
            terminator,
            pipeline_mode,
            source_mode=source_mode,
            max_events=max_events,
        )

    def get_emitter(self):
        return NextActivityEmitter(self.transformer, terminator=self.terminator)

    def get_predictor(self):
        return PredictorMap(self.model)

    def get_learner(self):
        return NextActivityLearner(self.model)

    def get_sink(self):
        return NextActivityEvaluator()


class OutcomePredictionPipeline(TaskPipeline):
    def __init__(
        self,
        model: Estimator | Bucketer,
        transformer: StreamingTransformer,
        terminator: TraceTerminator,
        pipeline_mode: PipelineMode = PipelineMode.PREDICT_AND_LEARN,
        source_mode: SourceMode = SourceMode.LOG,
        outcome_extractor: OutcomeExtractor | None = None,
        max_events: int | None = None,
    ):
        super().__init__(
            model,
            transformer,
            terminator,
            pipeline_mode,
            source_mode=source_mode,
            max_events=max_events,
        )
        self.outcome_extractor = outcome_extractor

    def get_emitter(self):
        return OutcomeEmitter(
            self.transformer,
            terminator=self.terminator,
            outcome_extractor=self.outcome_extractor,
        )

    def get_predictor(self):
        return PredictorMap(self.model)

    def get_learner(self):
        return OutcomeLearner(self.model)

    def get_sink(self):
        return OutcomeEvaluator()


class RemainingTimePredictionPipeline(TaskPipeline):
    def __init__(
        self,
        model: Estimator | Bucketer,
        transformer: StreamingTransformer,
        terminator: TraceTerminator,
        pipeline_mode: PipelineMode = PipelineMode.PREDICT_AND_LEARN,
        source_mode: SourceMode = SourceMode.LOG,
        target: TimeTarget = TimeTarget.SECONDS,
        max_events: int | None = None,
    ):
        super().__init__(
            model,
            transformer,
            terminator,
            pipeline_mode,
            source_mode=source_mode,
            max_events=max_events,
        )
        self.target = target

    def get_emitter(self):
        return RemainingTimeEmitter(self.transformer, terminator=self.terminator)

    def get_predictor(self):
        return PredictorMap(self.model)

    def get_learner(self):
        return RemainingTimeLearner(self.model, target=self.target)

    def get_sink(self):
        return RemainingTimeEvaluator(target=self.target)


class NextActivityTimePredictionPipeline(TaskPipeline):
    def __init__(
        self,
        model: Estimator | Bucketer,
        transformer: StreamingTransformer,
        terminator: TraceTerminator,
        pipeline_mode: PipelineMode = PipelineMode.PREDICT_AND_LEARN,
        source_mode: SourceMode = SourceMode.LOG,
        target: TimeTarget = TimeTarget.SECONDS,
        max_events: int | None = None,
    ):
        super().__init__(
            model,
            transformer,
            terminator,
            pipeline_mode,
            source_mode=source_mode,
            max_events=max_events,
        )
        self.target = target

    def get_emitter(self):
        return NextActivityTimeEmitter(self.transformer, terminator=self.terminator)

    def get_predictor(self):
        return PredictorMap(self.model)

    def get_learner(self):
        return NextActivityTimeLearner(self.model, target=self.target)

    def get_sink(self):
        return NextActivityTimeEvaluator(target=self.target)
