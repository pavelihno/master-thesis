import pandas as pd
from pm4py import convert_to_dataframe, read_xes
from pm4py.objects.log.obj import EventLog
from pm4py.util import xes_constants as xes_util
from pybeamline.bevent import BEvent
from pybeamline.stream.base_source import BaseSource
from pybeamline.stream.stream import Stream


def xes_log_source_from_file(log: str, max_events: int | None = None) -> Stream[BEvent]:
    return Stream.source(XesLogSource(read_xes(log), max_events=max_events))


class XesLogSource(BaseSource[BEvent]):
    def __init__(self, raw_log: EventLog | pd.DataFrame, max_events: int | None = None):
        self.log = raw_log
        self.max_events = max_events
        if type(self.log) is not pd.DataFrame:
            self.log = convert_to_dataframe(self.log)
        if xes_util.DEFAULT_TIMESTAMP_KEY in self.log.columns:
            self.log = self.log.sort_values(by=[xes_util.DEFAULT_TIMESTAMP_KEY])

    def execute(self):
        emitted = 0
        for _, event in self.log.iterrows():
            if self.max_events is not None and emitted >= self.max_events:
                break

            time = None
            if xes_util.DEFAULT_TIMESTAMP_KEY in event:
                time = event[xes_util.DEFAULT_TIMESTAMP_KEY]
            e = BEvent(
                event[xes_util.DEFAULT_NAME_KEY],
                event['case:' + xes_util.DEFAULT_TRACEID_KEY],
                'log-file',
                time,
            )
            for col in self.log.columns:
                if col not in [
                    xes_util.DEFAULT_NAME_KEY,
                    'case:' + xes_util.DEFAULT_NAME_KEY,
                    xes_util.DEFAULT_TIMESTAMP_KEY,
                ]:
                    if event[col] == event[col]:  # verify for nan
                        if col.startswith('case:'):
                            e.trace_attributes[col[5:]] = event[col]
                        else:
                            e.event_attributes[col] = event[col]
            self.produce(e)
            emitted += 1
        self.completed()
