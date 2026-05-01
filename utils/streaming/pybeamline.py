from collections.abc import Iterator

import pandas as pd
from pm4py import convert_to_dataframe, read_xes
from pybeamline.bevent import BEvent
from pybeamline.stream.stream import Stream


def xes_log_source_from_file(
    log: str,
    max_events: int | None = None,
    censored: bool = False,
    end_events: set[str] | None = None,
    case_id_col: str = 'case:concept:name',
    activity_col: str = 'concept:name',
    time_col: str = 'time:timestamp',
) -> Stream[BEvent]:
    return Stream.from_iterable(
        _iter_bevents(
            read_xes(log),
            max_events=max_events,
            censored=censored,
            end_events=end_events,
            case_id_col=case_id_col,
            time_col=time_col,
            activity_col=activity_col,
        )
    )


def _iter_bevents(
    raw_log: pd.DataFrame,
    max_events: int | None = None,
    censored: bool = False,
    end_events: set[str] | None = None,
    case_id_col: str = 'case:concept:name',
    activity_col: str = 'concept:name',
    time_col: str = 'time:timestamp',
) -> Iterator[BEvent]:
    log = raw_log if type(raw_log) is pd.DataFrame else convert_to_dataframe(raw_log)

    if time_col in log.columns:
        log = log.sort_values(by=[time_col])

    end_events = end_events if end_events is not None else set()
    if censored and end_events:
        terminal_events = (
            log.groupby(case_id_col, sort=False)[activity_col]
            .last()
            .rename('terminal_event')
        )
        allowed_traces = terminal_events[terminal_events.isin(end_events)].index
        log = log[log[case_id_col].isin(allowed_traces)]

    emitted = 0
    for _, event in log.iterrows():
        if max_events is not None and emitted >= max_events:
            break

        time = event[time_col] if time_col in event else None
        bevent = BEvent(event[activity_col], event[case_id_col], 'log-file', time)

        for col in log.columns:
            if col in {activity_col, case_id_col, time_col}:
                continue

            if pd.notna(event[col]):
                if col.startswith('case:') and col != case_id_col:
                    bevent.trace_attributes[col[5:]] = event[col]
                else:
                    bevent.event_attributes[col] = event[col]

        yield bevent
        emitted += 1
