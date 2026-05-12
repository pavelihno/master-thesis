from collections.abc import Iterator

import pandas as pd
from pm4py import convert_to_dataframe, read_xes
from pybeamline.bevent import BEvent
from pybeamline.sources import string_test_source
from pybeamline.stream.stream import Stream


def xes_log_source_from_file(
    log: str,
    max_events: int | None = None,
    inject_terminal: bool = False,
    case_id_col: str = 'case:concept:name',
    activity_col: str = 'concept:name',
    time_col: str = 'time:timestamp',
) -> Stream[BEvent]:
    return Stream.from_iterable(
        _iter_bevents(
            read_xes(log),
            max_events=max_events,
            inject_terminal=inject_terminal,
            case_id_col=case_id_col,
            time_col=time_col,
            activity_col=activity_col,
        )
    )


def xes_log_source_from_dataframe(
    data_frame: pd.DataFrame,
    max_events: int | None = None,
    inject_terminal: bool = False,
    case_id_col: str = 'case:concept:name',
    activity_col: str = 'concept:name',
    time_col: str = 'time:timestamp',
) -> Stream[BEvent]:
    return Stream.from_iterable(
        _iter_bevents(
            data_frame,
            max_events=max_events,
            inject_terminal=inject_terminal,
            case_id_col=case_id_col,
            activity_col=activity_col,
            time_col=time_col,
        )
    )


def _iter_bevents(
    raw_log: pd.DataFrame,
    max_events: int | None = None,
    inject_terminal: bool = False,
    case_id_col: str = 'case:concept:name',
    activity_col: str = 'concept:name',
    time_col: str = 'time:timestamp',
) -> Iterator[BEvent]:
    log = raw_log if type(raw_log) is pd.DataFrame else convert_to_dataframe(raw_log)

    log = log.sort_values(by=[time_col])

    # Mark only the last event of each trace with is_terminal=True
    if inject_terminal:
        last_event_indices = log.groupby(case_id_col).tail(1).index
        log['is_terminal'] = log.index.isin(last_event_indices)

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


def xes_log_source_from_string(
    trace_string: str,
    max_events: int | None = None,
    inject_terminal: bool = False,
) -> Stream[BEvent]:
    """Create a stream from a string trace (e.g., 'ABC', 'ABCD')."""
    source = string_test_source([trace_string])

    if inject_terminal:
        bevents = list(source)
        bevents = _inject_terminal_to_last_of_traces(bevents)
        source = Stream.from_iterable(bevents)

    if max_events is not None:
        bevents = list(source)[:max_events]
        source = Stream.from_iterable(bevents)

    return source


def xes_log_source_from_bevents(
    bevents: list[BEvent],
    max_events: int | None = None,
    inject_terminal: bool = False,
) -> Stream[BEvent]:
    """Create a stream from a list of BEvent objects."""
    bevents = list(bevents)

    if max_events is not None:
        bevents = bevents[:max_events]

    if inject_terminal:
        bevents = _inject_terminal_to_last_of_traces(bevents)

    return Stream.from_iterable(bevents)


def _inject_terminal_to_last_of_traces(bevents: list[BEvent]) -> list[BEvent]:
    """Mark only the last event of each trace with is_terminal=True."""
    if not bevents:
        return bevents

    # Group events by trace_name to find last event of each trace
    traces = {}
    for event in bevents:
        trace_name = event.get_trace_name()
        if trace_name not in traces:
            traces[trace_name] = []
        traces[trace_name].append(event)

    # Mark last event of each trace
    for trace_events in traces.values():
        if trace_events:
            trace_events[-1].event_attributes['is_terminal'] = True

    return bevents
