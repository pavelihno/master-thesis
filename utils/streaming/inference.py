import random

import pandas as pd
from pybeamline.sources import xes_log_source_from_file

from utils.streaming.sinks import TraceCollectorSink
from utils.streaming.transformers import StreamingTransformer


def load_traces(dataset_path: str) -> dict[str, list]:
    """Read an XES file and return a dict mapping trace_id -> list of BEvents."""
    sink = TraceCollectorSink()
    xes_log_source_from_file(dataset_path).sink(sink)
    return sink.to_dict()


def infer_trace(
    model,
    transformer: StreamingTransformer,
    dataset_path: str,
    trace_id: str | None = None,
    seed: int | None = None,
):
    """Run next-activity inference on a single trace from an XES dataset."""
    traces = load_traces(dataset_path)

    if trace_id is None:
        rng = random.Random(seed)
        trace_id = rng.choice(list(traces.keys()))

    if trace_id not in traces:
        raise KeyError(f"trace_id '{trace_id}' not found in {dataset_path}")

    events = traces[trace_id]
    buf = []
    rows = []

    for i, event in enumerate(events):
        y_true = event.get_event_name()

        if len(buf) == 0:
            y_pred = None
            y_pred_prob: dict = {}
        else:
            features = transformer.transform(buf)
            y_pred = model.predict_one(features)
            y_pred_prob = (
                model.predict_proba_one(features)
                if hasattr(model, 'predict_proba_one')
                else {}
            )

        rows.append(
            {
                'step': i,
                'prefix_len': len(buf),
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob.get(y_pred) if y_pred_prob else None,
                'correct': (y_pred == y_true) if y_pred is not None else None,
            }
        )

        buf.append(event)

    return pd.DataFrame(rows), trace_id
