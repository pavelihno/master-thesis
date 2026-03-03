import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def plot_stream_metric(
    results_df: pd.DataFrame,
    metric: str = 'f1',
    window: int = 100,
    center_window: bool = False,
    figsize: tuple[int, int] = (12, 4),
    save_path: str | None = None,
    actual_drift_points: list[int] | None = None,
) -> plt.Axes:
    """
    Plot a prequential metric over stream progress with concept drift markers.
    """
    stream_df = (
        results_df.sort_values('trace_n')
        .dropna(subset=['y_pred'])
        .reset_index(drop=True)
        .copy()
    )

    if metric == 'accuracy':
        correct = (stream_df['y_true'] == stream_df['y_pred']).astype(float)
        metric_values = correct.rolling(
            window=window, center=center_window, min_periods=1
        ).mean()
        ylabel = 'Accuracy'
        line_label = f'Accuracy (window={window}, center={center_window})'

    elif metric == 'f1':

        def _window_f1(positions: np.ndarray) -> float:
            sub = stream_df.iloc[positions.astype(int)]
            return f1_score(
                sub['y_true'], sub['y_pred'], average='macro', zero_division=0
            )

        # Roll over a numeric positional index
        pos = pd.Series(np.arange(len(stream_df)))
        roll = pos.rolling(window=window, center=center_window, min_periods=1)
        metric_values = roll.apply(_window_f1, raw=True)
        ylabel = 'Macro F1-score'
        line_label = f'Macro F1 (window={window}, center={center_window})'

    else:
        raise ValueError(f'Unknown {metric}')

    # Rows where the drift counter first increments
    detected_drift_points = (
        results_df[results_df['n_drifts'].diff() > 0]
        .drop_duplicates('n_drifts')['trace_n']
        .values
    )

    _, ax = plt.subplots(figsize=figsize)

    ax.plot(
        stream_df['trace_n'],
        metric_values,
        linewidth=1.5,
        color='steelblue',
        label=line_label,
    )
    for i, dt in enumerate(detected_drift_points):
        ax.axvline(
            dt,
            color='red',
            linewidth=0.8,
            linestyle='--',
            alpha=0.7,
            label='Detected drift' if i == 0 else None,
        )

    if actual_drift_points:
        for i, dt in enumerate(actual_drift_points):
            ax.axvline(
                dt,
                color='green',
                linewidth=1.2,
                linestyle='-',
                alpha=0.8,
                label='Actual drift' if i == 0 else None,
            )

    ax.set_xlabel('Stream Progress (Trace #)')
    ax.set_ylabel(ylabel)
    ax.set_title(line_label)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return ax
