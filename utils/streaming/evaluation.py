import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    roll_kw = {'window': window, 'center': center_window, 'min_periods': 1}

    if metric == 'accuracy':
        correct = (stream_df['y_true'] == stream_df['y_pred']).astype(float)

        metric_values = correct.rolling(**roll_kw).mean()

        ylabel = 'Accuracy'
        line_label = f'Accuracy (window={window}, center={center_window})'

    elif metric == 'f1':
        unique_classes = np.union1d(
            stream_df['y_true'].unique(), stream_df['y_pred'].unique()
        )

        tp_cols, fp_cols, fn_cols = [], [], []
        for c in unique_classes:
            is_true_c = stream_df['y_true'] == c
            is_pred_c = stream_df['y_pred'] == c

            tp_cols.append((is_true_c & is_pred_c).astype(float))
            fp_cols.append((~is_true_c & is_pred_c).astype(float))
            fn_cols.append((is_true_c & ~is_pred_c).astype(float))

        tp_roll = pd.concat(tp_cols, axis=1).rolling(**roll_kw).sum().values
        fp_roll = pd.concat(fp_cols, axis=1).rolling(**roll_kw).sum().values
        fn_roll = pd.concat(fn_cols, axis=1).rolling(**roll_kw).sum().values

        nom = 2 * tp_roll
        denom = 2 * tp_roll + fp_roll + fn_roll

        f1_per_class = np.full_like(denom, np.nan)
        np.divide(nom, denom, out=f1_per_class, where=denom > 0)

        # Macro average: only over classes *present* in each window (denom > 0)
        with np.errstate(all='ignore'):
            metric_values = pd.Series(
                np.nanmean(f1_per_class, axis=1), index=stream_df.index
            )

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
