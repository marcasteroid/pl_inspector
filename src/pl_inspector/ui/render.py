"""Rendering helpers for the pl-inspector dashboard.

All functions in this module are **pure**: they accept Python data structures
(lists of dicts, NumPy arrays) and return either a :class:`pandas.DataFrame`
or a :class:`matplotlib.figure.Figure`.  No Gradio components are created here.

Empty / missing data is handled gracefully: DataFrame renderers return an
empty DataFrame with correct column names; plot renderers return a figure that
shows a short explanatory message instead of crashing.

Functions
---------

DataFrame builders
~~~~~~~~~~~~~~~~~~
- :func:`events_to_df`    — flat table of execution events
- :func:`gradients_to_df` — flat table of training-step records

Matplotlib figures
~~~~~~~~~~~~~~~~~~
- :func:`plot_timing_histogram`    — ``details.duration_ms`` histogram from events
- :func:`plot_loss_curve`          — loss over global steps
- :func:`plot_grad_norm_curve`     — gradient norm over global steps
- :func:`plot_metric_comparison`   — overlay a metric from two runs
- :func:`plot_snapshot_probs`      — probability bar chart from a snapshot array
- :func:`plot_snapshot_amplitudes` — top-k amplitude magnitudes from a snapshot array
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend; safe for Gradio

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

_C_PRIMARY = "#4C72B0"
_C_SECONDARY = "#DD8452"
_C_ACCENT = "#55A868"
_C_MUTED = "#888888"
_C_BG = "#F8F9FA"
_C_GRID = "#E0E0E0"

_FONT_FAMILY = "DejaVu Sans"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

JsonDict = Dict[str, Any]


def _empty_figure(message: str, *, figsize: Tuple[float, float] = (7, 3)) -> matplotlib.figure.Figure:
    """Return a Matplotlib figure displaying *message* in the centre.

    Used whenever input data is empty or insufficient to draw a real plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_C_BG)
    ax.set_facecolor(_C_BG)
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=12, color=_C_MUTED,
        wrap=True,
    )
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def _styled_ax(ax: plt.Axes, *, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply a consistent, clean style to *ax* in-place."""
    ax.set_facecolor(_C_BG)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, color=_C_GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _safe_floats(records: List[JsonDict], key: str) -> List[float]:
    """Extract numeric values for *key* from *records*, replacing missing/non-numeric with NaN."""
    out: List[float] = []
    for r in records:
        v = r.get(key)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            out.append(float(v))
        else:
            out.append(float("nan"))
    return out


def _safe_ints(records: List[JsonDict], key: str) -> List[int]:
    """Extract integer values for *key*, falling back to sequential indices."""
    return [
        int(r[key]) if isinstance(r.get(key), int) else i
        for i, r in enumerate(records)
    ]


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------


_EVENTS_COLUMNS = [
    "event_index", "timestamp", "category", "name",
    "duration_ms", "output_value", "device_name",
    "device_wires", "shots", "error_type", "error_message",
]

_GRADIENTS_COLUMNS = [
    "global_step", "epoch", "batch_index", "loss",
    "grad_norm", "timestamp",
]


def events_to_df(events: List[JsonDict]) -> pd.DataFrame:
    """Build a flat :class:`pandas.DataFrame` from a list of execution event records.

    Nested fields are promoted to top-level columns:

    - ``details.duration_ms`` → ``duration_ms``
    - ``details.output_summary.value`` → ``output_value``
    - ``details.device_name`` → ``device_name``

    Parameters
    ----------
    events:
        List of event dicts as produced by :func:`pl_inspector.ui.io.load_events`.

    Returns
    -------
    pandas.DataFrame
        One row per event; missing fields are ``NaN`` / empty string.
        Returns an empty DataFrame with canonical columns when *events* is empty.
    """
    if not events:
        return pd.DataFrame(columns=_EVENTS_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for ev in events:
        details: Dict[str, Any] = ev.get("details", {}) or {}
        out_summary = details.get("output_summary") or {}
        err = details.get("error") or {}
        rows.append({
            "event_index": ev.get("event_index"),
            "timestamp": ev.get("timestamp"),
            "category": ev.get("category"),
            "name": ev.get("name"),
            "duration_ms": details.get("duration_ms"),
            "output_value": out_summary.get("value"),
            "device_name": details.get("device_name"),
            "device_wires": details.get("device_wires"),
            "shots": details.get("shots"),
            "error_type": err.get("type"),
            "error_message": err.get("message"),
        })

    df = pd.DataFrame(rows)
    # Cast numeric columns where possible
    for col in ("event_index", "duration_ms", "output_value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def gradients_to_df(gradients: List[JsonDict]) -> pd.DataFrame:
    """Build a flat :class:`pandas.DataFrame` from training-step gradient records.

    Extra metric keys nested under ``"metrics"`` are expanded as additional
    columns (prefixed with ``"m_"`` to avoid accidental column-name collisions).

    Parameters
    ----------
    gradients:
        List of training-step dicts as produced by
        :func:`pl_inspector.ui.io.load_gradients`.

    Returns
    -------
    pandas.DataFrame
        One row per step.  Returns empty DataFrame with canonical columns when
        *gradients* is empty.
    """
    if not gradients:
        return pd.DataFrame(columns=_GRADIENTS_COLUMNS)

    # Collect all extra metric keys across all records
    extra_keys: list[str] = []
    for g in gradients:
        for k in (g.get("metrics") or {}).keys():
            col = f"m_{k}"
            if col not in extra_keys:
                extra_keys.append(col)

    rows: List[Dict[str, Any]] = []
    for g in gradients:
        metrics: Dict[str, Any] = g.get("metrics") or {}
        row: Dict[str, Any] = {
            "global_step": g.get("global_step"),
            "epoch": g.get("epoch"),
            "batch_index": g.get("batch_index"),
            "loss": g.get("loss"),
            "grad_norm": g.get("grad_norm"),
            "timestamp": g.get("timestamp"),
        }
        for k in extra_keys:
            bare = k[2:]  # strip "m_"
            row[k] = metrics.get(bare)
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ("global_step", "loss", "grad_norm", *extra_keys):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Execution timing histogram
# ---------------------------------------------------------------------------


def plot_timing_histogram(
    events: List[JsonDict],
    *,
    bins: int = 20,
    title: str = "Execution Timing Distribution",
    per_category: bool = True,
) -> matplotlib.figure.Figure:
    """Return a histogram of ``duration_ms`` values from execution events.

    When *per_category* is ``True`` and multiple categories exist, each
    category gets its own colour in an overlapping histogram.

    Parameters
    ----------
    events:
        List of event dicts (must contain ``details.duration_ms``).
    bins:
        Number of histogram bins.
    title:
        Figure title.
    per_category:
        If ``True``, colour-code bars by event category.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = events_to_df(events)
    durations = df["duration_ms"].dropna()

    if durations.empty:
        return _empty_figure("No timing data available in events.")

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_C_BG)

    if per_category and "category" in df.columns:
        categories = df["category"].dropna().unique().tolist()
        colours = [_C_PRIMARY, _C_SECONDARY, _C_ACCENT, "#9467BD", "#8C564B"]
        for i, cat in enumerate(categories):
            cat_dur = df.loc[df["category"] == cat, "duration_ms"].dropna()
            if not cat_dur.empty:
                ax.hist(
                    cat_dur, bins=bins, alpha=0.72,
                    color=colours[i % len(colours)],
                    label=str(cat), edgecolor="white", linewidth=0.4,
                )
        if len(categories) > 1:
            ax.legend(frameon=False, fontsize=9)
    else:
        ax.hist(durations, bins=bins, color=_C_PRIMARY,
                edgecolor="white", linewidth=0.4)

    _styled_ax(ax, title=title, xlabel="Duration (ms)", ylabel="Count")
    plt.tight_layout()
    return fig


def plot_event_counts(
    events: List[JsonDict],
    *,
    title: str = "Event Call Counts",
) -> matplotlib.figure.Figure:
    """Return a bar chart of call counts per event name (QNode)."""
    if not events:
        return _empty_figure("No events available for call counts.")

    df = events_to_df(events)
    counts = df["name"].value_counts().sort_values(ascending=False)

    if counts.empty:
        return _empty_figure("No event names found.")

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_C_BG)

    counts.plot(kind="bar", color=_C_PRIMARY, ax=ax, edgecolor="white", linewidth=0.5)

    _styled_ax(ax, title=title, xlabel="Event Name (QNode)", ylabel="Call Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_timing_comparison(
    events_a: List[JsonDict],
    events_b: List[JsonDict],
    *,
    bins: int = 20,
    label_a: str = "Run A",
    label_b: str = "Run B",
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Overlay timing histograms from two runs.

    Parameters
    ----------
    events_a:
        Execution events for the primary run.
    events_b:
        Execution events for the comparison run.
    bins:
        Number of histogram bins.
    label_a:
        Legend label for Run A.
    label_b:
        Legend label for Run B.
    title:
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not events_a and not events_b:
        return _empty_figure("No timing data available for comparison.")

    df_a = events_to_df(events_a)
    df_b = events_to_df(events_b)

    dur_a = df_a["duration_ms"].dropna() if not df_a.empty else pd.Series(dtype=float)
    dur_b = df_b["duration_ms"].dropna() if not df_b.empty else pd.Series(dtype=float)

    if dur_a.empty and dur_b.empty:
        return _empty_figure("No duration_ms values found in events.")

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_C_BG)

    # Use a common range if both have data
    all_vals = pd.concat([dur_a, dur_b])
    hrange = (all_vals.min(), all_vals.max()) if not all_vals.empty else None

    if not dur_a.empty:
        ax.hist(
            dur_a, bins=bins, range=hrange, alpha=0.6,
            color=_C_PRIMARY, label=label_a, edgecolor="white", linewidth=0.3,
        )
    if not dur_b.empty:
        ax.hist(
            dur_b, bins=bins, range=hrange, alpha=0.6,
            color=_C_SECONDARY, label=label_b, edgecolor="white", linewidth=0.3,
        )

    resolved_title = title or f"Timing Comparison: {label_a} vs {label_b}"
    _styled_ax(ax, title=resolved_title, xlabel="Duration (ms)", ylabel="Count")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------


def plot_loss_curve(
    gradients: List[JsonDict],
    *,
    title: str = "Training Loss",
    log_scale: bool = False,
    label: str = "loss",
    color: str = _C_PRIMARY,
) -> matplotlib.figure.Figure:
    """Return a line plot of training loss over global step.

    Parameters
    ----------
    gradients:
        List of training-step dicts.
    title:
        Figure title.
    log_scale:
        If ``True``, use a log scale for the y-axis.
    label:
        Legend label for the line.
    color:
        Line color.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not gradients:
        return _empty_figure("No gradient records — loss curve unavailable.")

    steps = _safe_ints(gradients, "global_step")
    losses = _safe_floats(gradients, "loss")

    if all(math.isnan(v) for v in losses):
        return _empty_figure("No loss values found in gradient records.")

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_C_BG)
    ax.plot(steps, losses, color=color, linewidth=2, label=label,
            marker="o", markersize=3, markerfacecolor="white", markeredgewidth=1)
    if log_scale:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.set_yscale("log")
    _styled_ax(ax, title=title, xlabel="Global Step", ylabel="Loss")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    return fig


def plot_grad_norm_curve(
    gradients: List[JsonDict],
    *,
    title: str = "Gradient Norm",
    label: str = "grad_norm",
    color: str = _C_SECONDARY,
) -> matplotlib.figure.Figure:
    """Return a line plot of gradient norm over global step.

    Parameters
    ----------
    gradients:
        List of training-step dicts.
    title:
        Figure title.
    label:
        Legend label.
    color:
        Line color.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not gradients:
        return _empty_figure("No gradient records — norm curve unavailable.")

    steps = _safe_ints(gradients, "global_step")
    norms = _safe_floats(gradients, "grad_norm")

    if all(math.isnan(v) for v in norms):
        return _empty_figure("No grad_norm values found in gradient records.")

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_C_BG)
    ax.plot(steps, norms, color=color, linewidth=2, label=label,
            marker="s", markersize=3, markerfacecolor="white", markeredgewidth=1)
    _styled_ax(ax, title=title, xlabel="Global Step", ylabel="Gradient Norm")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comparison overlay
# ---------------------------------------------------------------------------


def plot_metric_comparison(
    gradients_a: List[JsonDict],
    gradients_b: List[JsonDict],
    *,
    metric: str = "loss",
    label_a: str = "Run A",
    label_b: str = "Run B",
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Overlay *metric* from two runs on a single figure.

    Parameters
    ----------
    gradients_a:
        Training-step records for the primary run.
    gradients_b:
        Training-step records for the comparison run.
    metric:
        Field name to compare (``"loss"``, ``"grad_norm"``, or any other
        numeric field present in the records).
    label_a:
        Legend label for *gradients_a*.
    label_b:
        Legend label for *gradients_b*.
    title:
        Figure title; defaults to ``"<metric>: <label_a> vs <label_b>"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not gradients_a and not gradients_b:
        return _empty_figure("No gradient data available for comparison.")

    resolved_title = title or f"{metric}: {label_a} vs {label_b}"
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_C_BG)

    for grads, label, color, marker in (
        (gradients_a, label_a, _C_PRIMARY, "o"),
        (gradients_b, label_b, _C_SECONDARY, "s"),
    ):
        if not grads:
            continue
        steps = _safe_ints(grads, "global_step")
        vals = _safe_floats(grads, metric)
        if all(math.isnan(v) for v in vals):
            continue
        ax.plot(steps, vals, color=color, linewidth=2, label=label,
                marker=marker, markersize=3, markerfacecolor="white", markeredgewidth=1)

    _styled_ax(ax, title=resolved_title, xlabel="Global Step", ylabel=metric)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Snapshot visualisations
# ---------------------------------------------------------------------------


def _flatten_to_real(arr: np.ndarray) -> np.ndarray:
    """Return a real-valued 1-D view / copy of *arr*.

    - Complex arrays → magnitude.
    - Multi-dimensional arrays → flattened.
    - Already 1-D real → returned as-is (view when possible).
    """
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return arr.ravel().astype(float)


def plot_snapshot_probs(
    arr: np.ndarray,
    *,
    key: str = "array",
    title: Optional[str] = None,
    top_k: Optional[int] = None,
) -> matplotlib.figure.Figure:
    """Return a probability bar chart from a snapshot array.

    Interprets the array as a probability distribution (normalises to sum=1
    if needed).  Works naturally with statevectors where ``|amp|²`` gives
    probabilities, and raw probability arrays.

    Parameters
    ----------
    arr:
        Input NumPy array.  Complex arrays are converted to ``|amp|²``.
        The result is normalised to sum to 1.
    key:
        Array key (used in default title).
    title:
        Explicit figure title.
    top_k:
        If given, show only the *top_k* largest probabilities.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if arr is None or arr.size == 0:
        return _empty_figure("Snapshot array is empty.")

    flat = _flatten_to_real(arr)

    # If it looks like a statevector (complex amplitudes → probs via |·|²),
    # we may have already taken abs; check if the original was complex.
    # For display purposes treat flat values as probabilities.
    total = flat.sum()
    if total > 0:
        probs = flat / total
    else:
        return _empty_figure(f"Snapshot array '{key}' sums to zero — cannot normalise.")

    n = len(probs)
    labels = [f"|{i}⟩" for i in range(n)]

    if top_k is not None and top_k < n:
        indices = np.argsort(probs)[::-1][:top_k]
        indices = np.sort(indices)  # restore order
        probs = probs[indices]
        labels = [labels[i] for i in indices]

    resolved_title = title or f"Probability Distribution — {key}"

    fig, ax = plt.subplots(figsize=(max(6, len(probs) * 0.4 + 2), 4))
    fig.patch.set_facecolor(_C_BG)
    x = np.arange(len(probs))
    bars = ax.bar(x, probs, color=_C_PRIMARY, edgecolor="white",
                  linewidth=0.5, zorder=3)

    # Annotate bars above a threshold
    threshold = 0.02
    for bar, p in zip(bars, probs):
        if p >= threshold:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{p:.3f}",
                ha="center", va="bottom", fontsize=7, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if len(probs) > 8 else 0,
                       ha="right", fontsize=8)
    _styled_ax(ax, title=resolved_title, xlabel="Basis State", ylabel="Probability")
    plt.tight_layout()
    return fig


def plot_snapshot_amplitudes(
    arr: np.ndarray,
    *,
    key: str = "array",
    top_k: int = 16,
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Return a horizontal bar chart of the top-*k* amplitude magnitudes.

    Useful for inspecting which basis states carry the most weight in a
    statevector or other amplitude array.

    Parameters
    ----------
    arr:
        Input NumPy array.  Complex arrays are automatically converted to
        magnitudes (``|amp|``).  Multi-dimensional arrays are flattened.
    key:
        Array key (used in default title).
    top_k:
        Maximum number of amplitudes to display.
    title:
        Explicit figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if arr is None or arr.size == 0:
        return _empty_figure("Snapshot array is empty.")

    flat = _flatten_to_real(arr)
    n = len(flat)

    k = min(top_k, n)
    indices = np.argsort(flat)[::-1][:k]
    magnitudes = flat[indices]
    labels = [f"|{i}⟩" for i in indices]

    resolved_title = title or f"Top-{k} Amplitude Magnitudes — {key}"

    fig, ax = plt.subplots(figsize=(7, max(3, k * 0.35 + 1.5)))
    fig.patch.set_facecolor(_C_BG)

    # Colour bars by rank: strongest = primary colour, fades for lower ranks
    cmap = plt.cm.get_cmap("Blues_r", k + 2)  # type: ignore[attr-defined]
    colors = [cmap(i / (k + 1)) for i in range(k)]

    y = np.arange(k)
    bars = ax.barh(y, magnitudes, color=colors, edgecolor="white",
                   linewidth=0.5, zorder=3)

    for bar, mag in zip(bars, magnitudes):
        ax.text(
            bar.get_width() + 0.001 * (magnitudes.max() or 1),
            bar.get_y() + bar.get_height() / 2,
            f"{mag:.4f}",
            va="center", ha="left", fontsize=7, color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    _styled_ax(ax, title=resolved_title, xlabel="Magnitude |amplitude|", ylabel="Basis State")
    plt.tight_layout()
    return fig


def plot_snapshot_array(
    arr: np.ndarray,
    *,
    key: str = "array",
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Auto-dispatch snapshot visualisation based on array shape.

    - 1-D or flattened-small (≤ 64 elements) → :func:`plot_snapshot_probs`
    - 2-D → heatmap (``imshow``)
    - Higher-D → flatten and treat as 1-D line plot

    Parameters
    ----------
    arr:
        NumPy array to visualise.
    key:
        Array key (used in default title).
    title:
        Explicit figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if arr is None or arr.size == 0:
        return _empty_figure(f"Snapshot array '{key}' is empty.")

    resolved_title = title or key

    # 2-D: heatmap
    if arr.ndim == 2:
        data = np.abs(arr) if np.iscomplexobj(arr) else arr.astype(float)
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(_C_BG)
        im = ax.imshow(data, aspect="auto", cmap="viridis", origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _styled_ax(ax, title=resolved_title, xlabel="Column", ylabel="Row")
        plt.tight_layout()
        return fig

    # 1-D with ≤ 64 elements (likely a statevector / probability array)
    flat = _flatten_to_real(arr)
    if flat.size <= 64:
        return plot_snapshot_probs(arr, key=key, title=resolved_title)

    # Fallback: simple line plot for large 1-D arrays
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_C_BG)
    ax.plot(flat, color=_C_PRIMARY, linewidth=1)
    _styled_ax(ax, title=resolved_title, xlabel="Index", ylabel="Value")
    plt.tight_layout()
    return fig
