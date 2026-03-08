"""In-memory state containers for the pl-inspector dashboard.

This module defines the two central state objects that the Gradio event
handlers pass around:

- :class:`LoadedRun` — all artefacts for a single run, with helper
  accessors and a user-friendly display label.
- :class:`CompareState` — a pair of ``LoadedRun`` instances together with
  pre-computed metric comparison data (step overlap, delta series).

Factories
---------
- :func:`load_run`      — load one run from disk into a :class:`LoadedRun`
- :func:`load_run_pair` — load two runs and build a :class:`CompareState`

Design notes
------------
- ``LoadedRun`` is **mutable** (not frozen) so Gradio callbacks can patch
  ``warnings`` incrementally without constructing a fresh object each time.
- ``CompareState`` intentionally stores *lists* for computed metric data
  (rather than NumPy arrays) to stay dependency-light at the state layer;
  the render layer converts to arrays for plotting.
- The ``snapshots`` field eagerly loads array data into a plain ``dict``
  so the UI does not need to juggle open file handles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io import RunData, get_run_dir, load_run_dir, load_snapshots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

JsonDict = Dict[str, Any]

# ---------------------------------------------------------------------------
# LoadedRun
# ---------------------------------------------------------------------------


@dataclass
class LoadedRun:
    """In-memory representation of a fully (or partially) loaded run.

    Constructed via :func:`load_run` or :meth:`from_run_data`; do not
    instantiate directly unless writing tests.

    Parameters
    ----------
    run_id:
        The directory name used as the run identifier
        (e.g. ``"run-20260307T090603Z"``).
    run_dir:
        Resolved absolute path to the run directory.
    meta:
        Parsed ``meta.json`` payload, or ``None`` if absent / unreadable.
    summary:
        Parsed ``summary.json`` payload, or ``None`` if absent / unreadable.
    events:
        Ordered list of parsed event records from ``events.jsonl``.
    gradients:
        Ordered list of parsed training-step records from ``gradients.jsonl``.
    snapshots:
        Mapping from array key to NumPy array, eagerly loaded from
        ``snapshots.npz``.  Empty dict when the file is absent.
    warnings:
        List of human-readable warning strings collected during loading
        (e.g. parse errors, missing optional files).
    load_errors:
        Mapping from artefact name to error message for any files that failed
        to load.  Same as :attr:`io.RunData.load_errors`.
    """

    run_id: str
    run_dir: Path
    meta: Optional[JsonDict] = None
    summary: Optional[JsonDict] = None
    events: List[JsonDict] = field(default_factory=list)
    gradients: List[JsonDict] = field(default_factory=list)
    snapshots: Dict[str, np.ndarray] = field(default_factory=dict)
    snapshot_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)
    load_errors: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Safe accessors
    # ------------------------------------------------------------------

    @property
    def has_meta(self) -> bool:
        """Return ``True`` when ``meta.json`` was loaded successfully."""
        return self.meta is not None

    @property
    def has_summary(self) -> bool:
        """Return ``True`` when ``summary.json`` was loaded successfully."""
        return self.summary is not None

    @property
    def has_events(self) -> bool:
        """Return ``True`` when at least one event record is available."""
        return bool(self.events)

    @property
    def has_gradients(self) -> bool:
        """Return ``True`` when at least one gradient / training-step record is available."""
        return bool(self.gradients)

    @property
    def has_snapshots(self) -> bool:
        """Return ``True`` when at least one snapshot array was loaded."""
        return bool(self.snapshots)

    @property
    def has_errors(self) -> bool:
        """Return ``True`` when at least one artefact failed to load."""
        return bool(self.load_errors)

    @property
    def snapshot_keys(self) -> List[str]:
        """Return the sorted list of array keys in ``snapshots``."""
        return sorted(self.snapshots.keys())

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def display_label(self, *, max_desc_chars: int = 40) -> str:
        """Return a concise, human-readable label for use in UI dropdowns.

        Format: ``<run_id>  [<description>]``

        The description is truncated to *max_desc_chars* characters.
        If no description is present in meta, only the run ID is returned.

        Parameters
        ----------
        max_desc_chars:
            Maximum number of characters to show from the description.

        Returns
        -------
        str
            A non-empty label string.
        """
        if self.meta:
            desc: Optional[str] = self.meta.get("description")
            if desc and isinstance(desc, str):
                desc = desc.strip()
                if len(desc) > max_desc_chars:
                    desc = desc[:max_desc_chars].rstrip() + "…"
                return f"{self.run_id}  [{desc}]"
        return self.run_id

    def created_at(self) -> Optional[str]:
        """Return the ISO 8601 creation timestamp from meta, or ``None``."""
        if self.meta:
            return self.meta.get("created_at")
        return None

    def num_steps(self) -> int:
        """Return the number of training steps recorded in ``gradients.jsonl``."""
        return len(self.gradients)

    def loss_series(self) -> List[float]:
        """Return an ordered list of loss values from the gradient records.

        Records missing the ``"loss"`` key contribute ``float("nan")`` so
        that plot indices align with step numbers.
        """
        return [
            float(g["loss"]) if isinstance(g.get("loss"), (int, float)) else float("nan")
            for g in self.gradients
        ]

    def grad_norm_series(self) -> List[float]:
        """Return an ordered list of gradient norm values from the records.

        Missing values are represented as ``float("nan")``.
        """
        return [
            float(g["grad_norm"])
            if isinstance(g.get("grad_norm"), (int, float))
            else float("nan")
            for g in self.gradients
        ]

    def global_steps(self) -> List[int]:
        """Return an ordered list of ``global_step`` integers.

        Defaults to a 0-based index when the field is absent.
        """
        return [
            int(g["global_step"]) if isinstance(g.get("global_step"), int) else i
            for i, g in enumerate(self.gradients)
        ]

    def get_snapshot(self, key: str) -> Optional[np.ndarray]:
        """Return the NumPy array for *key*, or ``None`` if absent.

        Parameters
        ----------
        key:
            Array key as stored in ``snapshots.npz``.
        """
        return self.snapshots.get(key)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_run_data(cls, data: RunData) -> "LoadedRun":
        """Construct a :class:`LoadedRun` from a :class:`~pl_inspector.ui.io.RunData`.

        Eagerly loads snapshot arrays from ``snapshots.npz`` if the file is
        present, capturing any errors as warnings rather than propagating them.

        Parameters
        ----------
        data:
            A :class:`~pl_inspector.ui.io.RunData` as returned by
            :func:`pl_inspector.ui.io.load_run_dir`.

        Returns
        -------
        LoadedRun
            Fully populated state object ready for the dashboard.
        """
        warnings_list: List[str] = []

        # Convert existing load_errors into warning strings for the UI
        for artefact, msg in data.load_errors.items():
            warnings_list.append(f"⚠ {artefact}: {msg}")

        # Eagerly load snapshot arrays into a plain dict
        snapshots: Dict[str, np.ndarray] = {}
        if data.snapshot_path is not None:
            try:
                with load_snapshots(data.run_dir) as npz:  # type: ignore[assignment]
                    for key in npz.files:
                        snapshots[key] = npz[key]
            except Exception as exc:  # noqa: BLE001
                msg = f"snapshots.npz: could not load arrays — {exc}"
                warnings_list.append(f"⚠ {msg}")
                logger.warning(msg)

        return cls(
            run_id=data.run_id,
            run_dir=data.run_dir,
            meta=data.meta,
            summary=data.summary,
            events=data.events,
            gradients=data.gradients,
            snapshots=snapshots,
            snapshot_path=data.snapshot_path,
            warnings=warnings_list,
            load_errors=dict(data.load_errors),
        )


# ---------------------------------------------------------------------------
# CompareState
# ---------------------------------------------------------------------------


@dataclass
class MetricComparison:
    """Pre-computed comparative data for a single metric between two runs.

    Parameters
    ----------
    metric:
        Name of the metric (e.g. ``"loss"``, ``"grad_norm"``).
    steps_a:
        Global step indices for run A.
    values_a:
        Metric values for run A, aligned with *steps_a*.
    steps_b:
        Global step indices for run B.
    values_b:
        Metric values for run B, aligned with *steps_b*.
    overlap_steps:
        Steps that appear in *both* runs (intersection of step sets).
    delta:
        ``value_b − value_a`` for each step in *overlap_steps*.
        Empty when there are no overlapping steps.
    """

    metric: str
    steps_a: List[int]
    values_a: List[float]
    steps_b: List[int]
    values_b: List[float]
    overlap_steps: List[int] = field(default_factory=list)
    delta: List[float] = field(default_factory=list)


def _compute_metric_comparison(
    run_a: LoadedRun,
    run_b: LoadedRun,
    metric: str,
) -> MetricComparison:
    """Compute a :class:`MetricComparison` for *metric* between two runs.

    Parameters
    ----------
    run_a:
        Primary run.
    run_b:
        Comparison run.
    metric:
        Field name to extract (``"loss"`` or ``"grad_norm"``).

    Returns
    -------
    MetricComparison
        Populated comparison, with delta for overlapping steps.
    """
    steps_a = run_a.global_steps()
    steps_b = run_b.global_steps()

    if metric == "loss":
        vals_a = run_a.loss_series()
        vals_b = run_b.loss_series()
    elif metric == "grad_norm":
        vals_a = run_a.grad_norm_series()
        vals_b = run_b.grad_norm_series()
    else:
        # Generic extraction — missing keys become NaN
        vals_a = [
            float(g[metric]) if isinstance(g.get(metric), (int, float)) else float("nan")
            for g in run_a.gradients
        ]
        vals_b = [
            float(g[metric]) if isinstance(g.get(metric), (int, float)) else float("nan")
            for g in run_b.gradients
        ]

    # Build step → value maps for delta computation
    map_a: Dict[int, float] = dict(zip(steps_a, vals_a))
    map_b: Dict[int, float] = dict(zip(steps_b, vals_b))
    overlap = sorted(set(map_a) & set(map_b))
    delta = [map_b[s] - map_a[s] for s in overlap]

    return MetricComparison(
        metric=metric,
        steps_a=steps_a,
        values_a=vals_a,
        steps_b=steps_b,
        values_b=vals_b,
        overlap_steps=overlap,
        delta=delta,
    )


@dataclass
class CompareState:
    """State for the "Comparison" tab — two loaded runs and pre-computed diffs.

    Parameters
    ----------
    run_a:
        Primary run (left / reference).
    run_b:
        Comparison run (right).
    comparisons:
        Pre-computed :class:`MetricComparison` objects keyed by metric name.
        Populated automatically by :func:`load_run_pair`.
    """

    run_a: LoadedRun
    run_b: LoadedRun
    comparisons: Dict[str, MetricComparison] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Safe accessors
    # ------------------------------------------------------------------

    @property
    def can_compare_gradients(self) -> bool:
        """Return ``True`` when both runs have gradient records."""
        return self.run_a.has_gradients and self.run_b.has_gradients

    @property
    def available_metrics(self) -> List[str]:
        """Return the list of metrics for which comparisons were computed."""
        return list(self.comparisons.keys())

    def get_comparison(self, metric: str) -> Optional[MetricComparison]:
        """Return the :class:`MetricComparison` for *metric*, or ``None``."""
        return self.comparisons.get(metric)

    def summary_table(self) -> List[Tuple[str, Any, Any]]:
        """Return a list of ``(metric, value_a, value_b)`` summary rows.

        Includes ``num_steps``, ``final_loss``, and ``final_grad_norm`` for
        each run, formatted for display in a comparison table.

        Returns
        -------
        list of (metric, value_a, value_b)
        """

        def _fmt(v: Any) -> str:
            if isinstance(v, float) and not (v != v):  # NaN check
                return f"{v:.6g}"
            return str(v)

        def _final(series: List[float]) -> str:
            for v in reversed(series):
                if v == v:  # not NaN
                    return _fmt(v)
            return "N/A"

        rows: List[Tuple[str, Any, Any]] = [
            ("run_id", self.run_a.run_id, self.run_b.run_id),
            ("num_steps", self.run_a.num_steps(), self.run_b.num_steps()),
            (
                "final_loss",
                _final(self.run_a.loss_series()),
                _final(self.run_b.loss_series()),
            ),
            (
                "final_grad_norm",
                _final(self.run_a.grad_norm_series()),
                _final(self.run_b.grad_norm_series()),
            ),
        ]
        return rows


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

_DEFAULT_COMPARE_METRICS = ("loss", "grad_norm")


def load_run(base_dir: str | Path, run_id: str) -> LoadedRun:
    """Load all artefacts for *run_id* and return a :class:`LoadedRun`.

    Parameters
    ----------
    base_dir:
        Root directory that contains the ``runs/`` sub-directory.
    run_id:
        Identifier of the run to load.

    Returns
    -------
    LoadedRun
        Populated state object; some fields may be empty if individual
        artefacts are missing — inspect :attr:`LoadedRun.warnings` and
        :attr:`LoadedRun.load_errors` for details.

    Raises
    ------
    ~pl_inspector.ui.errors.InvalidRunDirectoryError
        If *base_dir* is not a readable directory.
    ~pl_inspector.ui.errors.RunNotFoundError
        If the run directory does not exist under *base_dir*.
    """
    run_dir = get_run_dir(base_dir, run_id)
    data = load_run_dir(run_dir)
    return LoadedRun.from_run_data(data)


def load_run_pair(
    base_dir: str | Path,
    run_id_a: str,
    run_id_b: str,
    *,
    metrics: Tuple[str, ...] = _DEFAULT_COMPARE_METRICS,
) -> CompareState:
    """Load two runs and build a :class:`CompareState` with pre-computed diffs.

    Parameters
    ----------
    base_dir:
        Root directory that contains the ``runs/`` sub-directory.
    run_id_a:
        Identifier of the primary (reference) run.
    run_id_b:
        Identifier of the comparison run.
    metrics:
        Names of metrics to pre-compute comparisons for.  Defaults to
        ``("loss", "grad_norm")``.

    Returns
    -------
    CompareState
        Populated comparison state; badly loaded artefacts are captured in
        each ``LoadedRun.warnings`` rather than propagated.

    Raises
    ------
    ~pl_inspector.ui.errors.InvalidRunDirectoryError
        If *base_dir* is not a readable directory.
    ~pl_inspector.ui.errors.RunNotFoundError
        If either run directory does not exist.
    """
    run_a = load_run(base_dir, run_id_a)
    run_b = load_run(base_dir, run_id_b)

    comparisons: Dict[str, MetricComparison] = {}
    if run_a.has_gradients and run_b.has_gradients:
        for metric in metrics:
            try:
                comparisons[metric] = _compute_metric_comparison(run_a, run_b, metric)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not compute comparison for metric %r: %s", metric, exc)

    return CompareState(run_a=run_a, run_b=run_b, comparisons=comparisons)
