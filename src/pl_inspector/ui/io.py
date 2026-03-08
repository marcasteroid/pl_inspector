"""Read-only I/O helpers for the pl-inspector dashboard.

All functions in this module are **pure readers**: they load artefact files from
an existing ``runs/<run_id>/`` directory and return typed Python structures.
They never write to disk.

Expected on-disk layout (as produced by :mod:`pl_inspector.storage`)::

    <base_dir>/runs/<run_id>/
        meta.json         — RunMeta fields as JSON
        summary.json      — Free-form summary dict as JSON (optional)
        events.jsonl      — Newline-delimited ExecutionEvent JSON (optional)
        gradients.jsonl   — Newline-delimited TrainingStepRecord JSON (optional)
        snapshots.npz     — NumPy compressed archive (optional)

Public API
----------
- :func:`list_runs`      — enumerate run IDs under a base directory
- :func:`get_run_dir`    — resolve the directory path for a run ID
- :func:`load_meta`      — parse ``meta.json`` (required)
- :func:`load_summary`   — parse ``summary.json`` (optional, returns ``None``)
- :func:`load_events`    — parse ``events.jsonl`` (optional, returns ``[]``)
- :func:`load_gradients` — parse ``gradients.jsonl`` (optional, returns ``[]``)
- :func:`load_snapshots` — open ``snapshots.npz`` (optional, returns ``None``)
- :func:`load_run_dir`   — load all artefacts into a :class:`RunData` dataclass
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .errors import (
    ArtifactMissingError,
    ArtifactParseError,
    InvalidRunDirectoryError,
    RunNotFoundError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

JsonDict = Dict[str, Any]
"""A plain Python dictionary decoded from JSON."""

# ---------------------------------------------------------------------------
# Structured result container
# ---------------------------------------------------------------------------


@dataclass
class RunData:
    """All artefacts loaded for a single run, with per-artefact error capture.

    Fields are ``None`` / empty list when the corresponding file is absent or
    unreadable.  All loading errors are recorded in :attr:`load_errors` so the
    dashboard can still render partial results.

    Parameters
    ----------
    run_id:
        The directory name used as the run identifier.
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
    snapshot_keys:
        Names of arrays inside ``snapshots.npz`` (populated when the file
        is present but kept closed to avoid holding file handles).
    snapshot_path:
        Path to ``snapshots.npz``, or ``None`` if absent.
    load_errors:
        Mapping from artefact name (e.g. ``"meta.json"``) to a short error
        message.  Empty dict when all artefacts loaded without errors.
    """

    run_id: str
    run_dir: Path
    meta: Optional[JsonDict] = None
    summary: Optional[JsonDict] = None
    events: List[JsonDict] = field(default_factory=list)
    gradients: List[JsonDict] = field(default_factory=list)
    snapshot_keys: List[str] = field(default_factory=list)
    snapshot_path: Optional[Path] = None
    load_errors: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def has_events(self) -> bool:
        """Return ``True`` when at least one event record was loaded."""
        return bool(self.events)

    @property
    def has_gradients(self) -> bool:
        """Return ``True`` when at least one gradient/training-step record was loaded."""
        return bool(self.gradients)

    @property
    def has_snapshots(self) -> bool:
        """Return ``True`` when a ``snapshots.npz`` file is present."""
        return self.snapshot_path is not None

    @property
    def has_errors(self) -> bool:
        """Return ``True`` when at least one artefact failed to load."""
        return bool(self.load_errors)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

_RUNS_SUBDIR = "runs"


def _find_runs_root(base_dir: Path) -> Path:
    """Locate the ``runs/`` sub-directory inside *base_dir*.

    Tries ``<base_dir>/runs/`` first, then treats *base_dir* itself as the
    runs root if it contains subdirectories that look like run dirs.

    Returns the directory; does **not** raise if it does not exist.
    """
    candidate = base_dir / _RUNS_SUBDIR
    if candidate.is_dir():
        return candidate
    # Fallback: treat base_dir itself as the runs root
    return base_dir


def list_runs(base_dir: str | Path) -> List[str]:
    """Return a sorted list of run IDs under *base_dir*.

    A run ID is the directory name of any immediate sub-directory of the
    ``runs/`` sub-directory (or of *base_dir* itself if no ``runs/``
    sub-directory exists).  Sub-directories that contain no recognisable
    artefact files are silently excluded.

    Runs are returned newest-first: sorted by the ``created_at`` timestamp
    in ``meta.json`` when available, otherwise by directory mtime.

    Parameters
    ----------
    base_dir:
        Root directory that (optionally) contains a ``runs/`` sub-directory.

    Returns
    -------
    list[str]
        List of run ID strings, newest first.  Empty list if none found.

    Raises
    ------
    InvalidRunDirectoryError
        If *base_dir* does not exist or is not a readable directory.
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.exists():
        raise InvalidRunDirectoryError(str(base), "path does not exist")
    if not base.is_dir():
        raise InvalidRunDirectoryError(str(base), "path is not a directory")

    runs_root = _find_runs_root(base)
    if not runs_root.is_dir():
        return []

    _KNOWN_ARTEFACTS = {"meta.json", "summary.json", "events.jsonl",
                        "gradients.jsonl", "snapshots.npz"}

    candidates: List[Path] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        # Accept if the directory contains at least one recognised artefact
        contents = {p.name for p in child.iterdir()}
        if contents & _KNOWN_ARTEFACTS:
            candidates.append(child)

    def _sort_key(p: Path) -> str:
        meta = p / "meta.json"
        if meta.is_file():
            try:
                with meta.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                ts: str = data.get("created_at", "")
                if ts:
                    return ts
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        # Fallback: use ISO-formatted mtime so newer dirs sort higher
        return str(p.stat().st_mtime)

    candidates.sort(key=_sort_key, reverse=True)
    return [p.name for p in candidates]


def get_run_dir(base_dir: str | Path, run_id: str) -> Path:
    """Return the resolved path to the run directory for *run_id*.

    Parameters
    ----------
    base_dir:
        Root directory that contains a ``runs/`` sub-directory.
    run_id:
        The run identifier string.

    Returns
    -------
    Path
        Fully resolved path to the run directory.

    Raises
    ------
    InvalidRunDirectoryError
        If *base_dir* is not a readable directory.
    RunNotFoundError
        If the run directory does not exist.
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.is_dir():
        raise InvalidRunDirectoryError(str(base), "path is not a directory")

    runs_root = _find_runs_root(base)
    run_dir = runs_root / run_id

    if not run_dir.is_dir():
        raise RunNotFoundError(run_id)

    return run_dir.resolve()


# ---------------------------------------------------------------------------
# Individual artefact loaders
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> JsonDict:
    """Read and return a JSON file as a dict.

    Raises
    ------
    ArtifactMissingError
        If *path* does not exist.
    ArtifactParseError
        If *path* exists but cannot be decoded as JSON, or decodes to a
        non-dict value.
    """
    if not path.exists():
        raise ArtifactMissingError(str(path))
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ArtifactParseError(str(path), f"JSON decode error: {exc}") from exc
    except OSError as exc:
        raise ArtifactParseError(str(path), f"OS error while reading: {exc}") from exc

    if not isinstance(data, dict):
        raise ArtifactParseError(
            str(path),
            f"Expected a JSON object (dict), got {type(data).__name__!r}",
        )
    return data  # type: ignore[return-value]


def _read_jsonl(path: Path) -> List[JsonDict]:
    """Read a JSONL file, skipping and warning on malformed lines.

    Returns an empty list if the file does not exist.

    Raises
    ------
    ArtifactParseError
        If the file exists but cannot be opened at all (OS error).
    """
    if not path.exists():
        return []

    records: List[JsonDict] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        warnings.warn(
                            f"{path}:{lineno}: line is valid JSON but not an object "
                            f"(got {type(obj).__name__!r}); skipping.",
                            stacklevel=2,
                        )
                except json.JSONDecodeError as exc:
                    warnings.warn(
                        f"{path}:{lineno}: could not parse line as JSON ({exc}); skipping.",
                        stacklevel=2,
                    )
    except OSError as exc:
        raise ArtifactParseError(str(path), f"OS error while reading: {exc}") from exc

    return records


def load_meta(run_dir: str | Path) -> JsonDict:
    """Load and return the contents of ``meta.json`` for a run.

    Parameters
    ----------
    run_dir:
        Path to the individual run directory (e.g. ``runs/<run_id>/``).

    Returns
    -------
    dict
        Parsed JSON payload from ``meta.json``.

    Raises
    ------
    ArtifactMissingError
        If ``meta.json`` does not exist.
    ArtifactParseError
        If ``meta.json`` exists but cannot be decoded as a JSON object.
    """
    path = Path(run_dir) / "meta.json"
    return _read_json(path)


def load_summary(run_dir: str | Path) -> Optional[JsonDict]:
    """Load and return the contents of ``summary.json`` for a run.

    Unlike :func:`load_meta`, a missing file is not an error: the function
    returns ``None`` and logs a debug message.

    Parameters
    ----------
    run_dir:
        Path to the individual run directory.

    Returns
    -------
    dict or None
        Parsed JSON payload, or ``None`` if the file does not exist.

    Raises
    ------
    ArtifactParseError
        If the file exists but cannot be decoded.
    """
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        logger.debug("summary.json not found in %s — skipping.", run_dir)
        return None
    return _read_json(path)


def load_events(run_dir: str | Path) -> List[JsonDict]:
    """Load execution events from ``events.jsonl``.

    Each line of the file is expected to be a standalone JSON object.
    Lines that fail to parse emit a :class:`UserWarning` and are skipped.

    Parameters
    ----------
    run_dir:
        Path to the individual run directory.

    Returns
    -------
    list[dict]
        Ordered list of event dictionaries.  Empty list if absent or empty.

    Raises
    ------
    ArtifactParseError
        If the file exists but cannot be opened due to an OS error.
    """
    path = Path(run_dir) / "events.jsonl"
    records = _read_jsonl(path)
    if not records and path.exists():
        logger.debug("events.jsonl at %s is present but contains no valid records.", path)
    return records


def load_gradients(run_dir: str | Path) -> List[JsonDict]:
    """Load training-step records from ``gradients.jsonl``.

    Mirrors :func:`load_events` in behaviour: missing file → empty list;
    malformed lines → :class:`UserWarning` and skip.

    Parameters
    ----------
    run_dir:
        Path to the individual run directory.

    Returns
    -------
    list[dict]
        Ordered list of training-step dictionaries.  Empty list if absent.

    Raises
    ------
    ArtifactParseError
        If the file exists but cannot be opened due to an OS error.
    """
    path = Path(run_dir) / "gradients.jsonl"
    records = _read_jsonl(path)
    if not records and path.exists():
        logger.debug("gradients.jsonl at %s is present but contains no valid records.", path)
    return records


def load_snapshots(run_dir: str | Path) -> Optional[np.lib.npyio.NpzFile]:
    """Open ``snapshots.npz`` and return the lazy :class:`~numpy.lib.npyio.NpzFile`.

    The caller **must** close the returned object when done (use as a context
    manager or call ``.close()`` explicitly).

    Parameters
    ----------
    run_dir:
        Path to the individual run directory.

    Returns
    -------
    numpy.lib.npyio.NpzFile or None
        Open NpzFile handle, or ``None`` if the file does not exist.

    Raises
    ------
    ArtifactParseError
        If the file exists but cannot be loaded by NumPy.
    """
    path = Path(run_dir) / "snapshots.npz"
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError, Exception) as exc:  # noqa: BLE001
        raise ArtifactParseError(str(path), str(exc)) from exc


# ---------------------------------------------------------------------------
# Composite loader
# ---------------------------------------------------------------------------


def load_run_dir(run_dir: str | Path) -> RunData:
    """Load all artefacts from *run_dir* into a single :class:`RunData` object.

    Individual artefact failures are captured into :attr:`RunData.load_errors`
    rather than propagated, so the dashboard can always render partial results.
    Only a completely unreadable directory raises immediately.

    Parameters
    ----------
    run_dir:
        Path to the individual run directory (must be an existing directory).

    Returns
    -------
    RunData
        Fully or partially populated run data.  Inspect
        :attr:`RunData.load_errors` to see which (if any) artefacts failed.

    Raises
    ------
    InvalidRunDirectoryError
        If *run_dir* does not exist or is not a directory.
    """
    path = Path(run_dir).expanduser().resolve()

    if not path.exists():
        raise InvalidRunDirectoryError(str(path), "path does not exist")
    if not path.is_dir():
        raise InvalidRunDirectoryError(str(path), "path is not a directory")

    run_id = path.name
    errors: Dict[str, str] = {}

    # --- meta.json (recommended but gracefully handled if missing) ----------
    meta: Optional[JsonDict] = None
    try:
        meta = load_meta(path)
    except (ArtifactMissingError, ArtifactParseError) as exc:
        logger.warning("Could not load meta.json for %s: %s", run_id, exc)
        errors["meta.json"] = str(exc)

    # --- summary.json (optional) -------------------------------------------
    summary: Optional[JsonDict] = None
    try:
        summary = load_summary(path)
    except ArtifactParseError as exc:
        logger.warning("Could not load summary.json for %s: %s", run_id, exc)
        errors["summary.json"] = str(exc)

    # --- events.jsonl (optional) -------------------------------------------
    events: List[JsonDict] = []
    try:
        events = load_events(path)
    except ArtifactParseError as exc:
        logger.warning("Could not load events.jsonl for %s: %s", run_id, exc)
        errors["events.jsonl"] = str(exc)

    # --- gradients.jsonl (optional) ----------------------------------------
    gradients: List[JsonDict] = []
    try:
        gradients = load_gradients(path)
    except ArtifactParseError as exc:
        logger.warning("Could not load gradients.jsonl for %s: %s", run_id, exc)
        errors["gradients.jsonl"] = str(exc)

    # --- snapshots.npz (optional) ------------------------------------------
    snapshot_keys: List[str] = []
    snapshot_path: Optional[Path] = None
    npz_path = path / "snapshots.npz"
    if npz_path.exists():
        try:
            with load_snapshots(path) as npz:  # type: ignore[assignment]
                snapshot_keys = list(npz.files)
            snapshot_path = npz_path
        except ArtifactParseError as exc:
            logger.warning("Could not load snapshots.npz for %s: %s", run_id, exc)
            errors["snapshots.npz"] = str(exc)

    return RunData(
        run_id=run_id,
        run_dir=path,
        meta=meta,
        summary=summary,
        events=events,
        gradients=gradients,
        snapshot_keys=snapshot_keys,
        snapshot_path=snapshot_path,
        load_errors=errors,
    )


# ---------------------------------------------------------------------------
# Report Export
# ---------------------------------------------------------------------------

def export_run_report(
    run_dir: Path,
    meta: Optional[JsonDict],
    summary: Optional[JsonDict],
    events: List[JsonDict],
    gradients: List[JsonDict],
) -> Path:
    """Generate and save a Markdown report with embedded PNG plots.

    Creates a ``report.md`` file and a ``report_assets/`` directory within
    *run_dir*.

    Returns
    -------
    Path
        Absolute path to the generated ``report.md``.
    """
    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    from .render import (
        plot_loss_curve,
        plot_grad_norm_curve,
        plot_timing_histogram,
        plot_event_counts,
    )
    import matplotlib.pyplot as plt

    # 1. Generate plots
    plot_paths = {}
    
    # Loss
    if gradients:
        fig_loss = plot_loss_curve(gradients)
        loss_path = assets_dir / "loss.png"
        fig_loss.savefig(loss_path, bbox_inches="tight", dpi=100)
        plot_paths["loss"] = "report_assets/loss.png"
        plt.close(fig_loss)

    # Grad Norm
    if gradients:
        fig_gnorm = plot_grad_norm_curve(gradients)
        gnorm_path = assets_dir / "grad_norm.png"
        fig_gnorm.savefig(gnorm_path, bbox_inches="tight", dpi=100)
        plot_paths["grad_norm"] = "report_assets/grad_norm.png"
        plt.close(fig_gnorm)

    # Timing
    if events:
        fig_timing = plot_timing_histogram(events)
        timing_path = assets_dir / "timing.png"
        fig_timing.savefig(timing_path, bbox_inches="tight", dpi=100)
        plot_paths["timing"] = "report_assets/timing.png"
        plt.close(fig_timing)

    # Event Counts
    if events:
        fig_cnts = plot_event_counts(events)
        cnts_path = assets_dir / "event_counts.png"
        fig_cnts.savefig(cnts_path, bbox_inches="tight", dpi=100)
        plot_paths["counts"] = "report_assets/event_counts.png"
        plt.close(fig_cnts)

    # 2. Build Markdown
    run_id = run_dir.name
    md = [f"# Run Report: {run_id}\n"]
    
    if meta:
        md.append("## Meta")
        md.append("```json")
        md.append(json.dumps(meta, indent=2))
        md.append("```\n")

    if summary:
        md.append("## Summary")
        md.append("```json")
        md.append(json.dumps(summary, indent=2))
        md.append("```\n")

    md.append("## Statistics")
    md.append(f"- **Total Steps**: {len(gradients)}")
    md.append(f"- **Total Events**: {len(events)}")
    md.append("")

    if "loss" in plot_paths:
        md.append("## Training Loss")
        md.append(f"![Training Loss]({plot_paths['loss']})\n")

    if "grad_norm" in plot_paths:
        md.append("## Gradient Norm")
        md.append(f"![Gradient Norm]({plot_paths['grad_norm']})\n")

    if "timing" in plot_paths:
        md.append("## Execution Timing")
        md.append(f"![Timing]({plot_paths['timing']})\n")

    if "counts" in plot_paths:
        md.append("## Event Call Counts")
        md.append(f"![Event Counts]({plot_paths['counts']})\n")
        
        # Add a table of event counts
        import pandas as pd
        from .render import events_to_df
        df = events_to_df(events)
        counts = df["name"].value_counts().reset_index()
        counts.columns = ["QNode", "Calls"]
        md.append("### Call Details")
        md.append("| QNode | Calls |")
        md.append("| :--- | :--- |")
        for _, row in counts.iterrows():
            md.append(f"| {row['QNode']} | {row['Calls']} |")
        md.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    return report_path


def export_comparison_report(
    run_dir_a: Path,
    run_a_id: str,
    meta_a: Optional[JsonDict],
    sum_a: Optional[JsonDict],
    ev_a: List[JsonDict],
    gr_a: List[JsonDict],
    run_b_id: str,
    meta_b: Optional[JsonDict],
    sum_b: Optional[JsonDict],
    ev_b: List[JsonDict],
    gr_b: List[JsonDict],
) -> Path:
    """Generate and save a comparison Markdown report.

    Creates ``comparison_report.md`` and ``comparison_assets/`` within
    *run_dir_a*.
    """
    assets_dir = run_dir_a / "comparison_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    from .render import (
        plot_metric_comparison,
        plot_timing_comparison,
    )
    import matplotlib.pyplot as plt

    # 1. Generate plots
    plot_paths = {}

    # Loss comparison
    if gr_a and gr_b:
        fig_loss = plot_metric_comparison(gr_a, gr_b, metric="loss", label_a=run_a_id, label_b=run_b_id)
        path = assets_dir / "loss_compare.png"
        fig_loss.savefig(path, bbox_inches="tight", dpi=100)
        plot_paths["loss"] = "comparison_assets/loss_compare.png"
        plt.close(fig_loss)

    # Grad Norm comparison
    if gr_a and gr_b:
        fig_gnorm = plot_metric_comparison(gr_a, gr_b, metric="grad_norm", label_a=run_a_id, label_b=run_b_id)
        path = assets_dir / "grad_norm_compare.png"
        fig_gnorm.savefig(path, bbox_inches="tight", dpi=100)
        plot_paths["grad_norm"] = "comparison_assets/grad_norm_compare.png"
        plt.close(fig_gnorm)

    # Timing comparison
    if ev_a and ev_b:
        fig_timing = plot_timing_comparison(ev_a, ev_b, label_a=run_a_id, label_b=run_b_id)
        path = assets_dir / "timing_compare.png"
        fig_timing.savefig(path, bbox_inches="tight", dpi=100)
        plot_paths["timing"] = "comparison_assets/timing_compare.png"
        plt.close(fig_timing)

    # 2. Build Markdown
    md = [f"# Comparison Report: {run_a_id} vs {run_b_id}\n"]
    
    md.append("## Overview")
    md.append(f"| Metric | {run_a_id} (A) | {run_b_id} (B) |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| Steps | {len(gr_a)} | {len(gr_b)} |")
    md.append(f"| Events | {len(ev_a)} | {len(ev_b)} |")
    md.append("")

    if "loss" in plot_paths:
        md.append("## Loss Comparison")
        md.append(f"![Loss Comparison]({plot_paths['loss']})\n")

    if "grad_norm" in plot_paths:
        md.append("## Gradient Norm Comparison")
        md.append(f"![Grad Norm Comparison]({plot_paths['grad_norm']})\n")

    if "timing" in plot_paths:
        md.append("## Timing Comparison")
        md.append(f"![Timing Comparison]({plot_paths['timing']})\n")

    if ev_a and ev_b:
        md.append("## Event Counts Comparison")
        from .render import events_to_df
        df_a = events_to_df(ev_a)
        df_b = events_to_df(ev_b)
        cnts_a = df_a["name"].value_counts().to_dict()
        cnts_b = df_b["name"].value_counts().to_dict()
        all_names = sorted(set(cnts_a) | set(cnts_b))
        
        md.append("| QNode | Calls (A) | Calls (B) |")
        md.append("| :--- | :--- | :--- |")
        for name in all_names:
            md.append(f"| {name} | {cnts_a.get(name, 0)} | {cnts_b.get(name, 0)} |")
        md.append("")

    report_path = run_dir_a / "comparison_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    return report_path
