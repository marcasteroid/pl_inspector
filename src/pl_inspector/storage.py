"""Persistence and storage abstractions for pl-inspector.

This module provides a lightweight, file-based storage backend designed for
local experimentation. Each run is stored in its own directory, with a
conservative, JSON- and NumPy-friendly layout:

    runs/<run_id>/
        meta.json
        events.jsonl
        gradients.jsonl
        snapshots.npz
        summary.json

Higher-level tracing / training code is responsible for deciding *what* to
store and *when*; this module only implements the primitives for writing data
to disk in a robust, JSON-safe way.
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .exceptions import StorageError
from .models import ExecutionEvent, RunMeta, SnapshotRecord, TrainingStepRecord
from .utils import JSONValue, dataclass_to_json_dict, ensure_dir, to_json_compatible


class RunStorage:
    """File-based storage backend for a single pl-inspector run.

    Parameters
    ----------
    base_dir:
        Root directory under which the ``runs/`` directory will be created.
    run_id:
        Identifier for the run, typically produced by
        :func:`pl_inspector.utils.generate_run_id`.
    """

    def __init__(self, base_dir: Path | str, run_id: str) -> None:
        self._base_dir = Path(base_dir)
        self._runs_root = ensure_dir(self._base_dir / "runs")
        self._run_id = run_id
        self._run_dir = ensure_dir(self._runs_root / run_id)

        self._meta_path = self._run_dir / "meta.json"
        self._events_path = self._run_dir / "events.jsonl"
        self._gradients_path = self._run_dir / "gradients.jsonl"
        self._snapshots_path = self._run_dir / "snapshots.npz"
        self._summary_path = self._run_dir / "summary.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """Return the identifier of the associated run."""

        return self._run_id

    @property
    def run_dir(self) -> Path:
        """Return the directory used to store this run."""

        return self._run_dir

    def write_meta(self, meta: RunMeta) -> None:
        """Write run metadata to ``meta.json``.

        This will overwrite any existing metadata file for the run.
        """

        payload = self._encode_payload(meta)
        try:
            with self._meta_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise StorageError(f"Failed to write meta.json for run {self._run_id}") from exc

    def append_event(self, event: ExecutionEvent) -> None:
        """Append an execution event to ``events.jsonl`` as a single JSON line."""

        record = self._encode_payload(event)
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._events_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            raise StorageError(f"Failed to append to events.jsonl for run {self._run_id}") from exc

    def append_training_step(self, record: TrainingStepRecord) -> None:
        """Append a training step record to ``gradients.jsonl``."""

        payload = self._encode_payload(record)
        line = json.dumps(payload, ensure_ascii=False)
        try:
            with self._gradients_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            raise StorageError(
                f"Failed to append to gradients.jsonl for run {self._run_id}"
            ) from exc

    def save_snapshots(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        append: bool = False,
    ) -> None:
        """Persist snapshot arrays to ``snapshots.npz``.

        Parameters
        ----------
        arrays:
            Mapping from snapshot labels (or other identifiers) to NumPy arrays.
        append:
            If ``True``, merge the provided arrays with any existing contents of
            ``snapshots.npz``. Existing keys will be overwritten. If ``False``,
            the file will be replaced.
        """

        try:
            data_to_save: Dict[str, np.ndarray]
            if append and self._snapshots_path.exists():
                # Load existing arrays to merge with new ones.
                with np.load(self._snapshots_path, allow_pickle=False) as existing:
                    data_to_save = {k: existing[k] for k in existing.files}
                data_to_save.update(arrays)
            else:
                data_to_save = dict(arrays)

            # Ensure all values are NumPy arrays.
            normalised: Dict[str, np.ndarray] = {}
            for key, value in data_to_save.items():
                if not isinstance(value, np.ndarray):
                    normalised[key] = np.asanyarray(value)
                else:
                    normalised[key] = value

            np.savez_compressed(self._snapshots_path, **normalised)
        except (OSError, ValueError) as exc:
            raise StorageError(
                f"Failed to save snapshots.npz for run {self._run_id}"
            ) from exc

    def write_summary(self, summary: Mapping[str, Any] | SnapshotRecord) -> None:
        """Write a high-level run summary to ``summary.json``.

        The summary object should already be in a compact, JSON-friendly form.
        A :class:`SnapshotRecord` may be passed for convenience and will be
        serialised via its ``to_json_dict`` method.
        """

        if isinstance(summary, SnapshotRecord):
            payload: Dict[str, JSONValue] = summary.to_json_dict()
        elif is_dataclass(summary):
            payload = dataclass_to_json_dict(summary)
        else:
            # Coerce mapping-like content into a JSON-safe dictionary.
            if not isinstance(summary, Mapping):
                raise StorageError(
                    "write_summary expects a mapping, dataclass, or SnapshotRecord."
                )
            payload = {str(k): to_json_compatible(v) for k, v in summary.items()}

        try:
            with self._summary_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise StorageError(f"Failed to write summary.json for run {self._run_id}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_payload(self, obj: Any) -> Dict[str, JSONValue]:
        """Convert a dataclass-based record into a JSON-safe dictionary."""

        if hasattr(obj, "to_json_dict") and callable(getattr(obj, "to_json_dict")):
            data: Dict[str, JSONValue] = obj.to_json_dict()  # type: ignore[assignment]
        elif is_dataclass(obj):
            data = dataclass_to_json_dict(obj)
        else:
            raise StorageError(
                f"Unsupported payload type for storage: {type(obj)!r}. "
                "Expected a dataclass or object with to_json_dict()."
            )
        return data


