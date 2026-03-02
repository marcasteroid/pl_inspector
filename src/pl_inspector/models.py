"""Datamodels and typed containers used across pl-inspector.

These dataclasses provide a lightweight, serialisable representation of the
core concepts in pl-inspector:

- runs and their metadata,
- execution events,
- tracker snapshots,
- PennyLane snapshots,
- training steps.

All dataclasses are designed to be converted into JSON-safe dictionaries via
the utilities in :mod:`pl_inspector.utils`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .utils import JSONValue, dataclass_to_json_dict, summarize_output, utc_now_iso


@dataclass(slots=True)
class RunMeta:
    """Metadata describing a single inspection run.

    Parameters
    ----------
    run_id:
        Stable identifier for the run (typically generated via
        :func:`pl_inspector.utils.generate_run_id`).
    created_at:
        ISO 8601 timestamp for when the run was created.
    pl_version:
        PennyLane version string, if available.
    device_name:
        Name of the PennyLane device used for the run.
    device_wires:
        Human-readable description of the device wires configuration.
    description:
        Optional free-form description of the run.
    tags:
        Optional free-form tags for later filtering.
    extra:
        Arbitrary structured metadata (must be JSON-serialisable or convertible
        by :func:`pl_inspector.utils.to_json_compatible`).
    """

    run_id: str
    created_at: str = field(default_factory=utc_now_iso)
    pl_version: Optional[str] = None
    device_name: Optional[str] = None
    device_wires: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, JSONValue] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, JSONValue]:
        """Return a JSON-safe dictionary representation of this run."""

        return dataclass_to_json_dict(self)


@dataclass(slots=True)
class ExecutionEvent:
    """A single execution event emitted during a run.

    This is intentionally generic so that it can describe operation-level
    events, qnode call events, or other custom events as needed.
    """

    run_id: str
    event_index: int
    timestamp: str = field(default_factory=utc_now_iso)
    category: str = "op"  # e.g. "op", "qnode", "callback"
    name: Optional[str] = None
    details: Dict[str, JSONValue] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, JSONValue]:
        """Return a JSON-safe dictionary representation of this event."""

        return dataclass_to_json_dict(self)


@dataclass(slots=True)
class TrackerRecord:
    """A snapshot of PennyLane tracker data for a given step or run.

    Parameters
    ----------
    run_id:
        Identifier of the run this tracker record belongs to.
    step_index:
        Optional logical step index within the run.
    timestamp:
        ISO 8601 timestamp indicating when this record was captured.
    metrics:
        High-level summary metrics derived from PennyLane's tracker history.
    raw_history:
        Optional raw tracker history mapping for advanced consumers.
    """

    run_id: str
    step_index: Optional[int] = None
    timestamp: str = field(default_factory=utc_now_iso)
    metrics: Dict[str, JSONValue] = field(default_factory=dict)
    raw_history: Optional[Dict[str, JSONValue]] = None

    def to_json_dict(self) -> Dict[str, JSONValue]:
        """Return a JSON-safe dictionary representation of this tracker record."""

        return dataclass_to_json_dict(self)


@dataclass(slots=True)
class SnapshotRecord:
    """Record representing a PennyLane snapshot.

    Parameters
    ----------
    run_id:
        Identifier of the run this snapshot belongs to.
    label:
        Snapshot label from PennyLane.
    timestamp:
        ISO 8601 timestamp when the snapshot was captured or processed.
    wires:
        Optional textual representation of the wires involved in the snapshot.
    data_summary:
        Compact JSON-safe summary of the snapshot payload.
    metadata:
        Additional contextual information associated with the snapshot.
    """

    run_id: str
    label: str
    timestamp: str = field(default_factory=utc_now_iso)
    wires: Optional[str] = None
    data_summary: Dict[str, JSONValue] = field(default_factory=dict)
    metadata: Dict[str, JSONValue] = field(default_factory=dict)

    @classmethod
    def from_raw_data(
        cls,
        run_id: str,
        label: str,
        data: Any,
        *,
        wires: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SnapshotRecord":
        """Create a :class:`SnapshotRecord` from raw snapshot data."""

        summary = summarize_output(data)
        meta_dict: Dict[str, JSONValue] = {}
        if metadata is not None:
            for k, v in metadata.items():
                meta_dict[str(k)] = summary if k == "data_summary" else dataclass_to_json_dict(
                    v
                ) if False else summary  # placeholder pattern, not used currently
            # Simpler: re-encode metadata directly
            meta_dict = {str(k): summarize_output(v) for k, v in metadata.items()}

        return cls(
            run_id=run_id,
            label=label,
            wires=wires,
            data_summary=summary,
            metadata=meta_dict,
        )

    def to_json_dict(self) -> Dict[str, JSONValue]:
        """Return a JSON-safe dictionary representation of this snapshot."""

        return dataclass_to_json_dict(self)


@dataclass(slots=True)
class TrainingStepRecord:
    """Record describing a single Torch training step.

    Parameters
    ----------
    run_id:
        Identifier of the run this training step belongs to.
    global_step:
        Monotonic global step index across the entire run.
    epoch:
        Epoch index, if applicable.
    batch_index:
        Batch index within the epoch, if applicable.
    loss:
        Scalar loss value for this step.
    metrics:
        Additional scalar or small structured metrics (e.g., accuracy).
    grad_norm:
        Optional gradient norm summary.
    timestamp:
        ISO 8601 timestamp indicating when this step was recorded.
    """

    run_id: str
    global_step: int
    epoch: Optional[int] = None
    batch_index: Optional[int] = None
    loss: Optional[float] = None
    metrics: Dict[str, JSONValue] = field(default_factory=dict)
    grad_norm: Optional[float] = None
    timestamp: str = field(default_factory=utc_now_iso)

    def to_json_dict(self) -> Dict[str, JSONValue]:
        """Return a JSON-safe dictionary representation of this training step."""

        return dataclass_to_json_dict(self)


