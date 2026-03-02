"""Snapshot-based intermediate state capture utilities.

This module provides helpers for working with snapshot outputs produced by
PennyLane's snapshot-enabled execution modes. It focuses on *parsing* and
*summarising* snapshot payloads; deciding where and how to insert snapshots
into circuits is left to higher-level code.

Key responsibilities:

- detect when a snapshot payload looks like a state vector,
- summarise such state vectors (e.g. top-k amplitudes and probabilities),
- fall back to generic summaries for other payload types,
- produce :class:`pl_inspector.models.SnapshotRecord` instances.

The helpers here are intentionally conservative and defensive: if a device
does not expose state information, or the payload is not easily interpreted as
state, we still generate a :class:`SnapshotRecord` with a generic summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .models import SnapshotRecord
from .utils import JSONValue, summarize_output, to_json_compatible


def is_state_vector_like(obj: Any) -> bool:
    """Return ``True`` if ``obj`` looks like a state vector.

    Heuristics:

    - :class:`numpy.ndarray` with 1D shape,
    - length is a power of 2,
    - dtype is complex or float-like.
    """

    if not isinstance(obj, np.ndarray):
        return False
    if obj.ndim != 1:
        return False
    dim = obj.shape[0]
    if dim <= 0 or (dim & (dim - 1)) != 0:
        return False  # not a power of two
    if not np.issubdtype(obj.dtype, np.complexfloating) and not np.issubdtype(
        obj.dtype, np.floating
    ):
        return False
    return True


def _index_to_bitstring(index: int, num_qubits: int) -> str:
    """Convert a basis index to a zero-padded bitstring."""

    return format(index, f"0{num_qubits}b")


def summarize_state_vector(
    state: np.ndarray,
    *,
    top_k: int = 8,
) -> Dict[str, JSONValue]:
    """Summarise a state vector into a compact, JSON-safe structure.

    Parameters
    ----------
    state:
        1D NumPy array representing the state vector (complex or real).
    top_k:
        Number of largest-probability amplitudes to include in the summary.
    """

    if not is_state_vector_like(state):
        return summarize_output(state)

    # Ensure complex form for probability computation.
    complex_state = state.astype(np.complex128, copy=False)
    probs = np.abs(complex_state) ** 2
    dim = complex_state.shape[0]

    # Normalise probabilities defensively.
    total = probs.sum()
    if total > 0:
        probs = probs / total

    num_qubits = int(np.log2(dim))
    # Indices sorted by descending probability.
    order = np.argsort(probs)[::-1]
    top_indices = order[:top_k]

    top_entries = []
    for idx in top_indices:
        p = float(probs[idx])
        if p == 0.0:
            continue
        amp = complex_state[idx]
        entry: Dict[str, JSONValue] = {
            "index": int(idx),
            "bitstring": _index_to_bitstring(int(idx), num_qubits),
            "amplitude_real": float(amp.real),
            "amplitude_imag": float(amp.imag),
            "probability": p,
        }
        top_entries.append(entry)

    summary: Dict[str, JSONValue] = {
        "type": "state_vector",
        "num_qubits": num_qubits,
        "dimension": dim,
        "top_amplitudes": top_entries,
    }

    if top_entries:
        max_prob = max(e["probability"] for e in top_entries if "probability" in e)
        summary["max_probability"] = to_json_compatible(max_prob)

    return summary


def make_snapshot_record(
    run_id: str,
    label: str,
    data: Any,
    *,
    wires: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    top_k: int = 8,
) -> SnapshotRecord:
    """Create a :class:`SnapshotRecord` from raw snapshot data.

    This helper routes state-vector-like payloads through
    :func:`summarize_state_vector` and falls back to
    :func:`pl_inspector.utils.summarize_output` for other types.
    """

    if isinstance(data, np.ndarray) and is_state_vector_like(data):
        data_summary = summarize_state_vector(data, top_k=top_k)
    else:
        data_summary = summarize_output(data)

    meta_dict: Dict[str, JSONValue] = {}
    if metadata is not None:
        for k, v in metadata.items():
            meta_dict[str(k)] = summarize_output(v)

    return SnapshotRecord(
        run_id=run_id,
        label=label,
        wires=wires,
        data_summary=data_summary,
        metadata=meta_dict,
    )


