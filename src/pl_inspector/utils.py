"""Shared utilities and helpers for pl-inspector.

This module focuses on:
- ID and timestamp generation,
- filesystem convenience helpers,
- conversion of rich Python / NumPy objects into JSON-safe values,
- small summarisation helpers suitable for logging and storage.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, MutableSequence, Sequence, Union

import numpy as np


JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, Dict[str, "JSONValue"], List["JSONValue"]]


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""

    return datetime.now(timezone.utc).isoformat()


def generate_run_id(prefix: str = "run") -> str:
    """Generate a simple run identifier based on the current timestamp.

    The identifier is intentionally human-readable and monotonic with respect to
    wall-clock time; it is not intended to be cryptographically secure.
    """

    # Example: run-20260302T120304Z
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure that a directory exists and return it as a :class:`Path`.

    Parameters
    ----------
    path:
        Directory path to ensure.
    """

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _numpy_to_python(value: Any) -> Any:
    """Convert NumPy scalars and arrays into plain Python types."""

    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        # Convert to nested lists; callers can choose to summarise further.
        return value.tolist()
    return value


def to_json_compatible(value: Any) -> JSONValue:
    """Convert an arbitrary Python object into a JSON-safe representation.

    The conversion is conservative:

    - Basic primitives are passed through unchanged.
    - NumPy scalars / arrays are converted to Python scalars / nested lists.
    - Dataclasses are converted via :func:`dataclasses.asdict`.
    - Mappings and sequences are converted recursively.
    - :class:`datetime` objects are converted to ISO 8601 strings.
    - Unsupported types are stringified as a last resort.
    """

    # Primitives
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Datetime-like
    if isinstance(value, datetime):
        # Preserve timezone information if present.
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    # Dataclasses
    if is_dataclass(value):
        return dataclass_to_json_dict(value)

    # NumPy
    value = _numpy_to_python(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Mapping
    if isinstance(value, Mapping):
        out: MutableMapping[str, JSONValue] = {}
        for k, v in value.items():
            out[str(k)] = to_json_compatible(v)
        return dict(out)

    # Sequence (but not string which was handled above)
    if isinstance(value, (list, tuple, set, frozenset, Iterable)):
        out_seq: MutableSequence[JSONValue] = []
        for item in value:
            out_seq.append(to_json_compatible(item))
        return list(out_seq)

    # Fallback – string representation.
    return str(value)


def dataclass_to_json_dict(instance: Any) -> Dict[str, JSONValue]:
    """Convert a dataclass instance into a JSON-safe dictionary.

    Parameters
    ----------
    instance:
        Dataclass instance to convert.
    """

    if not is_dataclass(instance):
        raise TypeError("dataclass_to_json_dict expects a dataclass instance.")

    raw = asdict(instance)
    return {str(k): to_json_compatible(v) for k, v in raw.items()}


def summarize_output(value: Any, max_list_elements: int = 3) -> Dict[str, JSONValue]:
    """Summarise an output object into a compact, JSON-safe schema.

    This helper is intended for logging and storage, not for full-fidelity
    reconstruction of the original object.
    """

    # NumPy arrays
    if isinstance(value, np.ndarray):
        preview = value.ravel()[:max_list_elements].tolist()
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": to_json_compatible(preview),
        }

    # NumPy scalar
    if isinstance(value, np.generic):
        py = value.item()
        return {
            "type": "numpy_scalar",
            "value": to_json_compatible(py),
        }

    # Simple primitives
    if value is None or isinstance(value, (str, int, float, bool)):
        return {
            "type": type(value).__name__,
            "value": to_json_compatible(value),
        }

    # Sequences
    if isinstance(value, (list, tuple)):
        preview_seq = list(value[:max_list_elements])
        return {
            "type": type(value).__name__,
            "length": len(value),
            "preview": to_json_compatible(preview_seq),
        }

    # Mappings
    if isinstance(value, Mapping):
        keys = list(value.keys())
        preview_items = list(keys[:max_list_elements])
        return {
            "type": type(value).__name__,
            "size": len(value),
            "preview_keys": [str(k) for k in preview_items],
        }

    # Fallback: store a string representation.
    return {
        "type": type(value).__name__,
        "repr": str(value),
    }


