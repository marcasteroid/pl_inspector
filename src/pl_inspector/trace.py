"""Execution tracing primitives for pl-inspector.

This module provides a lightweight wrapper around PennyLane QNodes that:

- intercepts calls,
- records call timestamps,
- captures arguments and keyword arguments in a summarised form,
- measures execution duration,
- summarises outputs,
- extracts basic device and shot information when available,
- emits :class:`pl_inspector.models.ExecutionEvent` instances.

The wrapper does *not* modify PennyLane internals or perform any global
monkey-patching. Higher-level modules (e.g. ``watch``) can compose this
primitive with storage and context management.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Generic, Optional, Protocol, TypeVar

from .models import ExecutionEvent
from .utils import JSONValue, summarize_output, to_json_compatible, utc_now_iso


P = TypeVar("P")
R = TypeVar("R")


class _SupportsName(Protocol):
    """Minimal structural protocol capturing QNode-like attributes we care about."""

    __name__: str  # typical for Python callables


class _QNodeLike(Protocol):
    """Structural protocol for the subset of QNode attributes we inspect.

    This avoids importing PennyLane at type-check time while still giving some
    guidance to users of this module.
    """

    device: Any  # PennyLane device or similar
    shots: Any
    __call__: Callable[..., Any]


def _derive_qnode_name(qnode: Any) -> str:
    """Derive a readable name for a QNode-like callable."""

    if hasattr(qnode, "__name__"):
        return str(getattr(qnode, "__name__"))
    if hasattr(qnode, "name"):
        return str(getattr(qnode, "name"))
    return qnode.__class__.__name__


def _extract_device_info(qnode: Any) -> Dict[str, JSONValue]:
    """Extract basic device and shot information from a QNode, if available."""

    device = getattr(qnode, "device", None)
    device_name: Optional[str] = None
    device_wires: Optional[str] = None

    if device is not None:
        # Prefer a short, human-readable identifier where possible.
        if hasattr(device, "short_name"):
            device_name = str(getattr(device, "short_name"))
        elif hasattr(device, "name"):
            device_name = str(getattr(device, "name"))
        else:
            device_name = type(device).__name__

        wires = getattr(device, "wires", None)
        if wires is not None:
            device_wires = str(wires)

    # Shots may live on the QNode or the device, and can be various types.
    shots = getattr(qnode, "shots", None)
    if shots is None and device is not None:
        shots = getattr(device, "shots", None)

    total_shots: Optional[int] = None
    if shots is not None:
        if hasattr(shots, "total_shots"):
            # PennyLane Shots object
            total_shots = int(getattr(shots, "total_shots"))
        elif isinstance(shots, int):
            total_shots = shots
        else:
            # Last resort: try to coerce to int, otherwise fall back to string.
            try:
                total_shots = int(shots)
            except Exception:  # noqa: BLE001
                pass

    info: Dict[str, JSONValue] = {}
    if device_name is not None:
        info["device_name"] = device_name
    if device_wires is not None:
        info["device_wires"] = device_wires
    if total_shots is not None:
        info["shots"] = to_json_compatible(total_shots)

    return info


class QNodeTraceWrapper(Generic[P, R]):
    """Callable wrapper that traces QNode executions into :class:`ExecutionEvent`.

    Instances of this class are intended to be created via :func:`wrap_qnode`.
    """

    def __init__(
        self,
        qnode: Callable[..., R],
        *,
        run_id: str,
        on_event: Optional[Callable[[ExecutionEvent], None]] = None,
        name: Optional[str] = None,
    ) -> None:
        self._qnode = qnode
        self._run_id = run_id
        self._on_event = on_event
        self._name = name or _derive_qnode_name(qnode)
        self._event_index = 0

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        """Invoke the underlying QNode and emit an :class:`ExecutionEvent`."""

        timestamp = utc_now_iso()
        start_ns = time.perf_counter_ns()
        error: Optional[BaseException] = None
        output: Any = None

        try:
            output = self._qnode(*args, **kwargs)
            return output
        except BaseException as exc:  # noqa: BLE001
            error = exc
            raise
        finally:
            end_ns = time.perf_counter_ns()
            duration_ms = (end_ns - start_ns) / 1_000_000.0

            # Build argument summaries.
            arg_summaries = [summarize_output(a) for a in args]
            kwarg_summaries = {str(k): summarize_output(v) for k, v in kwargs.items()}
            output_summary: Dict[str, JSONValue] = summarize_output(output)

            details: Dict[str, JSONValue] = {
                "qnode_name": self._name,
                "duration_ms": to_json_compatible(duration_ms),
                "args": arg_summaries,
                "kwargs": kwarg_summaries,
                "output_summary": output_summary,
            }

            # Attach device information if accessible.
            details.update(_extract_device_info(self._qnode))

            # Attach error information if the call failed.
            if error is not None:
                details["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }

            event = ExecutionEvent(
                run_id=self._run_id,
                event_index=self._event_index,
                timestamp=timestamp,
                category="qnode",
                name=self._name,
                details=details,
            )
            self._event_index += 1

            if self._on_event is not None:
                self._on_event(event)


def wrap_qnode(
    qnode: Callable[..., R],
    *,
    run_id: str,
    on_event: Optional[Callable[[ExecutionEvent], None]] = None,
    name: Optional[str] = None,
) -> QNodeTraceWrapper[Any, R]:
    """Wrap a QNode-like callable to emit :class:`ExecutionEvent` on each call.

    Parameters
    ----------
    qnode:
        QNode-like callable to wrap. It is not modified in-place.
    run_id:
        Identifier for the logical run. Typically generated via
        :func:`pl_inspector.utils.generate_run_id`.
    on_event:
        Optional callback invoked with each :class:`ExecutionEvent` as it is
        produced, allowing callers to persist or further process events.
    name:
        Optional human-readable name to attach to events. If omitted, a
        best-effort name is derived from the callable.

    Returns
    -------
    QNodeTraceWrapper
        A callable object that behaves like the original QNode but emits
        structured execution events on each invocation.
    """

    return QNodeTraceWrapper(qnode, run_id=run_id, on_event=on_event, name=name)


