"""High-level watching utilities for PennyLane QNodes.

This module exposes the public :func:`watch` context manager, which provides a
minimal but usable MVP for tracing PennyLane QNodes and persisting basic run
information to disk.

Example
-------

.. code-block:: python

    import pennylane as qml
    from pl_inspector import watch

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    with watch(circuit, save_dir="./runs", capture="tracker") as run:
        y1 = run(0.1)          # calls the wrapped QNode
        y2 = run(0.2)
        run.show_trace()
        run_dir = run.export_json()

The original QNode is *not* modified in-place; instead, the session behaves
like the wrapped QNode (it is callable) and also exposes the underlying
``wrapped_qnode`` attribute for users who prefer that style.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import ExecutionEvent, RunMeta, TrackerRecord
from .storage import RunStorage
from .trace import QNodeTraceWrapper, wrap_qnode
from .tracker import TrackerAdapter
from .utils import JSONValue, ensure_dir, generate_run_id, to_json_compatible


class WatchSession:
    """Context-managed run/session object returned by :func:`watch`.

    Instances of :class:`WatchSession` are themselves callable and delegate
    calls to the wrapped QNode, making this pattern convenient:

    .. code-block:: python

        with watch(circuit) as run:
            y = run(0.5)
    """

    def __init__(
        self,
        qnode: Callable[..., Any],
        *,
        save_dir: str | Path = "./runs",
        capture: Optional[str] = "tracker",
    ) -> None:
        self._original_qnode = qnode
        self._save_dir = Path(save_dir)
        ensure_dir(self._save_dir)
        self._capture = capture

        self.run_id: str = generate_run_id()
        self.storage: RunStorage = RunStorage(base_dir=self._save_dir, run_id=self.run_id)

        self.wrapped_qnode: QNodeTraceWrapper[Any, Any] | None = None
        self._events: List[ExecutionEvent] = []
        self._tracker_adapter: Optional[TrackerAdapter] = None
        self._tracker_record: Optional[TrackerRecord] = None
        self._summary_written: bool = False

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "WatchSession":
        """Initialise storage, tracking, and the wrapped QNode."""

        # Derive basic device information for metadata.
        device = getattr(self._original_qnode, "device", None)
        device_name: Optional[str] = None
        device_wires: Optional[str] = None

        if device is not None:
            if hasattr(device, "short_name"):
                device_name = str(getattr(device, "short_name"))
            elif hasattr(device, "name"):
                device_name = str(getattr(device, "name"))
            else:
                device_name = type(device).__name__

            wires = getattr(device, "wires", None)
            if wires is not None:
                device_wires = str(wires)

        meta = RunMeta(
            run_id=self.run_id,
            device_name=device_name,
            device_wires=device_wires,
            description=f"watch() session for {self._original_qnode!r}",
        )
        self.storage.write_meta(meta)

        # Optional tracker capture.
        if self._capture == "tracker":
            self._tracker_adapter = TrackerAdapter(run_id=self.run_id, target=device or self._original_qnode)
            self._tracker_adapter.__enter__()

        # Prepare the wrapped QNode with an event callback that both records
        # events in memory and appends them to storage.
        def on_event(ev: ExecutionEvent) -> None:
            self._events.append(ev)
            self.storage.append_event(ev)

        self.wrapped_qnode = wrap_qnode(
            self._original_qnode,
            run_id=self.run_id,
            on_event=on_event,
        )

        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped QNode directly via the session object.

        This preserves a pleasant API where users can write ``run(x)`` inside
        the context block, while still allowing access to ``wrapped_qnode`` if
        they need the underlying callable explicitly.
        """

        if self.wrapped_qnode is None:
            raise RuntimeError(
                "WatchSession is not initialised. "
                "Use it as a context manager: 'with watch(qnode) as run: ...'."
            )
        return self.wrapped_qnode(*args, **kwargs)

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        """Tear down tracking and write a final summary."""

        # Close tracker context if it was created.
        if self._tracker_adapter is not None:
            try:
                self._tracker_adapter.__exit__(exc_type, exc, tb)
            finally:
                self._tracker_record = self._tracker_adapter.to_record()

        # Always attempt to write a summary, even if an exception occurred.
        self._write_summary_if_needed()

    # ------------------------------------------------------------------
    # User-facing helpers
    # ------------------------------------------------------------------

    def show_trace(self) -> None:
        """Print a readable console summary of the recorded execution events."""

        if not self._events:
            print(f"[pl-inspector] No events recorded for run {self.run_id}.")
            return

        print(f"[pl-inspector] Execution trace for run {self.run_id}:")
        for ev in self._events:
            details = ev.details
            name = details.get("qnode_name", ev.name or "<unnamed>")
            duration = details.get("duration_ms", None)
            duration_str = f"{duration:.3f} ms" if isinstance(duration, (int, float)) else str(
                duration
            )
            print(
                f"  #{ev.event_index:03d} "
                f"{ev.timestamp} "
                f"{name} "
                f"duration={duration_str}"
            )

    def export_json(self) -> Path:
        """Ensure run summary is written and return the run directory path."""

        self._write_summary_if_needed()
        return self.storage.run_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_summary_if_needed(self) -> None:
        if self._summary_written:
            return

        summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "num_events": len(self._events),
        }

        if self._events:
            first = self._events[0]
            last = self._events[-1]
            summary["first_timestamp"] = first.timestamp
            summary["last_timestamp"] = last.timestamp

        if self._tracker_record is not None:
            summary["tracker"] = self._tracker_record.to_json_dict()

        # storage.write_summary will convert the mapping to JSON-safe values.
        self.storage.write_summary(summary)
        self._summary_written = True


def watch(
    qnode: Callable[..., Any],
    *,
    save_dir: str | Path = "./runs",
    capture: Optional[str] = "tracker",
) -> WatchSession:
    """Create a new :class:`WatchSession` for the given QNode.

    Parameters
    ----------
    qnode:
        PennyLane QNode to be traced. It is not modified in-place; use the
        ``wrapped_qnode`` attribute of the yielded session inside the context.
    save_dir:
        Base directory under which run data will be stored. A subdirectory
        structure is created by :class:`pl_inspector.storage.RunStorage`.
    capture:
        Optional capture mode. Currently ``"tracker"`` enables
        :class:`pennylane.Tracker` integration; other values are ignored for
        now.
    """

    return WatchSession(qnode, save_dir=save_dir, capture=capture)


