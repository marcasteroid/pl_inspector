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

Snapshot capture
----------------

If you pass ``capture="snapshots"``, pl-inspector will **not** inject snapshots
into your circuit. Instead, it will try to **parse snapshot payloads that your
QNode already returns** (for example via PennyLane's snapshot-enabled
execution). If no snapshot payload is present in an output, the session will
record that fact and continue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .models import ExecutionEvent, RunMeta, TrackerRecord
from .models import SnapshotRecord
from .snapshots import make_snapshot_record
from .storage import RunStorage
from .trace import QNodeTraceWrapper, wrap_qnode
from .tracker import TrackerAdapter
from .utils import ensure_dir, generate_run_id


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
        self._snapshot_records: List[SnapshotRecord] = []
        self._snapshot_calls_with_payload: int = 0
        self._snapshot_calls_without_payload: int = 0
        self._snapshot_arrays_saved: int = 0
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
        # events in memory and appends them to storage. For snapshot capture,
        # we additionally parse the *returned* output for snapshot payloads.

        captured_outputs: List[Any] = []

        class _OutputCapturingCallable:
            """Proxy callable that captures outputs in call order.

            This is used for ``capture="snapshots"`` so we can parse the raw
            output alongside the generated :class:`ExecutionEvent` without
            modifying PennyLane internals.
            """

            def __init__(self, inner: Callable[..., Any]) -> None:
                self._inner = inner
                self.device = getattr(inner, "device", None)
                self.shots = getattr(inner, "shots", None)
                self.__name__ = getattr(inner, "__name__", inner.__class__.__name__)

            def __call__(self, *a: Any, **kw: Any) -> Any:
                out = self._inner(*a, **kw)
                captured_outputs.append(out)
                return out

        qnode_for_wrap: Callable[..., Any]
        if self._capture == "snapshots":
            qnode_for_wrap = _OutputCapturingCallable(self._original_qnode)
        else:
            qnode_for_wrap = self._original_qnode

        def on_event(ev: ExecutionEvent) -> None:
            # Attach snapshot parsing results (if requested) before persisting.
            if self._capture == "snapshots":
                raw_out: Any = captured_outputs.pop(0) if captured_outputs else None
                snapshots_payload = _extract_snapshot_payload(raw_out)
                if snapshots_payload is None:
                    self._snapshot_calls_without_payload += 1
                    ev.details["snapshots"] = {"found": False}
                else:
                    self._snapshot_calls_with_payload += 1
                    records, arrays_to_save = _snapshot_payload_to_records(
                        run_id=self.run_id,
                        call_index=ev.event_index,
                        payload=snapshots_payload,
                    )
                    self._snapshot_records.extend(records)

                    if arrays_to_save:
                        self.storage.save_snapshots(arrays_to_save, append=True)
                        self._snapshot_arrays_saved += len(arrays_to_save)

                    ev.details["snapshots"] = {
                        "found": True,
                        "num_entries": len(records),
                        "saved_arrays": len(arrays_to_save),
                    }

            self._events.append(ev)
            self.storage.append_event(ev)

        self.wrapped_qnode = wrap_qnode(
            qnode_for_wrap,
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

    def plot_probabilities(self, ax: Optional[Any] = None) -> Any:
        """Plot a probability bar chart from the latest snapshot data.
        
        Args:
            ax: Optional matplotlib Axes object to plot on.
            
        Returns:
            The matplotlib Axes object containing the plot.
        """
        if not self._snapshot_records:
            raise RuntimeError(
                "No snapshot data available to plot. "
                "Ensure the session was run with capture='snapshots'."
            )
            
        from .plotting import plot_probability_bar_chart
        import numpy as np
        
        npz_path = self.storage.run_dir / "snapshots.npz"
        if not npz_path.exists():
            raise RuntimeError("Snapshot array data not found on disk.")
            
        with np.load(npz_path, allow_pickle=False) as data:
            keys = sorted(data.files, key=lambda x: int(x.split('_')[0][4:]))
            if not keys:
                raise RuntimeError("No snapshot arrays found to plot.")
            
            prob_keys = [k for k in keys if "prob" in k.lower()]
            target_key = prob_keys[-1] if prob_keys else keys[-1]
            probs = data[target_key]
            
        return plot_probability_bar_chart(probs, ax=ax)

    def plot_top_amplitudes(self, top_k: int = 8, ax: Optional[Any] = None) -> Any:
        """Plot a bar chart of the top-k amplitude magnitudes from the latest snapshot data.
        
        Args:
            top_k: The number of highest amplitudes to display.
            ax: Optional matplotlib Axes object to plot on.
            
        Returns:
            The matplotlib Axes object containing the plot.
        """
        if not self._snapshot_records:
            raise RuntimeError(
                "No snapshot data available to plot. "
                "Ensure the session was run with capture='snapshots'."
            )
            
        from .plotting import plot_top_amplitude_magnitudes
        import numpy as np
        
        npz_path = self.storage.run_dir / "snapshots.npz"
        if not npz_path.exists():
            raise RuntimeError("Snapshot array data not found on disk.")
            
        with np.load(npz_path, allow_pickle=False) as data:
            keys = sorted(data.files, key=lambda x: int(x.split('_')[0][4:]))
            if not keys:
                raise RuntimeError("No snapshot arrays found to plot.")
            
            amp_keys = [k for k in keys if "amp" in k.lower() or "state" in k.lower()]
            target_key = amp_keys[-1] if amp_keys else keys[-1]
            amplitudes = data[target_key]
            
        return plot_top_amplitude_magnitudes(amplitudes, top_k=top_k, ax=ax)

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

        if self._capture == "snapshots":
            # Minimal snapshot summary (records already contain compact summaries).
            summary["snapshots"] = {
                "enabled": True,
                "calls_with_payload": self._snapshot_calls_with_payload,
                "calls_without_payload": self._snapshot_calls_without_payload,
                "num_records": len(self._snapshot_records),
                "arrays_saved": self._snapshot_arrays_saved,
                # Keep this bounded to avoid huge summary files.
                "records_preview": [r.to_json_dict() for r in self._snapshot_records[:50]],
            }

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
        Optional capture mode:

        - ``"tracker"``: best-effort capture of :class:`pennylane.Tracker` data.
        - ``"snapshots"``: parse snapshot payloads already returned by the QNode.

        In snapshot mode, pl-inspector does not insert snapshots into user
        circuits; it only parses and persists snapshot-capable outputs.
    """

    return WatchSession(qnode, save_dir=save_dir, capture=capture)


def _extract_snapshot_payload(output: Any) -> Optional[Mapping[str, Any]]:
    """Best-effort extraction of a snapshot payload from a QNode output.

    This function intentionally uses conservative heuristics. If no snapshot
    payload is detected, it returns ``None``.
    """

    if output is None:
        return None

    # Common pattern: (result, snapshots_dict)
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        maybe_payload = output[-1]
        if isinstance(maybe_payload, Mapping):
            return maybe_payload

    # Pattern: {"result": ..., "snapshots": {...}}
    if isinstance(output, Mapping) and "snapshots" in output:
        maybe_payload = output.get("snapshots")
        if isinstance(maybe_payload, Mapping):
            return maybe_payload

    # Some objects may expose a `.snapshots` attribute.
    maybe_attr = getattr(output, "snapshots", None)
    if isinstance(maybe_attr, Mapping):
        return maybe_attr

    # Last resort: if it's a mapping and keys look snapshot-ish, treat it as payload.
    if isinstance(output, Mapping):
        keys = list(output.keys())
        key_hint = any(
            isinstance(k, str) and ("snap" in k.lower() or "snapshot" in k.lower()) for k in keys
        )
        if key_hint:
            return output  # type: ignore[return-value]

    return None


def _snapshot_payload_to_records(
    *,
    run_id: str,
    call_index: int,
    payload: Mapping[str, Any],
) -> Tuple[List[SnapshotRecord], Dict[str, np.ndarray]]:
    """Convert a snapshot payload mapping into records and arrays to persist."""

    records: List[SnapshotRecord] = []
    arrays_to_save: Dict[str, np.ndarray] = {}

    for label, data in payload.items():
        label_str = str(label)
        record = make_snapshot_record(
            run_id=run_id,
            label=label_str,
            data=data,
            metadata={"call_index": call_index},
        )
        records.append(record)

        # Persist array-like payloads into snapshots.npz when safe.
        try:
            arr = data if isinstance(data, np.ndarray) else np.asanyarray(data)
        except Exception:  # noqa: BLE001
            continue

        # Avoid persisting object arrays (which imply pickling on load).
        if isinstance(arr, np.ndarray) and arr.dtype != object:
            key = f"call{call_index}_{label_str}"
            arrays_to_save[key] = arr

    return records, arrays_to_save


