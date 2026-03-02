"""Integration helpers around :class:`pennylane.Tracker`.

This module provides a thin, testable adapter around PennyLane's tracking
mechanism. It is intentionally small and focused so that higher-level
components (such as a future ``watch()`` context manager) can depend on it
without pulling in unnecessary complexity.

Key responsibilities:

- start and stop a :class:`pennylane.Tracker` context cleanly,
- normalise tracker data into a serialisable :class:`TrackerRecord`,
- extract ``latest``, ``history``, and ``totals`` where available,
- degrade gracefully when tracking is unavailable (e.g. PennyLane missing).
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict, Mapping, Optional

from .models import TrackerRecord
from .utils import JSONValue, to_json_compatible


class TrackerAdapter:
    """Thin adapter around :class:`pennylane.Tracker`.

    The adapter is a context manager that, when entered, attempts to start a
    PennyLane tracker for the given target (typically a device or QNode). When
    exited, it can be queried for a :class:`TrackerRecord` that captures a
    JSON-serialisable view of the tracker state.

    The adapter does *not* raise if tracking is unavailable; instead, it
    records this fact in the resulting :class:`TrackerRecord`.
    """

    def __init__(
        self,
        run_id: str,
        *,
        step_index: Optional[int] = None,
        target: Any | None = None,
    ) -> None:
        """Create a new tracker adapter.

        Parameters
        ----------
        run_id:
            Logical run identifier used to populate the resulting
            :class:`TrackerRecord`.
        step_index:
            Optional logical step index within the run.
        target:
            Optional PennyLane device or QNode to attach the tracker to. If
            omitted, the behaviour is delegated to :class:`pennylane.Tracker`
            without arguments.
        """

        self._run_id = run_id
        self._step_index = step_index
        self._target = target

        self._tracker: Any | None = None
        self._tracking_enabled: bool = False
        self._error_message: Optional[str] = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "TrackerAdapter":
        """Enter the tracking context, if possible.

        Failures to import PennyLane or instantiate the underlying tracker are
        recorded and surfaced via :meth:`to_record` instead of raising.
        """

        try:
            import pennylane as qml  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._tracking_enabled = False
            self._error_message = f"pennylane not available: {exc}"
            self._tracker = None
            return self

        try:
            if self._target is not None:
                tracker = qml.Tracker(self._target)
            else:
                tracker = qml.Tracker()
            self._tracker = tracker
            tracker.__enter__()
            self._tracking_enabled = True
        except Exception as exc:  # noqa: BLE001
            self._tracking_enabled = False
            self._error_message = f"failed to start Tracker: {exc}"
            self._tracker = None

        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        """Exit the tracking context if it was started successfully."""

        if self._tracker is not None:
            # Best-effort shutdown; errors here should not mask the original.
            try:
                self._tracker.__exit__(exc_type, exc, tb)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_record(self) -> TrackerRecord:
        """Return a :class:`TrackerRecord` representing the tracked data.

        If tracking was unavailable or failed to start, the resulting record
        will contain a ``tracking_available=False`` flag and a human-readable
        ``tracking_error`` message in ``metrics``.
        """

        metrics: Dict[str, JSONValue] = {}
        raw_history: Optional[Dict[str, JSONValue]] = None

        if not self._tracking_enabled or self._tracker is None:
            metrics["tracking_available"] = False
            if self._error_message is not None:
                metrics["tracking_error"] = self._error_message
            return TrackerRecord(
                run_id=self._run_id,
                step_index=self._step_index,
                metrics=metrics,
                raw_history=raw_history,
            )

        # Extract attributes defensively; some versions / configurations may
        # omit certain fields.
        latest: Mapping[str, Any] = getattr(self._tracker, "latest", {}) or {}
        totals: Mapping[str, Any] = getattr(self._tracker, "totals", {}) or {}
        history: Mapping[str, Any] = getattr(self._tracker, "history", {}) or {}

        metrics["tracking_available"] = True
        metrics["latest"] = to_json_compatible(latest)
        metrics["totals"] = to_json_compatible(totals)

        # Preserve history separately so consumers can choose how much detail
        # they need.
        raw_history = {
            str(key): to_json_compatible(value) for key, value in history.items()
        }

        return TrackerRecord(
            run_id=self._run_id,
            step_index=self._step_index,
            metrics=metrics,
            raw_history=raw_history,
        )


