"""Tests for the QNode tracing wrapper."""

from __future__ import annotations

import json
from typing import List

import numpy as np

from pl_inspector.models import ExecutionEvent
from pl_inspector.trace import wrap_qnode
from pl_inspector.utils import generate_run_id, to_json_compatible


def test_wrap_qnode_emits_execution_events() -> None:
    """Wrapping a callable should emit structured ExecutionEvent objects."""

    def dummy_qnode(x: float, scale: float = 2.0) -> float:
        return float(x * scale)

    run_id = generate_run_id()
    events: List[ExecutionEvent] = []

    def on_event(ev: ExecutionEvent) -> None:
        events.append(ev)

    wrapped = wrap_qnode(dummy_qnode, run_id=run_id, on_event=on_event, name="dummy")

    out = wrapped(1.5, scale=3.0)

    assert out == 4.5
    assert len(events) == 1

    ev = events[0]
    assert ev.run_id == run_id
    assert ev.event_index == 0
    assert ev.category == "qnode"
    assert ev.name == "dummy"

    details = ev.details
    assert details.get("qnode_name") == "dummy"
    assert "duration_ms" in details
    assert details["args"][0]["value"] == 1.5
    assert details["kwargs"]["scale"]["value"] == 3.0
    assert details["output_summary"]["value"] == 4.5


def test_execution_event_details_are_json_safe() -> None:
    """Event details should be convertible to a JSON-safe structure."""

    def dummy_qnode(x: float) -> np.ndarray:
        return np.array([x, x + 1.0])

    run_id = generate_run_id()
    captured: list[ExecutionEvent] = []

    def on_event(ev: ExecutionEvent) -> None:
        captured.append(ev)

    wrapped = wrap_qnode(dummy_qnode, run_id=run_id, on_event=on_event)
    _ = wrapped(0.5)

    assert captured, "Expected at least one event"
    ev = captured[0]

    # Ensure that we can serialise the event via the helper.
    payload = to_json_compatible(ev.to_json_dict())
    json.dumps(payload)

