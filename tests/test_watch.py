"""Tests for the high-level watch() context manager."""

from __future__ import annotations

from pathlib import Path

import pennylane as qml

from pl_inspector import watch


def _make_test_qnode() -> qml.QNode:
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x: float) -> float:
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    return circuit


def test_watch_context_basic_flow(tmp_path: Path) -> None:
    """watch() should trace calls, store files, and export a valid run path."""

    circuit = _make_test_qnode()

    with watch(circuit, save_dir=tmp_path, capture="tracker") as run:
        y1 = run(0.1)
        y2 = run(0.2)

        # QNode should behave normally.
        assert isinstance(y1, float)
        assert isinstance(y2, float)

        # show_trace should not crash.
        run.show_trace()

        run_dir = run.export_json()

    # After the context exits, the run directory and expected files should exist.
    assert run_dir.is_dir()

    # For the MVP, the watch() flow is only responsible for meta, events, and
    # a summary. Gradients and snapshots are written by higher-level callers.
    expected_files = {"meta.json", "events.jsonl", "summary.json"}
    present = {p.name for p in run_dir.iterdir()}
    assert expected_files.issubset(present)

