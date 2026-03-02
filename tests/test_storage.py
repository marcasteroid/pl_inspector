"""Tests for the file-based storage backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pl_inspector.models import ExecutionEvent, RunMeta, TrainingStepRecord
from pl_inspector.storage import RunStorage
from pl_inspector.utils import generate_run_id


def test_run_storage_creates_expected_files(tmp_path: Path) -> None:
    """RunStorage should create a per-run directory and core files."""

    base_dir = tmp_path
    run_id = generate_run_id()
    storage = RunStorage(base_dir=base_dir, run_id=run_id)

    # Write meta
    meta = RunMeta(run_id=run_id, device_name="default.qubit")
    storage.write_meta(meta)

    # Append an event
    event = ExecutionEvent(
        run_id=run_id,
        event_index=0,
        name="dummy",
    )
    storage.append_event(event)

    # Append a training step
    step = TrainingStepRecord(run_id=run_id, global_step=1, loss=0.1)
    storage.append_training_step(step)

    # Save snapshots
    arrays = {"snap0": np.arange(4).reshape(2, 2)}
    storage.save_snapshots(arrays)

    # Write a simple summary
    storage.write_summary({"num_events": 1})

    run_dir = storage.run_dir
    assert run_dir.is_dir()

    expected_files = {
        "meta.json",
        "events.jsonl",
        "gradients.jsonl",
        "snapshots.npz",
        "summary.json",
    }
    assert expected_files.issubset({p.name for p in run_dir.iterdir()})


def test_events_are_stored_as_json_lines(tmp_path: Path) -> None:
    """Events JSONL file should contain valid JSON per line."""

    base_dir = tmp_path
    run_id = generate_run_id()
    storage = RunStorage(base_dir=base_dir, run_id=run_id)

    for idx in range(3):
        ev = ExecutionEvent(run_id=run_id, event_index=idx, name=f"ev{idx}")
        storage.append_event(ev)

    events_path = storage.run_dir / "events.jsonl"
    assert events_path.is_file()

    with events_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert all(isinstance(obj, dict) for obj in parsed)

