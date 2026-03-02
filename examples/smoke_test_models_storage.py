"""
Smoke test for pl-inspector core data models and storage backend.

This script exercises:
- dataclass construction and JSON-safe serialisation,
- RunStorage file layout and basic write operations.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np

import pl_inspector
from pl_inspector.models import (
    ExecutionEvent,
    RunMeta,
    SnapshotRecord,
    TrackerRecord,
    TrainingStepRecord,
)
from pl_inspector.storage import RunStorage
from pl_inspector.utils import generate_run_id, summarize_output, to_json_compatible


def _assert_json_serialisable(obj: Any) -> None:
    """Raise if `obj` cannot be serialised by json.dumps after conversion."""
    try:
        json.dumps(to_json_compatible(obj))
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"Object is not JSON-serialisable: {obj!r}") from exc


def smoke_test_models() -> Dict[str, Any]:
    """Construct core models and verify JSON-safe serialisation."""
    run_id = generate_run_id()

    # Run meta
    meta = RunMeta(
        run_id=run_id,
        pl_version="0.0.0-test",
        device_name="default.qubit",
        description="Smoke test run",
        tags=["smoke", "test"],
    )
    meta_json = meta.to_json_dict()
    _assert_json_serialisable(meta_json)

    # Execution event
    event = ExecutionEvent(
        run_id=run_id,
        event_index=0,
        category="qnode",
        name="example_qnode",
        details={"shots": 1000, "depth": 3},
    )
    event_json = event.to_json_dict()
    _assert_json_serialisable(event_json)

    # Tracker record
    tracker = TrackerRecord(
        run_id=run_id,
        step_index=0,
        metrics={"num_executions": 1, "max_depth": 3},
    )
    tracker_json = tracker.to_json_dict()
    _assert_json_serialisable(tracker_json)

    # Snapshot record
    snapshot_array = np.arange(6).reshape(2, 3)
    snapshot = SnapshotRecord.from_raw_data(
        run_id=run_id,
        label="snap0",
        data=snapshot_array,
        metadata={"note": "example snapshot"},
    )
    snapshot_json = snapshot.to_json_dict()
    _assert_json_serialisable(snapshot_json)

    # Training step
    step = TrainingStepRecord(
        run_id=run_id,
        global_step=1,
        epoch=0,
        batch_index=0,
        loss=0.123,
        metrics={"accuracy": 0.9},
        grad_norm=1.5,
    )
    step_json = step.to_json_dict()
    _assert_json_serialisable(step_json)

    # Also test summarize_output directly
    summary = summarize_output(snapshot_array)
    _assert_json_serialisable(summary)

    return {
        "run_id": run_id,
        "meta": meta_json,
        "event": event_json,
        "tracker": tracker_json,
        "snapshot": snapshot_json,
        "step": step_json,
        "array_summary": summary,
    }


def smoke_test_storage(base_dir: Path, run_id: str) -> Path:
    """Exercise RunStorage by writing all core files and re-reading them."""
    storage = RunStorage(base_dir, run_id)

    # Use the same models as in the model smoke test
    meta = RunMeta(run_id=run_id, device_name="default.qubit")
    storage.write_meta(meta)

    event = ExecutionEvent(run_id=run_id, event_index=0, name="qnode_call")
    storage.append_event(event)

    step = TrainingStepRecord(run_id=run_id, global_step=1, loss=0.456)
    storage.append_training_step(step)

    # Snapshots
    arrays = {"snap0": np.arange(4).reshape(2, 2), "snap1": np.ones((1,))}
    storage.save_snapshots(arrays)

    # Summary
    summary_payload = {
        "num_events": 1,
        "num_steps": 1,
        "notes": "storage smoke test",
    }
    storage.write_summary(summary_payload)

    # Basic integrity checks: files exist and are parseable
    run_dir = storage.run_dir
    assert (run_dir / "meta.json").is_file()
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "gradients.jsonl").is_file()
    assert (run_dir / "snapshots.npz").is_file()
    assert (run_dir / "summary.json").is_file()

    # Read back a few things
    with (run_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta_loaded = json.load(f)
    _assert_json_serialisable(meta_loaded)

    with (run_dir / "events.jsonl").open("r", encoding="utf-8") as f:
        first_event_line = f.readline().strip()
    json.loads(first_event_line)

    with np.load(run_dir / "snapshots.npz", allow_pickle=False) as npz:
        assert set(npz.files) == {"snap0", "snap1"}

    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary_loaded = json.load(f)
    _assert_json_serialisable(summary_loaded)

    return run_dir


def main() -> None:
    print(f"pl_inspector version: {getattr(pl_inspector, '__version__', 'unknown')}")

    print("Running model smoke test...")
    model_info = smoke_test_models()
    print("  Run ID:", model_info["run_id"])
    print("  Meta keys:", sorted(model_info["meta"].keys()))
    print("  Event keys:", sorted(model_info["event"].keys()))
    print("  Training step keys:", sorted(model_info["step"].keys()))

    print("Running storage smoke test...")
    tmp_root = Path("smoke_test_runs")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        run_dir = smoke_test_storage(tmp_root, model_info["run_id"])
        print("  Storage run directory:", run_dir)
        print("  Contents:")
        for p in sorted(run_dir.iterdir()):
            print("   -", p.name)
        print("Smoke test completed successfully.")
    finally:
        # Comment this out if you want to inspect files after the run.
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()