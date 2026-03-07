"""Minimal CLI entry point for pl-inspector.

Usage:
    python -m pl_inspector path/to/run_dir

This reads meta.json and summary.json and prints a human-readable overview.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m pl_inspector <path_to_run_dir>")
        sys.exit(1 if len(sys.argv) != 2 else 0)

    run_dir = Path(sys.argv[1])
    
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a valid directory.")
        sys.exit(1)

    meta_file = run_dir / "meta.json"
    summary_file = run_dir / "summary.json"

    if not meta_file.exists() or not summary_file.exists():
        print(f"Error: Could not find meta.json and/or summary.json in {run_dir}")
        sys.exit(1)

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
            
    except Exception as e:
        print(f"Error reading JSON files: {e}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"pl-inspector Run Summary: {meta.get('run_id', 'Unknown')}")
    print(f"{'='*50}")
    
    print("\n[ Metadata ]")
    print(f"  Created:     {meta.get('created_at', 'Unknown')}")
    print(f"  Description: {meta.get('description', 'N/A')}")
    if meta.get('device_name'):
        print(f"  Device:      {meta['device_name']} (wires: {meta.get('device_wires', 'N/A')})")
        
    print("\n[ Summary ]")
    # Training diagnostic summary
    if summary.get("type") == "training_diagnostics":
        print("  Type: Training Diagnostics Session")
        print(f"  Steps logged: {summary.get('num_steps', 0)}")
        if 'first_step_timestamp' in summary:
            print(f"  Duration: {summary['first_step_timestamp']} to {summary.get('last_step_timestamp', '?')}")
            
    # Watch session summary
    else:
        print("  Type: Watch Session")
        print(f"  Events tracked: {summary.get('num_events', 0)}")
        if 'tracker' in summary:
            t = summary['tracker']
            metrics = t.get('metrics', {})
            print(f"  Execution totals: {metrics.get('executions', 0)} calls, {metrics.get('shots', 0)} shots")
            
        if 'snapshots' in summary:
            s = summary['snapshots']
            print("\n[ Snapshots ]")
            print(f"  Detected payload on {s.get('calls_with_payload', 0)} calls")
            print(f"  Total snapshot arrays saved: {s.get('arrays_saved', 0)}")
            
    print(f"\n{'-'*50}\n")


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()
