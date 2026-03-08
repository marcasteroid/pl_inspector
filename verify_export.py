from pathlib import Path
from pl_inspector.ui.io import load_run_dir, export_run_report
import json

base_dir = Path("demo_runs")
run_id = "run-20260308T132456Z"
run_dir = base_dir / "runs" / run_id

print(f"Loading data for {run_id}...")
run = load_run_dir(run_dir)

print(f"Exporting report for {run_id}...")
report_path = export_run_report(run_dir, run.meta, run.summary, run.events, run.gradients)

print(f"Report exported to: {report_path}")
print("Verifying content...")
content = report_path.read_text()
if "## Event Call Counts" in content:
    print("Found 'Event Call Counts' section!")
else:
    print("MISSING 'Event Call Counts' section!")

if "| circuit_a | 5 |" in content:
    print("Found circuit_a call count!")
if "| circuit_b | 3 |" in content:
    print("Found circuit_b call count!")
