#!/usr/bin/env bash
set -e

echo "========== 1. Running test suite =========="
source .venv/bin/activate
PYTHONPATH=src pytest -q tests/

echo ""
echo "========== 2. Running examples =========="
PYTHONPATH=src python3 examples/basic_trace.py
PYTHONPATH=src python3 examples/torch_training_demo.py
PYTHONPATH=src python3 examples/snapshot_demo.py

echo ""
echo "========== 3. Checking output artifacts =========="
# Find the most recently created run directory from the torch demo (which outputs to demo_runs)
LATEST_RUN=$(ls -td demo_runs/runs/* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "Error: No run directory found in demo_runs/runs."
    exit 1
fi

echo "Verifying files in: $LATEST_RUN"
for file in meta.json summary.json gradients.jsonl training_trends.png; do
    if [ ! -f "$LATEST_RUN/$file" ]; then
        echo "Error: Expected file $file is missing."
        exit 1
    fi
    echo "  [x] $file is present"
done

echo ""
echo "========== 4. Running CLI Summary =========="
PYTHONPATH=src python3 -m pl_inspector "$LATEST_RUN"

echo "All checks passed successfully! ✅"
