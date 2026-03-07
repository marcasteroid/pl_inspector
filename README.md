# pl-inspector

**pl-inspector** provides persistent observability, execution tracing, and training diagnostics for PennyLane quantum circuits. It builds strictly upon PennyLane's existing tracking and debugging primitives, persisting them into a structured, easily analyzable format.

## Overview

Working with parameterized quantum circuits often means crossing the boundary between classical ML frameworks (like PyTorch) and quantum simulators or hardware. While PennyLane includes excellent tools for local debugging (`qml.Tracker`, `qml.snapshots`), extracting these traces into a persistent, analyzable history across full training runs often requires repetitive boilerplate.

**pl-inspector** solves this by offering lightweight, context-managed session objects (`watch()` and `trace_training()`) that automatically intercept, summarize, and save these events into a structured JSON/NumPy layout on disk.

## Why Built-in PennyLane Debugging Is Not Enough

PennyLane's `qml.Tracker` and `qml.snapshots` are brilliant for immediate, interactive inspection. However, for persistent observability:

1. **Ephemeral by default**: Data lives in memory and vanishes as soon as the kernel restarts or the script finishes.
2. **Complex payload extraction**: Parsing snapshot objects (which might contain massive state vectors) requires custom summarization logic to prevent memory bloat or unwieldy output logs.
3. **No training context**: The raw PennyLane tracker doesn't inherently associate QNode executions with classical ML training steps, loss values, or gradient norms.
4. **Lack of persistence layout**: There is no standard, out-of-the-box way to dump run execution traces and parsed classical gradients into an organized directory structure.

`pl-inspector` bridges this gap, not by reinventing the tracker, but by persisting and contextualizing its outputs automatically.

## Key Features

- **Persistent Execution Tracing**: Wraps a QNode and writes execution events (metadata, timestamps, durations) seamlessly to disk.
- **Snapshot-Aware Summaries**: Automatically detects state-vector-like payloads from `qml.snapshots`, parsing out top-$k$ amplitudes and probability distributions, while persisting the raw arrays safely to `.npz`.
- **Torch Training Diagnostics**: Monitors and logs global loss, global gradient norms, and parameter update sizes for PyTorch models seamlessly.
- **Zero Framework Magic**: Unintrusive and explicit API. The original QNodes and Models are not magically mutated.

## Installation

_pl-inspector_ requires Python 3.8+ and PennyLane. PyTorch is required if utilizing the training diagnostics module.

```bash
# Clone the repository and install it locally
git clone https://github.com/your-org/pl-inspector.git
cd pl-inspector
pip install -e .
```

## Quickstart: `watch()`

Use `watch()` to trace QNode executions and optionally parse snapshots.

```python
import pennylane as qml
from pl_inspector import watch

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def my_circuit(x):
    qml.RX(x, wires=0)
    qml.CNOT(wires=[0, 1])
    # The session will automatically parse outputs of snapshot-enabled circuits
    return qml.expval(qml.PauliZ(1))

# By default, watch captures qml.Tracker data.
# Passing capture="snapshots" intercepts snapshot data attached to the output.
with watch(my_circuit, save_dir="./traces", capture="snapshots") as run:
    run(0.5)
    run(0.8)
    
    # Print a console trace summary
    run.show_trace()
    
    # Return the path to the written directory
    run_dir = run.export_json()
    print(f"Traces written to: {run_dir}")
```

## Quickstart: `trace_training()`

Use `trace_training()` to diagnose PyTorch model behavior, tracking global gradient norms to identify vanishing gradients or dying updates.

```python
import torch
import torch.nn as nn
from pl_inspector import trace_training

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# 1. Initialize session and attach model
session = trace_training(interface="torch", save_dir="./runs")
session.attach(model)

x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[0.5]])

for step in range(10):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    
    # 2. Log step (automatically computes gradient norms and update sizes)
    session.log_step(loss=loss.item(), step=step)
    optimizer.step()

# 3. Export data and optional utility plots
session.plot_loss()
session.plot_gradient_norms()
run_dir = session.export_json()
```

## Limitations

- **Snapshot Insertion**: `pl-inspector` **does not** inject snapshots into your QNodes. It only interprets payloads your QNode is already configured to return.
- **Interfaces**: Training diagnostics are currently restricted strictly to PyTorch (`interface="torch"`). JAX and TensorFlow are not currently supported.
- **Granularity**: The Torch diagnostics currently compute a global gradient norm across all tracked model parameters, rather than layer-by-layer granular summaries.

## Roadmap

- Support for JAX and TensorFlow training diagnostics.
- Granular layer-by-layer gradient norm tracking.
- Interactive terminal UI for analyzing past runs directly from the CLI.
- Extensible snapshot parsing heuristics.

## Development & Testing

We use `pytest` for all unit testing, and rely heavily on generic synthetic payloads to ensure the test suite remains fast and decoupled from deep PennyLane simulation internals.

```bash
# Install development dependencies
pip install pytest

# Run the test suite
python -m pytest tests/
```

---

### CV Project Blurb

> **pl-inspector**  
> *Developer / Maintainer*  
> Designed and built a Python observability tool for PennyLane quantum circuits. Developed context-managed tracing and diagnostics layers that wrap native framework events, persisting execution traces, parsing state-vector snapshots into configurable summaries, and logging PyTorch gradient norms to JSON/NumPy storage schemas on disk.
