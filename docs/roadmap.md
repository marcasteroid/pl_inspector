# pl-inspector Roadmap

This document outlines the scope, goals, and non-goals for `pl-inspector`. We prioritize maintaining a small, sharp tool that integrates cleanly with PennyLane and standard ML frameworks, rather than building an opaque framework wrapper.

---

## Current MVP Scope (v1)

The v1 release focuses strictly on foundational persistence and observability:

- **Execution Tracing:** Persisting `qml.Tracker` events (timestamps, metadata, call durations) to an organized directory structure.
- **Snapshot Extraction:** Parsing outputs from QNodes configured for `qml.snapshots`, summarising state-vector-like arrays (probabilities, top-k amplitudes), and storing raw arrays to `.npz`.
- **Torch-first Diagnostics:** A dedicated module for tracking global gradient norms, parameter updates, and detecting vanishing gradients in PyTorch training loops.
- **Simple Plotting:** Lightweight `matplotlib` wrappers for visualizing basic trace and training summary data.

## Explicit Non-Goals (v1)

To keep the MVP practical and maintainable, the following are explicitly out of scope for v1:

- **Automatic Snapshot Insertion:** We do not automatically inject snapshots or alter user circuit definitions. The user must configure their QNodes to return snapshots.
- **Support for JAX/TensorFlow:** Training diagnostics are implemented exclusively for PyTorch to ensure high-quality tracking before horizontal expansion.
- **Layer-by-Layer Gradients:** The MVP tracks *global* parameter norm metrics rather than granular, layer-specific tracking across the quantum-classical boundary.
- **Interactive UI/Dashboard:** No web dashboards or CLI GUIs are included. Traces are exported cleanly to standard formats (JSON/`.npz`) for users to inspect with their preferred tools.

---

## Why no custom device in v1?

A common pattern for observability tooling in quantum software is to implement a custom simulator or device wrapper (e.g., `qml.device("inspector.qubit")`). We explicitly avoided this approach for v1:

1. **Fragility:** Device APIs change. Tying tracing to device internals makes the library brittle against upstream framework updates.
2. **Performance:** Wrapping device executions can introduce significant overhead, perturbing the timing metrics the tool is trying to measure.
3. **Compatibility:** Users should be able to trace executions on *any* backend (hardware, lightning, default.qubit) without changing their device definitions. 

By wrapping the *callable* (the QNode or model) and intercepting existing framework trackers/outputs, we maintain compatibility and low overhead.

---

## Why Torch-first?

The choice to implement training diagnostics exclusively for PyTorch in v1 is pragmatic:

1. **Ecosystem Momentum:** PyTorch remains the dominant framework for hybrid quantum-classical machine learning research.
2. **Defensible Scope:** Deeply supporting the quirks of one autograd engine (parameter detachment, gradient extraction) ensures the resulting tool is actually useful and robust. 
3. **API Validation:** Building a tight integration with PyTorch allows us to validate the `pl-inspector` session API before committing to the abstraction debt required to support Jax (`optax`) and TensorFlow concurrently.

---

## Potential v2 Ideas

These are speculative directions for future development, pending feedback on the v1 MVP:

- **JAX and Optax Support:** Expanding the `trace_training` module to support JAX models and `optax` gradient transformations.
- **Granular Parameter Tracking:** Breaking down gradient norms and parameter updates by layer/module or isolating quantum parameters from classical ones.
- **Heuristic Anomaly Detection:** Expanding the current "vanishing gradient" detection into a broader suite of warnings (e.g., barren plateau detection, parameter saturation).
- **CLI Inspection Tooling:** A simple terminal UI (`pl-inspect view ./runs`) to quickly scan logs without writing a notebook.
