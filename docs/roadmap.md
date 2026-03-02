# pl-inspector roadmap

This document outlines the high-level development roadmap for pl-inspector.

## Near-term goals

- Implement persistent execution tracing around PennyLane QNodes.
- Add snapshot-based intermediate-state capture utilities using ``qml.snapshots``.
- Provide lightweight, Torch-first training diagnostics (loss curves, gradient stats).
- Establish a simple on-disk storage format and retrieval API.

## Medium-term goals

- Expand plotting utilities for common diagnostic views using matplotlib.
- Harden APIs, docs, and examples for real-world workflows.
- Improve configuration ergonomics and integration with existing PennyLane tooling.

## Long-term ideas

- Explore richer tracking for larger experiments while remaining filesystem-only.
- Consider additional optional integrations (e.g., experiment tracking tools) without adding server components.

