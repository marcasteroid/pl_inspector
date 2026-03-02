"""
Top-level package for pl-inspector.

This package provides observability tooling for PennyLane, including:
- persistent execution tracing,
- snapshot-based intermediate-state capture,
- Torch-focused training diagnostics.

Functionality is under active development and APIs may change.
"""

__all__ = [
    "watch",
    "trace",
    "tracker",
    "snapshots",
    "training",
    "storage",
    "plotting",
    "models",
    "utils",
    "exceptions",
]

