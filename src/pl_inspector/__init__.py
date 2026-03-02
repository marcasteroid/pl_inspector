"""
Top-level package for pl-inspector.

This package provides observability tooling for PennyLane, including:
- persistent execution tracing,
- snapshot-based intermediate-state capture,
- Torch-focused training diagnostics.

At this stage, only the foundational datamodels and base error type are
considered part of the (still experimental) public surface.
"""

from __future__ import annotations

from .exceptions import PLInspectorError
from .models import (
    ExecutionEvent,
    RunMeta,
    SnapshotRecord,
    TrackerRecord,
    TrainingStepRecord,
)

__all__ = [
    "RunMeta",
    "ExecutionEvent",
    "TrackerRecord",
    "SnapshotRecord",
    "TrainingStepRecord",
    "PLInspectorError",
]

__version__ = "0.1.0"

