"""
Top-level package for pl-inspector.

This package provides observability tooling for PennyLane, including:
- persistent execution tracing,
- snapshot-based intermediate-state capture,
- Torch-focused training diagnostics.

The public surface area is intentionally small and focused on high-level
session managers.
"""

from __future__ import annotations

from .training import trace_training
from .watch import watch

__all__ = [
    "watch",
    "trace_training",
]

__version__ = "0.1.0"
