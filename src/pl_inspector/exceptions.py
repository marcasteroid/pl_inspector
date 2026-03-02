"""Custom exception types for pl-inspector."""

from __future__ import annotations


class PLInspectorError(Exception):
    """Base exception for all pl-inspector errors."""


class StorageError(PLInspectorError):
    """Errors related to persisting or loading inspector state."""


class SnapshotSupportError(PLInspectorError):
    """Raised when PennyLane snapshot support is unavailable or misconfigured."""


class TrainingInterfaceError(PLInspectorError):
    """Raised for issues in the Torch training integration layer."""


