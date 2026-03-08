"""Dashboard-specific exception types for the pl-inspector UI.

All exceptions inherit from :class:`pl_inspector.exceptions.PLInspectorError`
so that callers can catch a single base type if desired.

Hierarchy
---------
::

    PLInspectorError
    └── DashboardError
        ├── InvalidRunDirectoryError   — path is not a readable directory
        ├── RunNotFoundError           — run_id not present under runs/
        ├── ArtifactMissingError       — expected file absent (usually non-fatal)
        └── ArtifactParseError         — file present but cannot be decoded
"""

from __future__ import annotations

from ..exceptions import PLInspectorError


class DashboardError(PLInspectorError):
    """Base exception raised by the pl-inspector dashboard."""


class InvalidRunDirectoryError(DashboardError):
    """Raised when the supplied path is not a valid, readable directory.

    This is a hard error — the caller passed something that cannot be used
    as a base or run directory at all (e.g. a file, a non-existent path that
    should have existed, or a permission-denied directory).

    Parameters
    ----------
    path:
        The filesystem path that was rejected.
    reason:
        Human-readable explanation of why the path is invalid.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Invalid run directory {path!r}: {reason}")
        self.path = path
        self.reason = reason


class RunNotFoundError(DashboardError):
    """Raised when the requested run directory does not exist under ``runs/``.

    Parameters
    ----------
    run_id:
        The identifier (or path fragment) of the run that could not be found.
    """

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run not found: {run_id!r}")
        self.run_id = run_id


class ArtifactMissingError(DashboardError):
    """Raised when an expected artefact file is absent inside a run directory.

    This is typically *non-fatal*: loaders catch it internally and return an
    empty default.  It is re-raised only when the caller explicitly requires
    the file to be present.

    Parameters
    ----------
    path:
        The filesystem path that was expected but is missing.
    """

    def __init__(self, path: str) -> None:
        super().__init__(f"Artefact not found: {path!r}")
        self.path = path


class ArtifactParseError(DashboardError):
    """Raised when an artefact file exists but cannot be parsed correctly.

    Parameters
    ----------
    path:
        The filesystem path of the malformed artefact.
    detail:
        Human-readable description of the parse failure.
    """

    def __init__(self, path: str, detail: str) -> None:
        super().__init__(f"Failed to parse {path!r}: {detail}")
        self.path = path
        self.detail = detail
