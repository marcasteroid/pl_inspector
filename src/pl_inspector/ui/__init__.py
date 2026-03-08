"""Gradio-based GUI dashboard for pl-inspector.

This sub-package provides a local, read-only dashboard for inspecting
``runs/<run_id>/`` artefacts produced by pl-inspector:

- ``meta.json`` / ``summary.json`` — run metadata
- ``events.jsonl`` — execution events
- ``gradients.jsonl`` — training step records
- ``snapshots.npz`` — NumPy arrays captured mid-circuit

Entry points
------------
- :func:`pl_inspector.ui.launch` — programmatic launcher
- ``python -m pl_inspector.ui`` — CLI launcher

Architecture
------------
::

    ui/
    ├── __init__.py  — public surface
    ├── app.py       — Gradio layout & wiring
    ├── io.py        — disk I/O helpers (read-only)
    ├── render.py    — plot / table builders
    ├── state.py     — lightweight in-memory run state
    └── errors.py    — dashboard-specific exceptions

"""

from __future__ import annotations

from .app import build_app, launch, main

__all__ = ["build_app", "launch", "main"]
