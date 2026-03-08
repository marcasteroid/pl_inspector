"""
Top-level package for pl-inspector.

This package provides observability tooling for PennyLane, including:
- persistent execution tracing,
- snapshot-based intermediate-state capture,
- Torch-focused training diagnostics.
- a local Gradio-based inspection dashboard (``pl_inspector.ui``).

The public surface area is intentionally small and focused on high-level
session managers.
"""

from __future__ import annotations

from .training import trace_training
from .watch import watch


def launch_dashboard(
    runs_dir: str = "./runs",
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    share: bool = False,
    open_browser: bool = True,
) -> None:
    """Launch the pl-inspector Gradio dashboard.

    This is a convenience wrapper around :func:`pl_inspector.ui.launch`.
    Gradio is imported lazily, so this function may be called from
    environments where Gradio is not installed without causing an
    ``ImportError`` at import time.

    Parameters
    ----------
    runs_dir:
        Root directory that contains the ``runs/`` sub-directory.
    host:
        Local host address for the Gradio server (default: "127.0.0.1").
    port:
        Local port for the Gradio server (default: 7860).
    share:
        Create a public Gradio share link (requires internet).
    open_browser:
        Automatically open the browser after launch.
    """
    from .ui.app import launch  # deferred – keeps gradio optional

    launch(
        runs_dir=runs_dir,
        host=host,
        port=port,
        share=share,
        open_browser=open_browser,
    )


__all__ = [
    "watch",
    "trace_training",
    "launch_dashboard",
]

__version__ = "0.1.0"
