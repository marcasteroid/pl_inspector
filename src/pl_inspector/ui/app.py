"""Main Gradio application for the pl-inspector dashboard.

Provides a local, read-only single-run inspection UI.  No backend services,
no websockets, no React/Vue — pure Gradio Blocks.

Layout
------
::

    ┌──────────────────────────────────────────────┐
    │  🔬 pl-inspector Dashboard                   │
    ├─────────────────┬────────────────────────────┤
    │ Sidebar         │ Main panel                 │
    │  base_dir input │  ⚠ Warnings accordion      │
    │  [Load] button  │  ┌─────────────────────┐   │
    │  run dropdown   │  │ Overview │Events│... │   │
    │  [Refresh]      │  └─────────────────────┘   │
    └─────────────────┴────────────────────────────┘

Tabs
~~~~
1. **Overview** — meta.json + summary.json as JSON viewers
2. **Events** — events table + execution timing histogram
3. **Training** — gradient/loss plots + data table
4. **Snapshots** — snapshot key selector + auto-dispatched plot

Usage
-----
::

    # Via module (recommended)
    python -m pl_inspector.ui.app [--runs-dir ./runs] [--port 7860]

    # Programmatic
    from pl_inspector.ui.app import launch
    launch(runs_dir="./runs", port=7860)
"""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _get_gr():
    """Import and return Gradio, or raise ImportError."""
    try:
        import gradio as gr
        return gr
    except ImportError as exc:
        raise ImportError(
            "The pl-inspector dashboard requires 'gradio'. "
            "Install it with:  pip install gradio"
        ) from exc

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
DASHBOARD_CSS = """
#header { text-align: center; padding: 8px 0 4px 0; }
#warnings_box { border-left: 4px solid #e6ac00; background: #fffbea;
                border-radius: 4px; padding: 6px 10px; font-size: 0.88em; }
#sidebar { min-width: 240px; max-width: 280px; }
.tab-content { padding-top: 8px; }
.metric-card { background: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #e9ecef; text-align: center; }
.metric-value { font-size: 1.4em; font-weight: bold; color: #2d3748; }
.metric-label { font-size: 0.85em; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }
"""

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_RUNS_DIR: str = "./runs"
DEFAULT_PORT: int = 7860
DASHBOARD_TITLE: str = "pl-inspector Dashboard"
_NO_RUNS_MSG: str = "No runs found. Check the directory path and click Refresh."
_NO_RUN_SELECTED: str = "← Select a run and click Load."

from .render import (
    events_to_df,
    gradients_to_df,
    plot_loss_curve,
    plot_grad_norm_curve,
    plot_timing_histogram,
    plot_timing_comparison,
    plot_snapshot_array,
    plot_snapshot_amplitudes,
    plot_metric_comparison,
    plot_event_counts as render_event_counts,
)

# ---------------------------------------------------------------------------
# Internal helpers (Gradio-free)
# ---------------------------------------------------------------------------


def _discover_runs(base_dir: str) -> List[str]:
    """Return run IDs under *base_dir*, with a friendly fallback on error."""
    from .io import list_runs
    from .errors import InvalidRunDirectoryError

    base_dir = base_dir.strip() or DEFAULT_RUNS_DIR
    try:
        runs = list_runs(base_dir)
        return runs if runs else []
    except InvalidRunDirectoryError:
        return []
    except Exception:  # noqa: BLE001
        logger.debug("list_runs failed: %s", traceback.format_exc())
        return []


def _load_run_safe(base_dir: str, run_id: str):
    """Load a run and return ``(LoadedRun | None, error_str | None)``."""
    from .state import load_run
    from .errors import RunNotFoundError, InvalidRunDirectoryError

    base_dir = base_dir.strip() or DEFAULT_RUNS_DIR
    try:
        run = load_run(base_dir, run_id)
        return run, None
    except (RunNotFoundError, InvalidRunDirectoryError) as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error: {exc}\n{traceback.format_exc()}"


def _format_warnings(run) -> str:
    """Return a Markdown warning string, or empty string if none."""
    if run is None:
        return ""
    lines = list(run.warnings or [])
    for name, msg in (run.load_errors or {}).items():
        entry = f"⚠ **{name}**: {msg}"
        if entry not in lines:
            lines.append(entry)
    if not lines:
        return ""
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app(default_runs_dir: str = DEFAULT_RUNS_DIR) -> "gr.Blocks":  # noqa: F821
    """Construct and return the Gradio Blocks application.

    Parameters
    ----------
    default_runs_dir:
        Pre-filled value for the base directory textbox.

    Returns
    -------
    gradio.Blocks
        The fully assembled dashboard (not yet launched).

    Raises
    ------
    ImportError
        If *gradio* is not installed.
    """
    gr = _get_gr()

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    with gr.Blocks(title=DASHBOARD_TITLE) as demo:
        # ---- shared in-memory state: Run A ---------------------------------
        _state_meta_a      = gr.State(None)
        _state_summary_a   = gr.State(None)
        _state_events_a    = gr.State([])
        _state_gradients_a = gr.State([])
        _state_snap_keys_a = gr.State([])
        _state_snap_path_a = gr.State(None)

        # ---- shared in-memory state: Run B ---------------------------------
        _state_meta_b      = gr.State(None)
        _state_summary_b   = gr.State(None)
        _state_events_b    = gr.State([])
        _state_gradients_b = gr.State([])
        _state_snap_keys_b = gr.State([])
        _state_snap_path_b = gr.State(None)

        # ---- comparison state ----------------------------------------------
        _state_compare     = gr.State(None)  # CompareState | None

        # ---- header -------------------------------------------------------
        gr.Markdown(f"# 🔬 {DASHBOARD_TITLE}", elem_id="header")

        # ---- outer layout: sidebar + main panel ---------------------------
        with gr.Row(equal_height=False):

            # ================================================================
            # SIDEBAR
            # ================================================================
            with gr.Column(scale=1, min_width=240, elem_id="sidebar"):
                gr.Markdown("### 📁 Run Selection")

                inp_dir = gr.Textbox(
                    label="Data Root Directory",
                    value=default_runs_dir,
                    placeholder="e.g., ./demo_runs",
                    lines=1,
                    elem_id="inp_dir",
                )

                with gr.Row():
                    btn_refresh = gr.Button("↺ Refresh", variant="secondary", size="sm")

                gr.Markdown("---")
                gr.Markdown("#### 🟢 Run A")
                dd_run_a = gr.Dropdown(
                    label="Select Run A",
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                    elem_id="dd_run_a",
                )
                btn_load_a = gr.Button("▶ Load Run A", variant="primary", size="sm", interactive=False)

                gr.Markdown("---")
                gr.Markdown("#### 🟠 Run B")
                dd_run_b = gr.Dropdown(
                    label="Select Run B",
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                    elem_id="dd_run_b",
                )
                btn_load_b = gr.Button("▶ Load Run B", variant="secondary", size="sm", interactive=False)

            # ================================================================
            # MAIN PANEL
            # ================================================================
            with gr.Column(scale=4):

                # ---- warnings accordion -----------------------------------
                with gr.Accordion("⚠ Load warnings", open=False, visible=False) as acc_warnings:
                    md_warnings = gr.Markdown(elem_id="warnings_box")

                # ---- tabs -------------------------------------------------
                with gr.Tabs():

                    # ========================================================
                    # TAB 1: OVERVIEW
                    # ========================================================
                    with gr.TabItem("📋 Overview"):
                        md_overview_hint = gr.Markdown(
                            _NO_RUN_SELECTED, visible=True, elem_id="overview_hint"
                        )
                        
                        with gr.Column(visible=False) as col_overview:
                            with gr.Row():
                                with gr.Column(elem_classes="metric-card"):
                                    gr.Markdown("<div class='metric-label'>Total Calls</div>", elem_id="m_label_calls")
                                    val_total_calls = gr.Markdown("<div class='metric-value'>-</div>", elem_id="m_val_calls")
                                with gr.Column(elem_classes="metric-card"):
                                    gr.Markdown("<div class='metric-label'>Total Steps</div>", elem_id="m_label_steps")
                                    val_total_steps = gr.Markdown("<div class='metric-value'>-</div>", elem_id="m_val_steps")
                                with gr.Column(elem_classes="metric-card"):
                                    gr.Markdown("<div class='metric-label'>Final Loss</div>", elem_id="m_label_loss")
                                    val_final_loss = gr.Markdown("<div class='metric-value'>-</div>", elem_id="m_val_loss")
                                with gr.Column(elem_classes="metric-card"):
                                    gr.Markdown("<div class='metric-label'>Duration</div>", elem_id="m_label_duration")
                                    val_duration = gr.Markdown("<div class='metric-value'>-</div>", elem_id="m_val_duration")

                            with gr.Accordion("Raw Metadata (JSON Explorer)", open=False):
                                with gr.Row():
                                    with gr.Column():
                                        json_meta = gr.JSON(label="meta.json", value=None)
                                    with gr.Column():
                                        json_summary = gr.JSON(label="summary.json", value=None)
                            
                            btn_export_a = gr.Button("📄 Export Report", variant="secondary", size="sm")
                            md_export_status_a = gr.Markdown("")

                    # ========================================================
                    # TAB 2: EVENTS
                    # ========================================================
                    with gr.TabItem("⚡ Events"):
                        md_events_empty = gr.Markdown(
                            "No execution events found. Ensure `pl_inspector.watch()` was used during the run.",
                            visible=True
                        )
                        with gr.Column(visible=False) as col_events:
                            with gr.Row():
                                md_event_summary = gr.Markdown(elem_id="event_summary")
                                dd_event_filter = gr.Dropdown(
                                    label="Filter by QNode",
                                    choices=["All"],
                                    value="All",
                                    interactive=True,
                                )

                            with gr.Row():
                                plot_event_counts = gr.Plot(label="Call counts per QNode")
                                plot_timing = gr.Plot(label="Timing histogram (duration_ms)")

                            df_events = gr.Dataframe(
                                label="Execution events",
                                wrap=False,
                                max_height=320,
                            )

                    # ========================================================
                    # TAB 3: TRAINING
                    # ========================================================
                    with gr.TabItem("📈 Training"):
                        md_training_empty = gr.Markdown(
                            "No gradient records found. This occurs if training wasn't traced with `pl_inspector.trace_training()`.",
                            visible=True
                        )
                        with gr.Column(visible=False) as col_training:
                            with gr.Row():
                                plot_loss   = gr.Plot(label="Loss curve")
                                plot_gnorm  = gr.Plot(label="Gradient norm")
                            df_grads = gr.Dataframe(
                                label="Training steps",
                                wrap=False,
                                max_height=320,
                            )

                    # ========================================================
                    # TAB 4: SNAPSHOTS
                    # ========================================================
                    with gr.TabItem("🔭 Snapshots"):
                        md_snap_empty = gr.Markdown(
                            "No snapshots loaded.", visible=True
                        )
                        with gr.Column(visible=False) as col_snapshots:
                            dd_snap_key = gr.Dropdown(
                                label="Snapshot array key",
                                choices=[],
                                value=None,
                                allow_custom_value=False,
                            )
                            with gr.Row():
                                plot_snap_dist = gr.Plot(label="Distribution / heatmap")
                                plot_snap_amp  = gr.Plot(label="Top-k amplitudes")

                    # ========================================================
                    # TAB 5: COMPARE
                    # ========================================================
                    with gr.TabItem("🆚 Compare"):
                        md_compare_hint = gr.Markdown(
                            "Load two runs to see comparison.", visible=True
                        )
                        with gr.Column(visible=False) as col_compare:
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### summary.json (A)")
                                    json_summary_a = gr.JSON(label="Summary A", value=None)
                                with gr.Column():
                                    gr.Markdown("#### summary.json (B)")
                                    json_summary_b = gr.JSON(label="Summary B", value=None)

                            gr.Markdown("#### Key Metrics")
                            df_compare_metrics = gr.Dataframe(
                                label="Metric comparison",
                                wrap=False,
                                interactive=False,
                            )

                            with gr.Row():
                                plot_compare_loss   = gr.Plot(label="Loss Comparison (A vs B)")
                                plot_compare_gnorm  = gr.Plot(label="Grad Norm Comparison (A vs B)")
                            plot_compare_timing = gr.Plot(label="Timing Comparison (A vs B)")

                            btn_export_compare = gr.Button("📄 Export Comparison Report", variant="secondary", size="sm")
                            md_export_status_compare = gr.Markdown("")

        # ====================================================================
        # CALLBACKS
        # ====================================================================

        # ---- refresh run list ---------------------------------------------
        def cb_refresh(base_dir: str):
            """Populate the run dropdown from the current base directory."""
            runs = _discover_runs(base_dir)
            if runs:
                return gr.update(choices=runs, value=runs[0])
            return gr.update(choices=[], value=None, placeholder=_NO_RUNS_MSG)

        btn_refresh.click(
            fn=cb_refresh,
            inputs=[inp_dir],
            outputs=[dd_run_a],
        )
        btn_refresh.click(
            fn=cb_refresh,
            inputs=[inp_dir],
            outputs=[dd_run_b],
        )

        # Auto-refresh when the directory path is committed
        inp_dir.submit(
            fn=cb_refresh,
            inputs=[inp_dir],
            outputs=[dd_run_a],
        )
        inp_dir.submit(
            fn=cb_refresh,
            inputs=[inp_dir],
            outputs=[dd_run_b],
        )

        dd_run_a.change(fn=lambda v: gr.update(interactive=bool(v)), inputs=[dd_run_a], outputs=[btn_load_a])
        dd_run_b.change(fn=lambda v: gr.update(interactive=bool(v)), inputs=[dd_run_b], outputs=[btn_load_b])

        # ---- load a run ---------------------------------------------------
        def _cb_load_run(base_dir: str, run_id: Optional[str], is_primary: bool = True):
            """Load a run and return update values for state and UI."""
            # --- default empties ---
            s_meta, s_summary, s_events, s_grads, s_skeys, s_spath = None, None, [], [], [], None
            acc_open, acc_vis, warn_text = False, False, ""

            # UI components (only updated if is_primary)
            o_ui = [gr.update(visible=False), gr.update(visible=True), # col_overview, hint
                    "-", "-", "-", "-", # metric values
                    None, None, # JSONs
                    gr.update(interactive=False)] # export button
            
            ev_ui = [gr.update(visible=True), gr.update(visible=False),
                     None, "All", None, None, None] # empty_vis, col_vis, summary, filter, counts_plot, timing_plot, df
            tr_ui = [gr.update(visible=True), gr.update(visible=False), None, None, None] # empty_vis, col_vis, loss, gnorm, df
            sn_ui = [gr.update(visible=True), gr.update(visible=False), gr.update(choices=[], value=None), None, None] # empty_vis, col_vis, keys, dist, amp

            if not run_id:
                return (s_meta, s_summary, s_events, s_grads, s_skeys, s_spath,
                        acc_open, gr.update(visible=acc_vis), warn_text,
                        *o_ui, *ev_ui, *tr_ui, *sn_ui)

            run, err = _load_run_safe(base_dir, run_id)
            if err:
                warn_text = f"❌ Failed to load run:\n{err}"
                acc_vis, acc_open = True, True
                return (s_meta, s_summary, s_events, s_grads, s_skeys, s_spath,
                        acc_open, gr.update(visible=acc_vis), warn_text,
                        *o_ui, *ev_ui, *tr_ui, *sn_ui)

            # --- capture state ---
            s_meta = run.meta
            s_summary = run.summary
            s_events = run.events
            s_grads = run.gradients
            s_skeys = run.snapshot_keys
            s_spath = str(run.snapshot_path) if run.snapshot_path else None

            if not is_primary:
                # For Run B, we only care about state (it will trigger compare update)
                return (s_meta, s_summary, s_events, s_grads, s_skeys, s_spath,
                        gr.update(open=acc_open), gr.update(visible=acc_vis), _format_warnings(run),
                        *o_ui, *ev_ui, *tr_ui, *sn_ui)

            # --- warnings ---
            warn_text = _format_warnings(run)
            if warn_text:
                acc_vis, acc_open = True, True

            # --- overview stats ---
            v_calls = str(run.summary.get("total_calls", "-"))
            v_steps = str(run.summary.get("total_steps", "-"))
            v_loss = "-"
            if run.summary.get("final_loss") is not None:
                v_loss = f"{run.summary['final_loss']:.4f}"
            v_dur = "-"
            if run.summary.get("total_duration_ms") is not None:
                v_dur = f"{run.summary['total_duration_ms']/1000:.2f}s"
            elif run.events:
                 durs = [e.get("details", {}).get("duration_ms", 0) for e in run.events]
                 v_dur = f"{sum(durs)/1000:.2f}s"

            m_calls = f"<div class='metric-value'>{v_calls}</div>"
            m_steps = f"<div class='metric-value'>{v_steps}</div>"
            m_loss = f"<div class='metric-value'>{v_loss}</div>"
            m_dur = f"<div class='metric-value'>{v_dur}</div>"

            o_ui = [gr.update(visible=True), gr.update(visible=False),
                    m_calls, m_steps, m_loss, m_dur,
                    run.meta, run.summary,
                    gr.update(interactive=True)]

            # --- events ---
            if run.has_events:
                qnodes = sorted(list(set(e.get("name") for e in run.events if e.get("name"))))
                filter_choices = ["All"] + qnodes
                
                # Compute stats
                avg_dur = np.mean([e.get("details", {}).get("duration_ms", 0) for e in run.events])
                total_calls = len(run.events)
                summary_md = f"**Total Calls**: {total_calls} | **Avg Duration**: {avg_dur:.2f} ms"
                
                ev_ui = [gr.update(visible=False), gr.update(visible=True),
                         summary_md, gr.update(choices=filter_choices, value="All"),
                         render_event_counts(run.events), plot_timing_histogram(run.events),
                         events_to_df(run.events)]

            # --- training ---
            if run.has_gradients:
                tr_ui = [gr.update(visible=False), gr.update(visible=True),
                         plot_loss_curve(run.gradients), plot_grad_norm_curve(run.gradients),
                         gradients_to_df(run.gradients)]

            # --- snapshots ---
            if run.has_snapshots:
                first_key = run.snapshot_keys[0]
                arr = run.get_snapshot(first_key)
                sn_ui = [gr.update(visible=False), gr.update(visible=True),
                         gr.update(choices=run.snapshot_keys, value=first_key),
                         plot_snapshot_array(arr, key=first_key) if arr is not None else None,
                         plot_snapshot_amplitudes(arr, key=first_key) if arr is not None else None]

            return (s_meta, s_summary, s_events, s_grads, s_skeys, s_spath,
                    gr.update(open=acc_open), gr.update(visible=acc_vis), warn_text,
                    *o_ui, *ev_ui, *tr_ui, *sn_ui)

        def cb_update_compare(meta_a, sum_a, ev_a, gr_a, meta_b, sum_b, ev_b, gr_b):
            """Update comparison tab when either run changes."""
            if not meta_a or not meta_b:
                return gr.update(visible=True), gr.update(visible=False), None, None, None, None, None, None

            from .state import LoadedRun, load_run_pair

            # We can't use load_run_pair easily because it expects base_dir/run_id.
            # But we already have the data in state!
            # Let's mock LoadedRun objects or just use the pre-computed logic.
            # Actually, state.py has `CompareState.summary_table` which is useful.

            run_a = LoadedRun(run_id=meta_a.get("run_id", "A"), run_dir=Path("."), meta=meta_a, summary=sum_a, events=ev_a, gradients=gr_a)
            run_b = LoadedRun(run_id=meta_b.get("run_id", "B"), run_dir=Path("."), meta=meta_b, summary=sum_b, events=ev_b, gradients=gr_b)

            from .state import CompareState, _compute_metric_comparison, _DEFAULT_COMPARE_METRICS
            comparisons = {}
            if run_a.has_gradients and run_b.has_gradients:
                for metric in _DEFAULT_COMPARE_METRICS:
                    try:
                        comparisons[metric] = _compute_metric_comparison(run_a, run_b, metric)
                    except Exception:  # noqa: BLE001
                        pass

            cs = CompareState(run_a=run_a, run_b=run_b, comparisons=comparisons)

            # Build summary table dataframe
            metrics_rows = cs.summary_table()
            # Add some extra rows if available
            # total calls (from summary)
            calls_a = sum_a.get("total_calls", "N/A")
            calls_b = sum_b.get("total_calls", "N/A")
            metrics_rows.insert(1, ("total_calls", calls_a, calls_b))

            df_metrics = pd.DataFrame(metrics_rows, columns=["Metric", "Run A", "Run B"])

            # Plots
            fig_loss = plot_metric_comparison(gr_a, gr_b, metric="loss", label_a=run_a.run_id, label_b=run_b.run_id)
            fig_gnorm = plot_metric_comparison(gr_a, gr_b, metric="grad_norm", label_a=run_a.run_id, label_b=run_b.run_id)
            fig_timing = plot_timing_comparison(ev_a, ev_b, label_a=run_a.run_id, label_b=run_b.run_id)

            return (gr.update(visible=False), gr.update(visible=True),
                    sum_a, sum_b, df_metrics, fig_loss, fig_gnorm, fig_timing)

        _load_outputs_a = [
            _state_meta_a, _state_summary_a, _state_events_a, _state_gradients_a,
            _state_snap_keys_a, _state_snap_path_a,
            acc_warnings, acc_warnings, md_warnings,
            col_overview, md_overview_hint,
            val_total_calls, val_total_steps, val_final_loss, val_duration,
            json_meta, json_summary,
            btn_export_a,
            md_events_empty, col_events, md_event_summary, dd_event_filter, plot_event_counts, plot_timing, df_events,
            md_training_empty, col_training, plot_loss, plot_gnorm, df_grads,
            md_snap_empty, col_snapshots, dd_snap_key, plot_snap_dist, plot_snap_amp,
        ]

        _load_outputs_b = [
            _state_meta_b, _state_summary_b, _state_events_b, _state_gradients_b,
            _state_snap_keys_b, _state_snap_path_b,
            acc_warnings, acc_warnings, md_warnings,
            col_overview, md_overview_hint,
            val_total_calls, val_total_steps, val_final_loss, val_duration,
            json_meta, json_summary,
            btn_export_a, # We still update Run A UI even if loading into B? 
                          # Actually _cb_load_run returns defaults for UI if is_primary=False.
                          # But Gradio requires mapping.
            md_events_empty, col_events, md_event_summary, dd_event_filter, plot_event_counts, plot_timing, df_events,
            md_training_empty, col_training, plot_loss, plot_gnorm, df_grads,
            md_snap_empty, col_snapshots, dd_snap_key, plot_snap_dist, plot_snap_amp,
        ]

        btn_load_a.click(
            fn=lambda d, r: _cb_load_run(d, r, is_primary=True),
            inputs=[inp_dir, dd_run_a],
            outputs=_load_outputs_a,
        )

        btn_load_b.click(
            fn=lambda d, r: _cb_load_run(d, r, is_primary=False),
            inputs=[inp_dir, dd_run_b],
            outputs=_load_outputs_b,
        )

        # Trigger comparison update when either state changes
        _compare_inputs = [
            _state_meta_a, _state_summary_a, _state_events_a, _state_gradients_a,
            _state_meta_b, _state_summary_b, _state_events_b, _state_gradients_b,
        ]
        _compare_outputs = [
            md_compare_hint, col_compare,
            json_summary_a, json_summary_b, df_compare_metrics,
            plot_compare_loss, plot_compare_gnorm, plot_compare_timing
        ]

        for s in [_state_meta_a, _state_meta_b]:
            s.change(fn=cb_update_compare, inputs=_compare_inputs, outputs=_compare_outputs)

        # ---- snapshot key selector ----------------------------------------
        def cb_snapshot_key(snap_key: Optional[str], snap_path: Optional[str]):
            """Re-render snapshot plots when the selected array key changes."""
            if not snap_key or not snap_path:
                return None, None
            import numpy as np
            from .errors import ArtifactParseError

            npz_path = Path(snap_path)
            if not npz_path.exists():
                return None, None
            try:
                with np.load(npz_path, allow_pickle=False) as npz:
                    arr = npz[snap_key]
                return (
                    plot_snapshot_array(arr, key=snap_key),
                    plot_snapshot_amplitudes(arr, key=snap_key),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not render snapshot key %r: %s", snap_key, exc)
                return None, None

        dd_snap_key.change(
            fn=cb_snapshot_key,
            inputs=[dd_snap_key, _state_snap_path_a],
            outputs=[plot_snap_dist, plot_snap_amp],
        )

        # ---- event filter -------------------------------------------------
        def cb_event_filter(qnode_name, events):
            if not events:
                return None, None, events_to_df([])
            
            filtered = events
            if qnode_name and qnode_name != "All":
                filtered = [e for e in events if e.get("name") == qnode_name]
            
            return (
                render_event_counts(filtered, title=f"Event counts: {qnode_name}"),
                plot_timing_histogram(filtered, title=f"Timing distribution: {qnode_name}"),
                events_to_df(filtered)
            )

        dd_event_filter.change(
            fn=cb_event_filter,
            inputs=[dd_event_filter, _state_events_a],
            outputs=[plot_event_counts, plot_timing, df_events],
        )

        # ---- initial population on load -----------------------------------
        demo.load(
            fn=cb_refresh,
            inputs=[inp_dir],
            outputs=[dd_run_a],
        )
        demo.load(
            fn=cb_refresh,
            inputs=[inp_dir],
            outputs=[dd_run_b],
        )

        # ---- export report ------------------------------------------------
        def cb_export_run(base_dir, run_id, meta, summary, events, gradients):
            if not run_id:
                return "❌ No run selected."
            try:
                from .io import get_run_dir, export_run_report
                rdir = get_run_dir(base_dir, run_id)
                path = export_run_report(rdir, meta, summary, events, gradients)
                return f"✅ Report exported to: `{path}`"
            except Exception as exc:
                return f"❌ Export failed: {exc}"

        def cb_export_compare(base_dir, r_a, m_a, s_a, e_a, g_a, r_b, m_b, s_b, e_b, g_b):
            if not r_a or not r_b:
                return "❌ Both runs must be loaded for comparison report."
            try:
                from .io import get_run_dir, export_comparison_report
                rdir_a = get_run_dir(base_dir, r_a)
                path = export_comparison_report(rdir_a, r_a, m_a, s_a, e_a, g_a, r_b, m_b, s_b, e_b, g_b)
                return f"✅ Comparison report exported to: `{path}`"
            except Exception as exc:
                return f"❌ Export failed: {exc}"

        btn_export_a.click(
            fn=cb_export_run,
            inputs=[inp_dir, dd_run_a, _state_meta_a, _state_summary_a, _state_events_a, _state_gradients_a],
            outputs=[md_export_status_a],
        )

        btn_export_compare.click(
            fn=cb_export_compare,
            inputs=[
                inp_dir,
                dd_run_a, _state_meta_a, _state_summary_a, _state_events_a, _state_gradients_a,
                dd_run_b, _state_meta_b, _state_summary_b, _state_events_b, _state_gradients_b,
            ],
            outputs=[md_export_status_compare],
        )

    return demo


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def launch(
    *,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    share: bool = False,
    open_browser: bool = True,
) -> None:
    """Build and launch the pl-inspector dashboard.

    Parameters
    ----------
    runs_dir:
        Root directory containing the ``runs/`` sub-directory.
    host:
        Local Gradio server host (default: "127.0.0.1").
    port:
        Local Gradio server port.
    share:
        Create a public Gradio share link.
    open_browser:
        Automatically open the browser on launch.
    """
    gr = _get_gr()
    runs_dir = str(Path(runs_dir).expanduser())
    app = build_app(default_runs_dir=runs_dir)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=open_browser,
        theme=gr.themes.Soft(),
        css=DASHBOARD_CSS,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pl_inspector.ui.app",
        description="Launch the pl-inspector Gradio dashboard.",
    )
    parser.add_argument(
        "--runs-dir",
        default=DEFAULT_RUNS_DIR,
        metavar="DIR",
        help=f"Root directory containing the runs/ sub-directory (default: {DEFAULT_RUNS_DIR!r}).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Local Gradio server host (default: '127.0.0.1').",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Local Gradio server port (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public Gradio share link (requires internet).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Suppress automatic browser launch.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — called by ``python -m pl_inspector.ui.app``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    launch(
        runs_dir=args.runs_dir,
        host=args.host,
        port=args.port,
        share=args.share,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
