"""Torch-first training diagnostics.

This module provides the :func:`trace_training` entry point and the associated
session object for tracking Torch model training. It includes tracking of
losses, gradient norms, parameter updates, and basic vanishing-gradient detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .exceptions import TrainingInterfaceError
from .models import RunMeta, TrainingStepRecord
from .storage import RunStorage
from .utils import ensure_dir, generate_run_id

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


class TrainingSession:
    """Session object for tracking PyTorch model training diagnostics.

    Parameters
    ----------
    save_dir:
        Directory for storing run outputs.
    """

    def __init__(self, save_dir: str | Path) -> None:
        if torch is None:
            raise TrainingInterfaceError(
                "PyTorch is not installed. The training diagnostics module requires torch."
            )

        self._save_dir = Path(save_dir)
        ensure_dir(self._save_dir)

        self.run_id: str = generate_run_id()
        self.storage: RunStorage = RunStorage(base_dir=self._save_dir, run_id=self.run_id)

        meta = RunMeta(
            run_id=self.run_id,
            description="trace_training() session",
            extra={"interface": "torch"}
        )
        self.storage.write_meta(meta)

        self.model: Optional["torch.nn.Module"] = None
        self._prev_params: Optional[List["torch.Tensor"]] = None
        self._records: List[TrainingStepRecord] = []
        self._consecutive_small_grads: int = 0
        self._vanishing_grad_threshold: float = 1e-4

    def attach(
        self,
        model: "torch.nn.Module",
        qnodes: Optional[Sequence[Any]] = None
    ) -> "TrainingSession":
        """Attach a PyTorch model to the tracking session.

        Args:
            model: The PyTorch model to track.
            qnodes: Optional list of specific QNodes to track (currently unused, 
                reserved for future granular tracking).
        """
        if not isinstance(model, torch.nn.Module):
            raise TrainingInterfaceError(
                f"Expected a torch.nn.Module, got {type(model)}."
            )
        self.model = model
        self._prev_params = None
        return self

    def log_step(
        self,
        loss: float,
        step: int,
        extra_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log metrics, global gradient norm, and parameter update norm for a step.

        Args:
            loss: The loss value for the current step.
            step: The global step index.
            extra_metrics: Additional metrics to log.
        """
        if self.model is None:
            raise RuntimeError("Must call attach() with a model before log_step().")

        metrics = dict(extra_metrics) if extra_metrics else {}

        # 1. Compute global gradient norm
        grad_norm = 0.0
        grads_present = False
        for p in self.model.parameters():
            if p.grad is not None:
                grads_present = True
                grad_norm += p.grad.detach().norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5 if grads_present else None

        # 2. Check for vanishing gradients
        if grad_norm is not None:
            if grad_norm < self._vanishing_grad_threshold:
                self._consecutive_small_grads += 1
            else:
                self._consecutive_small_grads = 0

            if self._consecutive_small_grads >= 3:
                metrics["vanishing_gradient"] = True

        # 3. Compute parameter update norm
        update_norm = None
        current_params = [p.detach().clone() for p in self.model.parameters()]
        
        if self._prev_params is not None:
            upd_norm_sq = 0.0
            for curr_p, prev_p in zip(current_params, self._prev_params):
                upd_norm_sq += (curr_p - prev_p).norm(2).item() ** 2
            update_norm = upd_norm_sq ** 0.5
            metrics["update_norm"] = update_norm

        self._prev_params = current_params

        # 4. Construct and append record
        record = TrainingStepRecord(
            run_id=self.run_id,
            global_step=step,
            loss=float(loss),
            metrics=metrics,
            grad_norm=float(grad_norm) if grad_norm is not None else None,
        )
        self._records.append(record)
        self.storage.append_training_step(record)

    def plot_gradient_norms(self, ax: Optional[Any] = None) -> Any:
        """Plot the history of gradient norms over the tracked steps."""
        if not self._records:
            raise RuntimeError("No training steps logged yet.")
            
        from .plotting import plot_gradient_norm_history
        steps = [r.global_step for r in self._records]
        norms = [r.grad_norm for r in self._records if r.grad_norm is not None]
        
        if not norms:
            raise RuntimeError("No gradient norms tracked (are parameters detached?).")
            
        return plot_gradient_norm_history(norms, steps=steps[:len(norms)], ax=ax)

    def plot_loss(self, ax: Optional[Any] = None) -> Any:
        """Plot the history of loss values over the tracked steps."""
        if not self._records:
            raise RuntimeError("No training steps logged yet.")
            
        from .plotting import plot_loss_curve
        steps = [r.global_step for r in self._records]
        losses = [r.loss for r in self._records if r.loss is not None]
        
        if not losses:
            raise RuntimeError("No loss values tracked.")
            
        return plot_loss_curve(losses, steps=steps[:len(losses)], ax=ax)

    def export_json(self) -> Path:
        """Ensure run summary is written and return the run directory path."""
        summary = {
            "run_id": self.run_id,
            "type": "training_diagnostics",
            "num_steps": len(self._records)
        }
        if self._records:
            summary["first_step_timestamp"] = self._records[0].timestamp
            summary["last_step_timestamp"] = self._records[-1].timestamp
            
        self.storage.write_summary(summary)
        return self.storage.run_dir


def trace_training(
    interface: str = "torch",
    save_dir: str | Path = "./runs",
) -> TrainingSession:
    """Create a new training diagnostics session.

    Args:
        interface: The ML interface to trace. Only "torch" is supported.
        save_dir: Base directory where run data is stored.

    Returns:
        A TrainingSession object to attach models and log metrics.
    """
    if interface != "torch":
        raise TrainingInterfaceError(
            f"Unsupported interface '{interface}'. Only 'torch' is currently supported."
        )
        
    return TrainingSession(save_dir=save_dir)
