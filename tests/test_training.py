import json
from pathlib import Path

import pytest
try:
    import torch
except ImportError:
    torch = None

from pl_inspector.exceptions import TrainingInterfaceError
from pl_inspector.training import trace_training


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_runs"


@pytest.fixture
def toy_model():
    if torch is None:
        pytest.skip("PyTorch is not installed.")
    
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)
            
        def forward(self, x):
            return self.linear(x)
            
    return ToyModel()


def test_unsupported_interface(run_dir):
    """Test that unsupported interfaces are rejected immediately."""
    with pytest.raises(TrainingInterfaceError, match="Unsupported interface"):
        trace_training(interface="jax", save_dir=run_dir)


@pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests.")
def test_attach_model(run_dir, toy_model):
    """Test attaching a Torch model works correctly."""
    session = trace_training(interface="torch", save_dir=run_dir)
    assert session.model is None
    
    session.attach(toy_model)
    assert session.model is toy_model
    assert session._prev_params is None


@pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests.")
def test_log_step_without_attach(run_dir):
    """Test that log_step raises an error if no model is attached."""
    session = trace_training(interface="torch", save_dir=run_dir)
    with pytest.raises(RuntimeError, match="Must call attach"):
        session.log_step(loss=0.5, step=1)


@pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests.")
def test_log_step_and_metrics(run_dir, toy_model):
    """Test logging steps, gradient norms, and parameter updates."""
    session = trace_training(interface="torch", save_dir=run_dir)
    session.attach(toy_model)
    
    optimizer = torch.optim.SGD(toy_model.parameters(), lr=0.1)
    
    # Step 1: Forward and backward
    x = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0]])
    
    output = toy_model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    session.log_step(loss=loss.item(), step=0, extra_metrics={"custom": 42})
    
    # Check that gradient norm is captured but update norm is not (first step)
    assert len(session._records) == 1
    record1 = session._records[0]
    assert record1.global_step == 0
    assert record1.loss == loss.item()
    assert record1.metrics["custom"] == 42
    assert record1.grad_norm is not None
    assert record1.grad_norm > 0
    assert "update_norm" not in record1.metrics
    
    # Perform optimization step
    optimizer.step()
    optimizer.zero_grad()
    
    # Step 2: Forward, backward
    output = toy_model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    # Log the second step
    session.log_step(loss=loss.item(), step=1)
    
    assert len(session._records) == 2
    record2 = session._records[1]
    assert record2.global_step == 1
    
    # After optimization step, the parameters should have been updated
    # meaning the update_norm metric should be present and > 0
    assert "update_norm" in record2.metrics
    assert record2.metrics["update_norm"] > 0
    assert record2.grad_norm is not None


@pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests.")
def test_export_json_and_storage(run_dir, toy_model):
    """Test that export_json writes the summary file correctly and layout is consistent."""
    session = trace_training(interface="torch", save_dir=run_dir)
    session.attach(toy_model)
    
    # Create an artificial step
    x = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0]])
    output = toy_model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    session.log_step(loss=loss.item(), step=0)
    
    out_dir = session.export_json()
    
    assert out_dir.exists()
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "gradients.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    
    # Read summary and check contents
    with open(out_dir / "summary.json", "r") as f:
        summary = json.load(f)
        
    assert summary["run_id"] == session.run_id
    assert summary["type"] == "training_diagnostics"
    assert summary["num_steps"] == 1
    assert "first_step_timestamp" in summary
    assert "last_step_timestamp" in summary
