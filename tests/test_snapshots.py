import numpy as np
import pytest

from pl_inspector.snapshots import (
    is_state_vector_like,
    make_snapshot_record,
    summarize_state_vector,
)
from pl_inspector.watch import _extract_snapshot_payload, watch


def test_is_state_vector_like():
    """Test the heuristics for state-vector-like detection."""
    # Valid state vectors
    assert is_state_vector_like(np.array([1.0, 0.0], dtype=np.complex128))
    assert is_state_vector_like(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64))

    # Invalid shapes or types
    assert not is_state_vector_like([1.0, 0.0])  # Not ndarray
    assert not is_state_vector_like(np.array([[1.0, 0.0], [0.0, 1.0]]))  # 2D
    assert not is_state_vector_like(np.array([1.0, 0.0, 0.0]))  # Not power of 2
    assert not is_state_vector_like(np.array([1, 0], dtype=np.int32))  # Not float/complex


def test_summarize_state_vector():
    """Test top-k amplitude extraction and probability summarization."""
    # 2-qubit state: |01> has highest prob (0.8), |10> and |11> have 0.1
    state = np.array([0.0, np.sqrt(0.8), np.sqrt(0.1), np.sqrt(0.1)], dtype=np.complex128)
    summary = summarize_state_vector(state, top_k=2)

    assert summary.get("type") == "state_vector"
    assert summary.get("num_qubits") == 2
    assert summary.get("dimension") == 4
    
    top_amps = summary.get("top_amplitudes", [])
    assert len(top_amps) == 2
    
    # Check top amplitude (index 1, bitstring '01')
    assert top_amps[0]["index"] == 1
    assert top_amps[0]["bitstring"] == "01"
    assert np.isclose(top_amps[0]["probability"], 0.8)

    # Check max probability metadata
    assert np.isclose(summary.get("max_probability"), 0.8)


def test_summarize_state_vector_fallback():
    """Test fallback to generic summary for non-state-vector arrays."""
    # Pass a 2D array, which fails is_state_vector_like
    data = np.array([[1, 2], [3, 4]])
    summary = summarize_state_vector(data)
    
    assert summary.get("type") != "state_vector"
    assert summary.get("shape") == [2, 2]


def test_make_snapshot_record():
    """Test snapshot record creation maps correctly to summary logic."""
    state = np.array([1.0, 0.0], dtype=np.complex128)
    record = make_snapshot_record(
        run_id="test_run",
        label="my_snapshot",
        data=state,
        metadata={"step": 5},
        top_k=1
    )
    
    assert record.run_id == "test_run"
    assert record.label == "my_snapshot"
    assert record.data_summary.get("type") == "state_vector"
    assert record.metadata.get("step") == 5


def test_extract_snapshot_payload():
    """Test graceful handling and extraction of existing snapshot payloads."""
    # Common format: tuple or list with mapping
    payload_dict = {"state": [1, 0]}
    assert _extract_snapshot_payload((1.0, payload_dict)) == payload_dict
    
    # Common format: dict containing 'snapshots' key
    assert _extract_snapshot_payload({"result": 1.0, "snapshots": payload_dict}) == payload_dict
    
    # Mock class exposing `.snapshots`
    class MockOutput:
        snapshots = payload_dict
    assert _extract_snapshot_payload(MockOutput()) == payload_dict
    
    # Fallback heuristic: dictionary with 'snap'/'snapshot' in key
    assert _extract_snapshot_payload({"my_snap_data": 123}) == {"my_snap_data": 123}

    # Negative cases
    assert _extract_snapshot_payload({"result": 1.0}) is None
    assert _extract_snapshot_payload(None) is None


def test_watch_session_snapshot_capture(tmp_path):
    """Test watch session tracking and persisting basic snapshot behavior."""
    def mock_qnode(*args):
        # A synthetic payload that bypasses PennyLane internals
        state = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.complex128)
        return 0.0, {"psi": state}

    with watch(mock_qnode, save_dir=tmp_path, capture="snapshots") as run:
        result = run()
        
    # Ensure that the original QNode return values are securely untouched
    assert result[0] == 0.0
    assert np.allclose(result[1]["psi"], np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]))
    
    assert run._snapshot_calls_with_payload == 1
    assert len(run._snapshot_records) == 1
    
    record = run._snapshot_records[0]
    assert record.label == "psi"
    assert record.data_summary.get("type") == "state_vector"
    assert record.data_summary.get("dimension") == 4
