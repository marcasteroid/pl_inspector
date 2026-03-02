import pennylane as qml
from pl_inspector.tracker import TrackerAdapter
from pl_inspector.utils import generate_run_id

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))

run_id = generate_run_id()
adapter = TrackerAdapter(run_id=run_id, target=dev)
with adapter:
    circuit(0.1)
    circuit(0.2)

record = adapter.to_record()
print(record.to_json_dict())