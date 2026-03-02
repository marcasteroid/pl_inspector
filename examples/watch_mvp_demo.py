import pennylane as qml
from pl_inspector import watch

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))

with watch(circuit, save_dir="./runs", capture="tracker") as run:
    run.wrapped_qnode(0.1)
    run.wrapped_qnode(0.2)
    run.show_trace()
    run_dir = run.export_json()
    print("Run directory:", run_dir)