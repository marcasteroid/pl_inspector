import pennylane as qml
from pl_inspector import watch
from pathlib import Path

# Set up a simple device and QNode
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit_a(x):
    qml.RX(x, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def circuit_b(y):
    qml.RY(y, wires=1)
    return qml.expval(qml.PauliX(1))

runs_dir = Path("demo_runs")

print("Generating run with events...")
with watch(circuit_a, save_dir=runs_dir) as run:
    # Multiple calls to create some counts
    for i in range(5):
        run(0.1 * i)
    
    # Switch to another QNode within the same logical "run" (if supported by the test script)
    # Actually watch() wraps a single QNode. To have multiple QNodes in one run, 
    # we would need to manually use storage and trace wrapper.
    
# Let's do another one to compare
print("Generating second run for comparison...")
with watch(circuit_b, save_dir=runs_dir) as run2:
    for i in range(3):
        run2(0.5)

print("Done! Runs saved to demo_runs/runs/")
