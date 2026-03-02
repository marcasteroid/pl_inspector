"""Basic tracing example for pl-inspector.

This script demonstrates the minimal end-to-end flow:

- creating a PennyLane device,
- defining a small QNode,
- wrapping it with :func:`pl_inspector.watch`,
- invoking the traced run object,
- printing a human-readable execution trace,
- exporting results to a run directory on disk.

You can run this script directly:

    python examples/basic_trace.py
"""

from __future__ import annotations

import pennylane as qml

from pl_inspector import watch


def main() -> None:
    # 1. Create a simple PennyLane device.
    dev = qml.device("default.qubit", wires=1)

    # 2. Define a tiny QNode we want to trace.
    @qml.qnode(dev)
    def circuit(x: float) -> float:
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    # 3. Use the watch() context manager to trace this circuit.
    #
    #    The yielded object `run` is callable: calling `run(x)` executes a
    #    wrapped version of the QNode that emits structured execution events
    #    and writes them to disk.
    with watch(circuit, save_dir="./runs", capture="tracker") as run:
        y1 = run(0.1)
        y2 = run(0.2)

        print("QNode outputs:", y1, y2)

        # 4. Print a human-readable execution trace.
        run.show_trace()

        # 5. Export results to disk and report the run directory.
        run_dir = run.export_json()
        print("Run data saved under:", run_dir)


if __name__ == "__main__":
    main()


