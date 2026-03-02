from pl_inspector.trace import wrap_qnode
from pl_inspector.utils import generate_run_id
from pl_inspector.models import ExecutionEvent


def dummy_qnode(x, scale=2.0):
    return x * scale


events: list[ExecutionEvent] = []


def on_event(ev: ExecutionEvent) -> None:
    events.append(ev)
    print(ev.to_json_dict())


run_id = generate_run_id()
wrapped = wrap_qnode(dummy_qnode, run_id=run_id, on_event=on_event)

print("Output:", wrapped(1.23, scale=3.0))
print("Recorded events:", len(events))