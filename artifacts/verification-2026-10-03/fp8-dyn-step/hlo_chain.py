"""Print the producer chain feeding the first N fp8 converts and the consumer chain after the first N fp8 dots.

    python hlo_chain.py <cache-dir> <MODULE_...> [n]
"""
import glob
import sys
import types

SITE = glob.glob("/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/.venv/lib/python3.12/site-packages")[0]
for name, path in (("torch_neuronx", f"{SITE}/torch_neuronx"), ("torch_neuronx.pyhlo", f"{SITE}/torch_neuronx/pyhlo"),
                   ("torch_neuronx.pyhlo.service", f"{SITE}/torch_neuronx/pyhlo/service")):
    stub = types.ModuleType(name)
    stub.__path__ = [path]
    sys.modules[name] = stub
from torch_neuronx.pyhlo.service import hlo_pb2  # noqa: E402
from torch_neuronx.pyhlo import xla_data_pb2  # noqa: E402

C, MOD = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NAMES = {v.number: v.name for v in xla_data_pb2.PrimitiveType.DESCRIPTOR.values}
m = hlo_pb2.HloModuleProto()
m.ParseFromString(open(f"{C}/{MOD}/model.hlo_module.pb", "rb").read())
comp = max(m.computations, key=lambda c: len(c.instructions))
by_id = {ins.id: ins for ins in comp.instructions}
users = {}
for ins in comp.instructions:
    for o in ins.operand_ids:
        users.setdefault(o, []).append(ins.id)


def fmt(ins):
    return f"{ins.opcode:14s} {NAMES.get(ins.shape.element_type, '?'):9s} {list(ins.shape.dimensions)} #{ins.id}"


def up(ins, depth, seen):
    if depth == 0 or ins.id in seen:
        return
    seen.add(ins.id)
    for o in ins.operand_ids:
        if o in by_id:
            p = by_id[o]
            print("    " * (4 - depth) + "<- " + fmt(p))
            up(p, depth - 1, seen)


def down(ins, depth, seen):
    if depth == 0 or ins.id in seen:
        return
    seen.add(ins.id)
    for u in users.get(ins.id, []):
        c = by_id[u]
        print("    " * (4 - depth) + "-> " + fmt(c))
        down(c, depth - 1, seen)


print(f"computation {comp.name}: {len(comp.instructions)} instructions")
shown = 0
for ins in comp.instructions:
    if ins.opcode == "convert" and NAMES.get(ins.shape.element_type) == "F8E4M3FN" and len(ins.shape.dimensions) == 3:
        print("\n=== quantize convert:", fmt(ins))
        up(ins, 4, set())
        shown += 1
        if shown >= N:
            break
shown = 0
for ins in comp.instructions:
    if ins.opcode == "dot" and NAMES.get(ins.shape.element_type) == "F8E4M3FN":
        print("\n=== dot:", fmt(ins))
        down(ins, 4, set())
        shown += 1
        if shown >= N:
            break
