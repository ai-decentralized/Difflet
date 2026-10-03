"""Print the full producer tree of the first dynamic activation scale (the F32 scalar that
feeds the quantize multiply) in a cached HLO module.

    python hlo_scale.py <cache-dir> <MODULE_...> [depth]
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
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 10
NAMES = {v.number: v.name for v in xla_data_pb2.PrimitiveType.DESCRIPTOR.values}
m = hlo_pb2.HloModuleProto()
m.ParseFromString(open(f"{C}/{MOD}/model.hlo_module.pb", "rb").read())
comps = {c.id: c for c in m.computations}
comp = max(m.computations, key=lambda c: len(c.instructions))
by_id = {ins.id: ins for ins in comp.instructions}


def fmt(ins):
    extra = ""
    if ins.opcode == "reduce":
        sub = comps.get(ins.called_computation_ids[0]) if ins.called_computation_ids else None
        ops = sorted({i.opcode for i in sub.instructions} - {"parameter"}) if sub else []
        extra = f" dims={list(ins.dimensions)} reducer={ops}"
    if ins.opcode == "constant" and ins.literal.f32s:
        extra = f" = {list(ins.literal.f32s)[:4]}"
    return f"{ins.opcode:14s} {NAMES.get(ins.shape.element_type, '?'):9s} {list(ins.shape.dimensions)} #{ins.id}{extra}"


def up(ins, depth, indent):
    if depth == 0:
        return
    for o in ins.operand_ids:
        p = by_id.get(o)
        if p is None:
            continue
        print("  " * indent + "<- " + fmt(p))
        if p.opcode in ("broadcast", "convert", "multiply", "divide", "maximum", "minimum", "negate", "reduce",
                        "clamp", "add", "subtract", "select", "compare", "abs", "reshape", "copy", "bitcast"):
            up(p, depth - 1, indent + 1)


for ins in comp.instructions:
    if ins.opcode == "convert" and NAMES.get(ins.shape.element_type) == "F8E4M3FN" and len(ins.shape.dimensions) == 3:
        mul = by_id[ins.operand_ids[0]]
        while mul.opcode in ("clamp",):
            mul = by_id[mul.operand_ids[0]]
        print("quantize multiply:", fmt(mul))
        # the scalar operand is the broadcast one
        for o in mul.operand_ids:
            p = by_id[o]
            if p.opcode == "broadcast":
                print("scale tree:")
                print("<- " + fmt(p))
                up(p, DEPTH, 1)
        break
