"""Op x element-type histogram of a cached HLO module, restricted to large tensors (read-only).

    python hlo_ops.py <cache-dir> <MODULE_...> [min_numel]

Shows which elementwise / reduce / convert ops run over activation-sized tensors — the
traffic a dynamic-quantization path adds around every dot.
"""
import collections
import glob
import math
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
MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 1e6
NAMES = {v.number: v.name for v in xla_data_pb2.PrimitiveType.DESCRIPTOR.values}
m = hlo_pb2.HloModuleProto()
m.ParseFromString(open(f"{C}/{MOD}/model.hlo_module.pb", "rb").read())
hist = collections.Counter()
elems = collections.Counter()
big_dots = collections.Counter()
for comp in m.computations:
    by_id = {ins.id: ins for ins in comp.instructions}
    for ins in comp.instructions:
        dims = list(ins.shape.dimensions)
        n = math.prod(dims) if dims else 1
        et = NAMES.get(ins.shape.element_type, "?")
        if n >= MIN:
            key = (ins.opcode, et, tuple(dims))
            hist[key] += 1
            elems[(ins.opcode, et)] += n
            if ins.opcode == "dot":
                ops = tuple(NAMES.get(by_id[o].shape.element_type, "?") for o in ins.operand_ids if o in by_id)
                big_dots[(ops, et, tuple(dims))] += 1
print(f"=== {MOD}: ops on tensors with numel >= {MIN:g}")
print("--- count by (op, dtype, shape):")
for (op, et, dims), v in sorted(hist.items(), key=lambda kv: -kv[1])[:40]:
    print(f"  {v:5d}  {op:18s} {et:9s} {list(dims)}")
print("--- total elements written by (op, dtype) [G]:")
for (op, et), v in sorted(elems.items(), key=lambda kv: -kv[1])[:20]:
    print(f"  {v / 1e9:8.2f}  {op:18s} {et}")
print("--- dots:")
for (ops, et, dims), v in big_dots.most_common(12):
    print(f"  {v:5d}  {ops} -> {et} {list(dims)}")
