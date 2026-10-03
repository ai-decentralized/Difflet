"""Count dot / convert ops by element type in cached HLO module protos (read-only).

    python hlo_dots2.py <neuron-compile-cache-dir> [MODULE_... ...]

Without module names: every module in the cache dir whose HLO has >= 50 dots, newest first.
Loads torch_neuronx's generated protobuf modules without executing the package __init__.
"""
import collections
import glob
import json
import os
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

C = sys.argv[1]
NAMES = {v.number: v.name for v in xla_data_pb2.PrimitiveType.DESCRIPTOR.values}
mods = sys.argv[2:]
if not mods:
    mods = sorted((d for d in os.listdir(C) if d.startswith("MODULE_")),
                  key=lambda d: os.path.getmtime(os.path.join(C, d)), reverse=True)


def analyse(mod):
    m = hlo_pb2.HloModuleProto()
    m.ParseFromString(open(f"{C}/{mod}/model.hlo_module.pb", "rb").read())
    dots, converts, types_ = collections.Counter(), collections.Counter(), collections.Counter()
    n_ins = 0
    for comp in m.computations:
        by_id = {ins.id: ins for ins in comp.instructions}
        for ins in comp.instructions:
            n_ins += 1
            et = NAMES.get(ins.shape.element_type, "?")
            types_[et] += 1
            if ins.opcode == "dot":
                ops = tuple(NAMES.get(by_id[o].shape.element_type, "?") if o in by_id else "?" for o in ins.operand_ids)
                dots[(ops, et)] += 1
            elif ins.opcode == "convert":
                o = ins.operand_ids[0] if ins.operand_ids else None
                src = NAMES.get(by_id[o].shape.element_type, "?") if o in by_id else "?"
                converts[(src, et)] += 1
    return len(m.computations), n_ins, dots, converts, types_


for mod in mods:
    try:
        ncomp, n_ins, dots, converts, types_ = analyse(mod)
    except Exception as exc:  # noqa: BLE001
        print(f"=== {mod}: unreadable ({exc})")
        continue
    if sum(dots.values()) < 50 and len(sys.argv) <= 2:
        continue
    flags = ""
    try:
        flags = json.dumps(json.load(open(f"{C}/{mod}/compile_flags.json")))[:300]
    except Exception:  # noqa: BLE001
        pass
    print(f"=== {mod}: {ncomp} computations, {n_ins} instructions, {sum(dots.values())} dots")
    print("  flags:", flags)
    print("  element types:", dict(types_.most_common(8)))
    print("  dot (operand types -> result):")
    for k, v in dots.most_common(10):
        print(f"    {v:5d}  {k[0]} -> {k[1]}")
    print("  convert (src -> dst):")
    for k, v in converts.most_common(10):
        print(f"    {v:5d}  {k[0]} -> {k[1]}")
