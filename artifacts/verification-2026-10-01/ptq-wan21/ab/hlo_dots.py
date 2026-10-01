"""Count dot / convert ops by element type in cached HLO module protos (read-only).

Loads torch_neuronx's generated protobuf modules without executing the
torch_neuronx package __init__ (which initialises the Neuron runtime).
"""
import collections, glob, sys, types

SITE = glob.glob("/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/.venv/lib/python3.12/site-packages")[0]
for name, path in (("torch_neuronx", f"{SITE}/torch_neuronx"), ("torch_neuronx.pyhlo", f"{SITE}/torch_neuronx/pyhlo"),
                   ("torch_neuronx.pyhlo.service", f"{SITE}/torch_neuronx/pyhlo/service")):
    stub = types.ModuleType(name)
    stub.__path__ = [path]
    sys.modules[name] = stub
from torch_neuronx.pyhlo.service import hlo_pb2  # noqa: E402
from torch_neuronx.pyhlo import xla_data_pb2  # noqa: E402

C = "/var/tmp/neuron-compile-cache/neuronxcc-2.26.6360.0+6f180f47"
NAMES = {v.number: v.name for v in xla_data_pb2.PrimitiveType.DESCRIPTOR.values}
MODS = [(a.split("=",1)[0], a.split("=",1)[1]) for a in sys.argv[1:]] or [("fp8", "MODULE_a1cd99ffd9d0c5735511+04929ab2"), ("bf16", "MODULE_f79148cf7ed21271b9e2+f934ae74")]

for tag, mod in MODS:
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
    print(f"=== {tag} {mod}: {len(m.computations)} computations, {n_ins} instructions")
    print("  element types:", dict(types_.most_common(8)))
    print("  dot (operand types -> result):")
    for k, v in dots.most_common(10):
        print(f"    {v:5d}  {k[0]} -> {k[1]}")
    print("  convert (src -> dst):")
    for k, v in converts.most_common(10):
        print(f"    {v:5d}  {k[0]} -> {k[1]}")
