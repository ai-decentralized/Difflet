"""Count all-reduce ops by element type and shape in compiled HLO modules.

Usage: python hlo_allreduce.py <hlo_module.pb> [...]
Prints one JSON line per module: {"module": path, "all_reduce": {"bf16 [4096,3072]": n, ...}}.
"""

import json
import sys
from collections import Counter

from libneuronxla.proto import hlo_pb2, xla_data_pb2


def _shape_str(shape) -> str:
    if shape.element_type == xla_data_pb2.TUPLE:
        return "(" + ", ".join(_shape_str(s) for s in shape.tuple_shapes) + ")"
    name = xla_data_pb2.PrimitiveType.Name(shape.element_type).lower()
    return f"{name} [{','.join(str(d) for d in shape.dimensions)}]"


def all_reduces(path: str) -> Counter:
    module = hlo_pb2.HloModuleProto()
    with open(path, "rb") as handle:
        module.ParseFromString(handle.read())
    counts: Counter = Counter()
    for computation in module.computations:
        for instruction in computation.instructions:
            if instruction.opcode in ("all-reduce", "all-reduce-start"):
                counts[_shape_str(instruction.shape)] += 1
    return counts


for path in sys.argv[1:]:
    print(json.dumps({"module": path, "all_reduce": dict(sorted(all_reduces(path).items()))}))
