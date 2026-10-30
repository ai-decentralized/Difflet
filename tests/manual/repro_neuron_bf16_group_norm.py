"""AWS repro: bf16 ``F.group_norm`` on a neuron device loses accuracy with the group size.

Relative L1 against fp32, 32 groups, one core. The CPU's bf16 stays at ~1.4e-3; the device's grows
with the elements per group (1.8e-2 at 262,144, 0.32 at 1,048,576, 0.71 at 2,097,152, 1.35 at
4,194,304), consistent with bf16 accumulation of the group statistics. Difflet's backend computes
GroupNorm in fp32 for bf16 device inputs (``--with-workaround`` shows that path).

    python tests/manual/repro_neuron_bf16_group_norm.py [--with-workaround]
"""

import os
import sys

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def main() -> int:
    if "--with-workaround" in sys.argv:
        from difflet.backends.registry import get_backend
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        get_backend("neuron").prepare_runtime(DiffletParallelConfig(tp_degree=1))
    else:
        import torch_neuronx  # noqa: F401  (the neuron device, without Difflet's workarounds)
    g = torch.Generator().manual_seed(0)
    for c, hw in ((512, 128), (512, 256), (256, 512), (128, 1024)):
        x = (torch.randn(1, c, hw, hw, generator=g) * 3 + 1).to(torch.bfloat16)
        w, b = torch.randn(c, generator=g).to(torch.bfloat16), torch.randn(c, generator=g).to(torch.bfloat16)
        ref = F.group_norm(x.float(), 32, w.float(), b.float(), 1e-6)
        cpu = F.group_norm(x, 32, w, b, 1e-6).float()
        dev = F.group_norm(x.to("neuron"), 32, w.to("neuron"), b.to("neuron"), 1e-6).float().cpu()

        def err(y):
            return float((y - ref).abs().mean() / ref.abs().mean())

        print(f"C={c:4d} HW={hw:4d} (group size {c // 32 * hw * hw:>9,}): bf16 CPU {err(cpu):.2e} | "
              f"bf16 device {err(dev):.2e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
