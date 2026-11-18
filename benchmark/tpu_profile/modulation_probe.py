"""Is FLUX's sharded AdaLN modulation (``DIFFLET_TPU_FLUX_SHARD_MODULATION``)
only a rounding change? On one TPU chip, compiled, from real FLUX.1-dev weights:

    python -m benchmark.tpu_profile.modulation_probe [--tp 4]

For a joint block's ``norm1.linear`` [18432, 3072] and a single block's
``norm.linear`` [9216, 3072], computes ``linear(silu(temb))`` in bf16 two ways --
the replicated weight (as without the switch: XLA lowers this GEMV as a
reduce fusion) and the concatenation of ``tp`` row shards (as each rank does
with it) -- and compares both against a float64 host reference of the same bf16
inputs. Equal-sized errors mean the switch moves rounding, not values; the
end-to-end PSNR between the two settings is then trajectory amplification of
that rounding (as for the Pallas kernel, docs/plans/2026-11-10-...).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

MODEL = "/mnt/models/hf/hub/models--black-forest-labs--FLUX.1-dev"
KEYS = ("transformer_blocks.0.norm1.linear", "single_transformer_blocks.0.norm.linear")


def _load(prefix: str):
    from safetensors import safe_open

    snapshot = next(p for p in sorted(Path(MODEL, "snapshots").iterdir())
                    if (p / "transformer").is_dir())
    index = json.loads((snapshot / "transformer/diffusion_pytorch_model.safetensors.index.json")
                       .read_text())["weight_map"]
    out = {}
    for kind in ("weight", "bias"):
        key = f"{prefix}.{kind}"
        with safe_open(str(snapshot / "transformer" / index[key]), framework="pt") as f:
            out[kind] = f.get_tensor(key)
    return out["weight"], out["bias"]


def _worker(tp: int, reply_q) -> None:
    try:
        from difflet.backends.tpu.ops_impl import device_runtime

        device_runtime.initialize_replica(0, 1)
        import torch
        import torch.nn.functional as F

        dev = device_runtime.device()
        torch.manual_seed(0)
        # temb's scale: FLUX's time_text_embed output is O(1) per channel.
        temb = torch.randn(1, 3072).to(torch.bfloat16)

        def replicated(x, w, b):
            return F.linear(F.silu(x), w, b)

        def sharded(x, *wb):
            act = F.silu(x)
            return torch.cat([F.linear(act, wb[2 * i], wb[2 * i + 1]) for i in range(tp)], dim=1)

        rep_c = torch.compile(replicated, backend="tpu")
        shard_c = torch.compile(sharded, backend="tpu")
        rows = []
        for prefix in KEYS:
            w, b = _load(prefix)
            w, b = w.to(torch.bfloat16), b.to(torch.bfloat16)
            ref = F.linear(F.silu(temb.double()), w.double(), b.double())
            n = w.shape[0] // tp
            parts = []
            for i in range(tp):
                parts += [w[i * n:(i + 1) * n].contiguous().to(dev), b[i * n:(i + 1) * n].to(dev)]
            x = temb.to(dev)
            rep = rep_c(x, w.to(dev), b.to(dev)).cpu().double()
            sh = shard_c(x, *parts).cpu().double()
            scale = ref.abs().max()

            def err(got):
                d = (got - ref).abs()
                return {"max_rel": float(d.max() / scale), "rms_rel": float(d.norm() / ref.norm())}

            rows.append({"linear": prefix, "shape": list(w.shape),
                         "replicated_vs_fp64": err(rep), "sharded_vs_fp64": err(sh),
                         "replicated_vs_sharded_differ": int((rep != sh).sum()),
                         "elements": rep.numel()})
        reply_q.put({"rows": rows})
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        reply_q.put({"error": repr(exc)})


def main() -> None:
    from difflet.backends.tpu.ops_impl import device_runtime

    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=4)
    args = parser.parse_args()
    device_runtime.prepare_launch(1)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(args.tp, reply_q))
    proc.start()
    result = reply_q.get(timeout=1800)
    proc.join(timeout=60)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
