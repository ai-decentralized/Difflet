"""Thin Wan adapter onto the unified gate (cclog 92).

Validates that ``difflet.pipeline.teacache_gate.run_gate`` — the model-agnostic gate core —
reproduces the existing per-model Wan gate when driven by the REAL transformer. Reuses
the existing gate's heavy helpers (transformer load, block-0 hook, inputs); the only new
code is the 3 thin callables (init_latent / step_fn / advance_fn). Random embeds (matches
the original gate's ~0.99 random-embed Pearson — a wiring check, not the production calib;
the production online_delta calib is emitted separately, cclog 91/92).

Run:
  PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH DIFFLET_BACKEND=cpu \
  PYTHONPATH=/home/ubuntu/difflet \
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python scripts/teacache_gate_wan.py --steps 16
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from wan_teacache_cpu_gate import (  # noqa: E402
    WAN_MODEL_ID,
    WAN_SUBFOLDER,
    Block0Capture,
    build_transformer,
    make_inputs,
    resolve_transformer_dir,
)

from difflet.pipeline.teacache_gate import run_gate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--prompt", default=None,
                    help="real UMT5 embeds (cclog 89 honest path); omit = fixed-random wiring check")
    args = ap.parse_args()
    dtype = torch.bfloat16

    tdir = resolve_transformer_dir(WAN_MODEL_ID, WAN_SUBFOLDER)
    model = build_transformer(tdir, dtype)
    latents, ehs = make_inputs(dtype, 226, model.config.text_dim,
                               prompt=args.prompt, transformer_dir=tdir)
    print(f"[wan-adapter] embeds={'REAL' if args.prompt else 'random'} "
          f"prompt={args.prompt!r}", flush=True)
    capture = Block0Capture(model.blocks[0])
    sigmas = torch.linspace(1.0, 0.0, args.steps + 1, dtype=torch.float32)
    print(f"[wan-adapter] model loaded; running run_gate for {args.steps} steps", flush=True)

    def init_latent():
        return latents.clone()

    def step_fn(i, x):
        timestep = torch.full((x.shape[0],), sigmas[i].item() * 1000.0, dtype=dtype)
        with torch.no_grad():
            np = model(x, timestep, ehs)
        np = np if torch.is_tensor(np) else np[0]
        return np.float(), capture.modulated_input()

    def advance_fn(x, np, i):
        return x - (sigmas[i].item() - sigmas[i + 1].item()) * np.to(dtype)

    cal, summ = run_gate(
        model="wan", shape_label="832x480x13", num_steps=args.steps,
        init_latent=init_latent, step_fn=step_fn, advance_fn=advance_fn,
    )
    capture.close()
    print(f"[wan-adapter] run_gate summary: {summ}", flush=True)
    print(f"[wan-adapter] -> method={summ['method']} probe_pearson={summ['probe_pearson']} "
          f"delta_autocorr={summ['delta_autocorr']}", flush=True)
    print(f"[wan-adapter] calib fields: online_delta_alpha={cal.online_delta_alpha} "
          f"cadence={cal.cadence} accumulate={cal.accumulate} fit_r2={cal.fit_r2}", flush=True)
    print("[wan-adapter] DONE", flush=True)


if __name__ == "__main__":
    main()
