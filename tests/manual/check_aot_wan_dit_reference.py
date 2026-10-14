"""AoT (Trainium, NxD) Wan DiT on the oracle's fixed inputs: its parity, and saved outputs.

Run on the AoT stack (scripts/setup_env.sh venv), single process, after ``difflet compile`` has
built the transformer at the oracle's shape (480x832x9 for the default oracle):

    PYTHONPATH=. python tests/manual/check_aot_wan_dit_reference.py \\
        --ref <oracle_high.pt> --out-dir <dir> [--model-id ... --revision ...]

Loads the transformer artifact the CLI compiled (located through the CLI's own parser and
cache layout, as the AoT transformer stage does), runs the DiT once per oracle timestep on the
oracle's latents and text, and judges each output against the oracle's fp32 reference and bf16
control with the same parity gate as the neuron preflight (tests/numerical/neuron_parity.py).
Each output is saved with ``save_reference`` (safetensors plus the build), so the neuron DiT can
later be compared with AoT's at identical inputs without the AoT stack. AoT's CLI holds the
high-noise expert only, so pass the ``transformer`` oracle (timesteps above the boundary).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from tests.numerical import neuron_parity as parity


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ref", type=Path, required=True, help="oracle .pt (tpu_oracle_reference wan)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model-id", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    p.add_argument("--revision", default="5be7df9619b54f4e2667b2755bc6a756675b5cd7")
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()

    blob = torch.load(args.ref, map_location="cpu", weights_only=False)
    if blob.get("expert", "transformer") != "transformer":
        p.error("AoT's CLI holds the high-noise expert only: pass a transformer/ oracle")
    hidden, text = blob["hidden_states"], blob["encoder_hidden_states"]
    frames = int(blob["frames"])
    _, _, _, lat_h, lat_w = hidden.shape
    h, w = lat_h * 8, lat_w * 8

    from difflet.cli.main import _build_parser
    from difflet.cli.orchestrators.wan import WanOrchestrator
    from difflet.models.wan.application import NeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from difflet.pipeline.path_resolver import resolve_model_path

    argv = ["generate", "--model-id", args.model_id, "--revision", args.revision,
            "--tp-degree", str(args.tp_degree), "--height", str(h), "--width", str(w),
            "--num-frames", str(frames)]
    if args.cache_dir:
        argv += ["--cache-dir", args.cache_dir]
    cli_args = _build_parser().parse_args(argv)
    cli_args.stage_mode = "generate"
    compiled_dir = WanOrchestrator(cli_args)._stage_compiled_dir("transformer", cli_args)
    parallel = DiffletParallelConfig(tp_degree=args.tp_degree)
    app = NeuronWanApplication(
        model_path=resolve_model_path(args.model_id, revision=args.revision, local_files_only=True),
        parallel=parallel, dtype=torch.bfloat16,
        shape={"height": h, "width": w, "num_frames": frames}, text_seq_len=text.shape[1],
        batch_size=1, enable_text_encoder=False, enable_transformer=True,
        enable_transformer_2=False, enable_vae_decoder=False,
    )
    app.load(str(compiled_dir), start_rank_id=0, local_ranks_size=parallel.world_size,
             skip_warmup=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for run in blob["runs"]:
        ts = float(run["timestep"][0])
        with torch.no_grad():
            out = app(hidden.to(torch.bfloat16), run["timestep"], text.to(torch.bfloat16))
        out = (out[0] if isinstance(out, (tuple, list)) else out).float().cpu()
        m = parity.metrics(out, run["reference"], control=run["reference_bf16"])
        m["vs_bf16"] = parity.metrics(out, run["reference_bf16"])
        passed, reasons = parity.check(m, control=parity.metrics(run["reference_bf16"],
                                                                  run["reference"]))
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL'} AoT parity t={ts:g} cos={m['cosine']:.6f} "
              f"rel_l1={m['rel_l1']:.3e} {reasons}", flush=True)
        parity.save_reference(args.out_dir / f"aot_dit_t{ts:g}", {"output": out},
                              {"backend": "trainium", "timestep": ts, "oracle": str(args.ref),
                               "metrics": m, "passed": passed})
    print(f"saved under {args.out_dir}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
