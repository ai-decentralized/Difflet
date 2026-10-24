"""HunyuanVideo 1.0 end to end: our neuron application's denoising loop on the device against
diffusers' full ``HunyuanVideoPipeline``, latents compared.

Phase 1, host CPU (no device, no torchrun): diffusers' ``HunyuanVideoPipeline.encode_prompt``
encodes the prompt once in bf16, as our rank-0 host encoder does (Llama with the template, the
95-token crop and hidden layer -3, right-padded to 256 tokens; CLIP's pooled output). From those
embeddings diffusers' own pipeline (FlowMatch Euler, guidance 6.0 x 1000 in the timestep
embedding) runs ``--steps`` steps (default 3) from seed 42 at ``--frames`` x ``--height`` x
``--width`` (default 5 x 320 x 512), ``output_type="latent"``, with the transformer in fp32 (the
reference) and in bf16 (the control). ``--layers`` / ``--single-layers`` cut the transformer to
its first blocks (default the whole model: about 52 GB of fp32 host RAM; the text encoders are
dropped before it loads).

    PYTHONPATH=. python tests/manual/check_neuron_hv1_e2e.py --make-ref W/hv1_e2e_ref.pt

Phase 2, device: ``TorchNeuronHunyuanVideoApplication`` (the registry's neuron application:
the DiT on ``tp x cp`` cores, ``HunyuanVideoOrchestrator`` on every rank, initial noise drawn
on rank 0 from the same seed as diffusers' ``prepare_latents``) from the same embeddings, after
``prepare_runtime``. Gate (``tests/numerical/neuron_parity.py`` ``check``): rel-L1 of our
latents vs the fp32 reference at most 1.15x the bf16 control's, cosine at least 0.995, no CPU
fallback. ``--encode`` runs our own host encoder on rank 0 as well (loads Llama on rank 0's host)
and reports its embeddings against the reference's. Launched without torchrun, the script
builds the torchrun command for the layout and runs it (``--dry-run`` prints it):

    PYTHONPATH=. python tests/manual/check_neuron_hv1_e2e.py --check W/hv1_e2e_ref.pt \\
        --tp-degree 2 --out-json W/e2e-tp2.json

The reference's shape, steps, depth and prompt win over the flags. Exit status 0 only when
every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["DIFFLET_BACKEND"] = "neuron"

import torch

from tests.manual.check_neuron_hv1_dit import (
    DEFAULT_MODEL_DIR,
    config_for,
    launch,
    layout_name,
    resolve_layout,
    transformer_dir,
)
from tests.numerical import neuron_parity as parity

H, W, FRAMES, STEPS, SEED = 320, 512, 5, 3, 42
GUIDANCE_SCALE = 6.0
TEXT_SEQ_LEN = 256
PROMPT = "A cat walks on the grass, realistic style."
EMBED_NAMES = ("prompt_embeds", "prompt_attention_mask", "pooled_prompt_embeds")


def _quiet_diffusers() -> None:
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.set_verbosity_error()


def encode(model_dir, prompt: str) -> dict[str, torch.Tensor]:
    """diffusers' bf16 ``encode_prompt``, the call our host encoder makes."""
    from diffusers import HunyuanVideoPipeline

    enc = HunyuanVideoPipeline.from_pretrained(str(model_dir), transformer=None, vae=None,
                                               torch_dtype=torch.bfloat16)
    with torch.no_grad():
        embeds, pooled, mask = enc.encode_prompt(prompt=prompt, prompt_2=None,
                                                 device=torch.device("cpu"),
                                                 max_sequence_length=TEXT_SEQ_LEN)
    return {"prompt_embeds": embeds.to(torch.bfloat16).contiguous(),
            "prompt_attention_mask": mask.to(torch.int64).contiguous(),
            "pooled_prompt_embeds": pooled.to(torch.bfloat16).contiguous()}


def make_ref(args) -> int:
    from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

    _quiet_diffusers()
    torch.set_num_threads(args.threads)
    config = config_for(args.model_dir, height=args.height, width=args.width, frames=args.frames,
                        layers=args.layers, single_layers=args.single_layers)
    layers, single = int(config.num_layers), int(config.num_single_layers)
    seconds: dict[str, float] = {}
    t0 = time.perf_counter()
    embeds = encode(args.model_dir, args.prompt)
    seconds["encode"] = time.perf_counter() - t0
    valid = int(embeds["prompt_attention_mask"].sum())
    print(f"encoded {args.prompt!r}: {valid} valid of {TEXT_SEQ_LEN} tokens "
          f"({seconds['encode']:.0f} s)", flush=True)

    t0 = time.perf_counter()
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        str(transformer_dir(args.model_dir)), num_layers=layers, num_single_layers=single,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).eval().requires_grad_(False)
    pipe = HunyuanVideoPipeline.from_pretrained(
        str(args.model_dir), transformer=transformer, vae=None, text_encoder=None,
        text_encoder_2=None, tokenizer=None, tokenizer_2=None, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    seconds["load"] = time.perf_counter() - t0
    out: dict[str, Any] = {}
    for name, dtype in (("ref_fp32", torch.float32), ("control_bf16", torch.bfloat16)):
        pipe.transformer.to(dtype)
        t0 = time.perf_counter()
        with torch.no_grad():
            latents = pipe(prompt_embeds=embeds["prompt_embeds"].to(dtype),
                           pooled_prompt_embeds=embeds["pooled_prompt_embeds"].to(dtype),
                           prompt_attention_mask=embeds["prompt_attention_mask"],
                           height=config.height, width=config.width,
                           num_frames=config.num_frames, num_inference_steps=args.steps,
                           guidance_scale=GUIDANCE_SCALE,
                           generator=torch.Generator().manual_seed(SEED),
                           output_type="latent", return_dict=False)[0]
        out[name] = latents.float()
        seconds[name] = time.perf_counter() - t0
        print(f"{name}: {seconds[name]:.0f} s, finite={bool(torch.isfinite(out[name]).all())}, "
              f"shape {tuple(out[name].shape)}, absmean {float(out[name].abs().mean()):.4f}",
              flush=True)
    control = parity.metrics(out["control_bf16"], out["ref_fp32"])
    print(f"control vs fp32: rel_l1={control['rel_l1']:.3e} cos={control['cosine']:.6f}",
          flush=True)
    path = Path(args.make_ref).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "hv1_e2e_ref", "embeds": embeds, **out, "control_metrics": control,
        "prompt": args.prompt, "seed": SEED, "steps": int(args.steps),
        "guidance_scale": GUIDANCE_SCALE, "height": int(config.height),
        "width": int(config.width), "num_frames": int(config.num_frames),
        "num_layers": layers, "num_single_layers": single,
        "checkpoint_num_layers": int(config.checkpoint_num_layers),
        "checkpoint_num_single_layers": int(config.checkpoint_num_single_layers),
        "valid_text": valid, "model_dir": str(args.model_dir), "seconds": seconds,
    }, path)
    print(f"wrote {path}: layers={layers}+{single} {config.num_frames}x{config.height}x"
          f"{config.width} steps={args.steps} seed={SEED}", flush=True)
    return 0


def _cut_transformer(app, *, layers: int, single: int) -> None:
    """Swap ``app``'s DiT for one cut to the reference's depth, before ``load``. The
    application has no depth option (production runs the whole model); the DiT application
    takes the cut config, and the adapter is built from ``app.transformer`` at load."""
    from difflet.backends.neuron.hunyuan_video.config import NeuronHunyuanVideoConfig
    from difflet.backends.neuron.hunyuan_video.transformer import (
        NeuronHunyuanVideoTransformerApplication,
    )

    full = app.config
    if (layers, single) == (full.num_layers, full.num_single_layers):
        return
    config = NeuronHunyuanVideoConfig(**{**full.__dict__, "num_layers": layers,
                                         "num_single_layers": single})
    old = app.transformer
    app.transformer = NeuronHunyuanVideoTransformerApplication(
        model_path=old.model_path, config=config, parallel=old.parallel, dtype=old.dtype,
        exec_mode=old.exec_mode, device=old.device)
    app.config = app.transformer.config


def check(args) -> int:
    t_start = time.perf_counter()
    ref = torch.load(Path(args.check).expanduser(), weights_only=False)
    if ref.get("kind") != "hv1_e2e_ref":
        raise SystemExit(f"{args.check} is not a check_neuron_hv1_e2e.py reference")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tp, cp = resolve_layout(args, world)
    cp_mode = args.cp_mode if cp > 1 else "gather_kv"
    layout = layout_name(tp, cp, cp_mode, args.sp)

    from difflet.backends.neuron.compile import configure_compile_cache

    configure_compile_cache()
    import torch_neuronx  # noqa: F401  (registers the device; the runtime starts later)
    import torch_neuronx.metrics as neuron_metrics

    neuron_metrics.set_enabled(True)

    from difflet.backends.neuron.runtime import track_fallbacks
    from difflet.backends.registry import get_backend
    from difflet.models.hunyuan_video.neuron_application import (
        TorchNeuronHunyuanVideoApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    parallel = DiffletParallelConfig(tp_degree=tp, cp_degree=cp, cp_mode=cp_mode,
                                     sp_enabled=bool(args.sp))
    get_backend("neuron").prepare_runtime(parallel)  # core binding + the numerics workarounds
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    say = print if rank == 0 else (lambda *a, **k: None)
    app = TorchNeuronHunyuanVideoApplication(
        model_path=str(Path(args.model_dir).expanduser()), parallel=parallel,
        dtype=torch.bfloat16,
        shape={"height": ref["height"], "width": ref["width"], "num_frames": ref["num_frames"]},
        text_seq_len=TEXT_SEQ_LEN, exec_mode=args.exec_mode, device="neuron")
    _cut_transformer(app, layers=int(ref["num_layers"]), single=int(ref["num_single_layers"]))
    if not args.encode:
        app.text_encoder = None  # the reference's embeddings go in; no Llama on rank 0
    t = time.perf_counter()
    with track_fallbacks() as load_fallbacks:
        app.load()
    load_s = time.perf_counter() - t
    say(f"{layout} {args.exec_mode}: layers={app.config.num_layers}+"
        f"{app.config.num_single_layers} {ref['num_frames']}x{ref['height']}x{ref['width']} "
        f"steps={ref['steps']}; load {load_s:.1f} s phases="
        f"{ {k: round(v, 1) for k, v in app.phase_seconds.items()} }", flush=True)

    checks: list[tuple[str, bool, str]] = []
    embeds = ref["embeds"]
    text = {"encoder_hidden_states": embeds["prompt_embeds"],
            "encoder_attention_mask": embeds["prompt_attention_mask"],
            "pooled_projections": embeds["pooled_prompt_embeds"]}
    if args.encode:
        ours = dict(zip(text, app.encode_prompt(ref["prompt"])))
        for name, value in ours.items():
            if value.is_floating_point():
                m = parity.metrics(value.float(), text[name].float())
                detail = f"rel_l1={m['rel_l1']:.3e} cos={m['cosine']:.6f}"
                ok = m["cosine"] >= parity.MIN_COSINE
            else:
                ok = torch.equal(value.cpu(), text[name])
                detail = f"equal={ok} valid={int(value.sum())}"
            checks.append((f"encode_{name}", ok, detail))
        text = {}  # the prompt goes through our encoder in generate()

    stamps: list[float] = []
    app.step_hook = lambda: stamps.append(time.perf_counter())
    with track_fallbacks() as fallbacks:
        t = time.perf_counter()
        kwargs = {"prompt": ref["prompt"]} if args.encode else text
        out = app.generate(**kwargs, seed=int(ref["seed"]),
                           num_inference_steps=int(ref["steps"]),
                           guidance_scale=float(ref["guidance_scale"]), output_type="latent")
        generate_s = time.perf_counter() - t
    latents = out.latents.float().cpu()
    step_s = [b - a for a, b in zip([t, *stamps], stamps)]

    r, c = ref["ref_fp32"], ref["control_bf16"]
    m = parity.metrics(latents, r, control=c)
    m["vs_bf16"] = parity.metrics(latents, c)
    control = parity.metrics(c, r)
    notes: list[str] = []
    ok, reasons = parity.check(m, control=control, notes=notes)
    ratio = m["rel_l1"] / control["rel_l1"] if control["rel_l1"] > 0 else float("inf")
    checks.append(("finite", bool(torch.isfinite(latents).all()),
                   f"absmean={float(latents.abs().mean()):.4f} shape={tuple(latents.shape)}"))
    checks.append(("parity_latents", ok,
                   f"rel_l1={m['rel_l1']:.3e} ({ratio:.3f}x control {control['rel_l1']:.3e}, "
                   f"gate {parity.MAX_RATIO_VS_CONTROL}x) cos={m['cosine']:.6f} "
                   f"err_cos={m['err_cos']:.3f} {parity.same_dtype_label(m, control)}"
                   + (f" {reasons}" if reasons else "") + (f" {notes}" if notes else "")))
    checks.append(("fallbacks", not (load_fallbacks or fallbacks),
                   f"load={list(load_fallbacks)} generate={list(fallbacks)}"))
    evidence = parity.device_evidence(app.transformer, fallbacks)
    checks.append(("unwarmed_shapes", evidence["unwarmed_shapes"] == [],
                   f"{evidence['unwarmed_shapes']}"))
    checks.append(("steps", len(stamps) == int(ref["steps"]),
                   f"{len(stamps)} denoise iterations"))
    passed = all(p for _, p, _ in checks)
    result = {
        "rank": rank, "world_size": world, "layout": layout, "exec_mode": args.exec_mode,
        "encode": bool(args.encode), "ok": passed, "reference": str(args.check),
        "checks": [{"name": n, "ok": p, "detail": d} for n, p, d in checks],
        "parity": {"vs_fp32": m, "control": control, "ratio_vs_control": ratio,
                   "reasons": reasons, "notes": notes},
        "device_evidence": evidence, "load_seconds": load_s, "generate_seconds": generate_s,
        "step_seconds": step_s, "total_seconds": time.perf_counter() - t_start,
    }
    if args.out_json is not None:
        path = Path(args.out_json).expanduser()
        if rank:
            path = path.with_name(f"{path.stem}.rank{rank}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str))
    for name, p, detail in checks:
        say(f"{'PASS' if p else 'FAIL'} {name:24s} {detail}", flush=True)
    peak = evidence["peak_device_mem_gb"]
    print(f"[rank {rank}] {'PASS' if passed else 'FAIL'} end to end {layout} {args.exec_mode}, "
          f"{ref['steps']} steps: generate {generate_s:.1f} s, steps "
          f"{[round(s, 2) for s in step_s]} s"
          + (f", peak_hbm={peak:.2f}GiB" if peak is not None else ""), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if passed else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    summary, details = __doc__.split("\n\n", 1)
    p = argparse.ArgumentParser(description=" ".join(summary.split()), epilog=details,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="the diffusers snapshot root")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-ref", metavar="PATH", help="phase 1: write the host reference")
    mode.add_argument("--check", metavar="PATH", help="phase 2: check the device against it")
    p.add_argument("--exec-mode", choices=("compile", "eager"), default="compile")
    p.add_argument("--tp-degree", type=int, default=2)
    p.add_argument("--cp-degree", type=int, default=None,
                   help="context parallelism over the image tokens (default: world / tp)")
    p.add_argument("--cp-mode", choices=("gather_kv", "ulysses", "ring"), default="gather_kv")
    p.add_argument("--sp", action="store_true", help="sequence parallelism within the TP group")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--frames", type=int, default=FRAMES, help="4k+1 (5 or 9 for the gate)")
    p.add_argument("--height", type=int, default=H)
    p.add_argument("--width", type=int, default=W)
    p.add_argument("--layers", type=int, default=None,
                   help="dual-stream blocks (default: the checkpoint's 20)")
    p.add_argument("--single-layers", type=int, default=None,
                   help="single-stream blocks (default: the checkpoint's 40)")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--encode", action="store_true",
                   help="phase 2: our host encoder encodes the prompt too (Llama on rank 0)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                   help="host threads for the reference")
    p.add_argument("--out-json", default=None, help="phase 2: the result (rank 0; others .rankN)")
    p.add_argument("--dry-run", action="store_true",
                   help="phase 2 outside torchrun: print the torchrun command, do not run it")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.make_ref:
        return make_ref(args)
    if "WORLD_SIZE" not in os.environ:
        cp = args.cp_degree if args.cp_degree is not None else 1
        return launch(int(args.tp_degree) * int(cp), argv, dry_run=args.dry_run,
                      script=__file__)
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
