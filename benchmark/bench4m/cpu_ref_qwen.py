#!/usr/bin/env python3
"""Qwen-Image CPU fp32 reference from a device run's exact initial latents.

Adapted from main's ``scripts/cpu_fp32_reference.py`` (540e5f0; copy kept as
``cpu_fp32_reference.main-540e5f0.py``), the ``qwen_image`` branch. Unchanged from main:
diffusers' own ``QwenImagePipeline`` in float32 on the CPU; the prompt is encoded with only
the text encoder loaded, which is freed before the transformer and VAE load; 1024x1024;
``true_cfg_scale=1.0`` and no negative prompt.

Changes (measurement tooling, plan D9):
* ``--initial`` names the device dump file (``initial.pt``, packed (1,4096,64) bf16). It is
  passed through diffusers' own ``latents=`` argument (``prepare_latents`` then only casts
  it to fp32, an exact upcast), and a forward pre-hook captures the transformer's FIRST
  ``hidden_states`` to prove that what enters the loop is bit-identical to the dump.
* ``--steps`` defaults to 50 (main defaulted to 20).
* Lossless output: ``out.pt`` (the decoded image, fp32 (1,3,H,W) in [0,1], diffusers'
  ``output_type="pt"``) and ``out.png`` written from it with the device CLIs' uint8 rule
  (``quality_metrics.to_uint8``), instead of diffusers' PIL rounding.
* ``final.pt`` is the packed fp32 loop output, the same point as the device's ``final.pt``
  (captured both by the step callback and at ``_unpack_latents``; the two must agree).
* ``reference.json``: versions, threads, wall time per phase, per-step and per-forward
  times, peak RSS, the injection check, the schedule, the hashes of inputs and outputs.
* ``--cut-layers N`` (smoke tests only): the transformer is built with its first N blocks.

``--dtype bfloat16`` (added 2026-10-28, the CPU bf16 baseline; default float32 runs the fp32
code path unchanged): the text encoder and the transformer are loaded with
``torch_dtype=bfloat16`` (diffusers' own bf16 load), the VAE in ``--vae-dtype`` (default
float32). Everything else is the fp32 reference's: same dump, prompt, schedule, guidance.
diffusers then keeps the loop latents in the prompt-embeds dtype (``prepare_latents(...,
prompt_embeds.dtype)``), so the bf16 dump enters the loop as is, the timestep goes to the DiT
in bf16 and the scheduler (fp32 inside its step) casts back to bf16 after every step; the VAE
input is ``latents.to(vae.dtype)``. The injection check then records ``prepare_latents``'s
output and the first DiT input (``cpu_ref_common.injection_check_cast``).

Guidance, mirrored from the device runs (both passed ``--guidance-scale 4.0`` and no
``--true-cfg-scale``): Qwen-Image's transformer has ``guidance_embeds=false``, so the 4.0 is
dropped by the DiT on both backends, and true CFG was off on both (native runs it only with
``--true-cfg-scale > 1``; AoT on main has no negative branch). Here ``guidance_scale=4.0`` is
passed (diffusers ignores it, same as the device DiT) and ``true_cfg_scale=1.0`` with no
negative prompt, so ``do_true_cfg`` is False: one conditional DiT forward per step.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_only  # noqa: E402  (first: sets the no-Neuron env before torch is imported)

import torch  # noqa: E402

import cpu_ref_common as C  # noqa: E402
from quality_metrics import to_uint8  # noqa: E402

HEIGHT = WIDTH = 1024
PACKED_SHAPE = (1, (HEIGHT // 16) * (WIDTH // 16), 64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-path", required=True, help="local Qwen/Qwen-Image snapshot")
    p.add_argument("--initial", type=Path, required=True, help="device dump initial.pt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--cut-layers", type=int, default=None, help="smoke only: first N DiT blocks")
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (default: torch's)")
    C.add_dtype_args(p)
    a = p.parse_args()
    dt, vdt = C.DTYPES[a.dtype], C.DTYPES[a.vae_dtype]
    reduced = a.dtype != "float32" or a.vae_dtype != "float32"
    if a.steps < 2:
        # FlowMatchEuler with shift_terminal divides by zero on a 1-sigma schedule (seen on LTX-2).
        p.error("--steps must be >= 2")

    torch.set_grad_enabled(False)
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rss = C.RssMonitor()
    t_start = time.perf_counter()
    rec: dict = {"model": "qwen_image", "steps": a.steps, "height": HEIGHT, "width": WIDTH,
                 "prompt": C.PROMPT, "cut_layers": a.cut_layers,
                 "env": C.environment(Path(__file__).resolve(), a),
                 "compute_dtype": {"text_encoder": a.dtype, "transformer": a.dtype, "vae": a.vae_dtype}}
    print(f"[cpu-ref] env {rec['env']}", flush=True)
    times: dict = {}

    initial, rec["initial"] = C.load_initial(a.initial, shape=PACKED_SHAPE)
    rss.mark("initial_loaded")

    cpu_only.install()
    from diffusers import QwenImagePipeline, QwenImageTransformer2DModel

    t = time.perf_counter()
    encoder = QwenImagePipeline.from_pretrained(
        a.model_path, transformer=None, vae=None, torch_dtype=dt
    )
    times["text_encoder_load_s"] = time.perf_counter() - t
    rss.mark("text_encoder_loaded")
    t = time.perf_counter()
    prompt_embeds, prompt_embeds_mask = encoder.encode_prompt(prompt=C.PROMPT, device="cpu")
    times["encode_s"] = time.perf_counter() - t
    rec["prompt_embeds"] = {"dtype": str(prompt_embeds.dtype), "shape": list(prompt_embeds.shape),
                            "mask": None if prompt_embeds_mask is None else
                            {"shape": list(prompt_embeds_mask.shape), "sum": int(prompt_embeds_mask.sum())}}
    rss.mark("encoded")
    del encoder
    C.free()
    rss.mark("text_encoder_freed")

    t = time.perf_counter()
    kwargs = {}
    if a.cut_layers:
        kwargs["transformer"] = QwenImageTransformer2DModel.from_pretrained(
            a.model_path, subfolder="transformer", num_layers=a.cut_layers, torch_dtype=dt
        )
    pipe = QwenImagePipeline.from_pretrained(
        a.model_path, text_encoder=None, tokenizer=None,
        torch_dtype={"transformer": dt, "vae": vdt, "default": torch.float32} if reduced else torch.float32,
        **kwargs
    )
    times["dit_vae_load_s"] = time.perf_counter() - t
    tr = pipe.transformer
    rec["transformer"] = {"dtype": str(next(tr.parameters()).dtype), "num_layers": len(tr.transformer_blocks),
                          "guidance_embeds": bool(tr.config.guidance_embeds),
                          "params_b": round(sum(x.numel() for x in tr.parameters()) / 1e9, 3)}
    rec["vae_dtype"] = str(next(pipe.vae.parameters()).dtype)
    if reduced:
        rec["param_dtypes"] = {"transformer": C.param_dtypes(tr), "vae": C.param_dtypes(pipe.vae)}
    if rec["transformer"]["dtype"] != str(dt) or rec["vae_dtype"] != str(vdt):
        raise RuntimeError(f"not {a.dtype}/{a.vae_dtype}: {rec['transformer']} vae {rec['vae_dtype']}")
    rss.mark("dit_vae_loaded")

    cap = C.FirstCallCapture(tr, ("hidden_states", "timestep") if reduced else ("hidden_states",))
    prep = C.PrepareSpy(pipe) if reduced else None
    finals: dict = {}
    orig_unpack = pipe._unpack_latents

    def unpack_spy(latents, *args, **kw):
        finals["at_unpack"] = latents.detach().clone()
        return orig_unpack(latents, *args, **kw)

    pipe._unpack_latents = unpack_spy  # staticmethod on the class; instance attr shadows it
    timer = C.StepTimer(a.steps)
    if reduced:
        timer.on_first = lambda: C.early_injection(a.out_dir, lambda: {
            "latents": C.injection_check_cast(cap.captured["hidden_states"], initial, prep.value)})
    timer.start()
    t = time.perf_counter()
    image = pipe(
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        latents=initial.to(torch.float32),
        true_cfg_scale=1.0,
        guidance_scale=a.guidance_scale,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=a.steps,
        output_type="pt",
        callback_on_step_end=timer,
    ).images
    times["pipeline_call_s"] = time.perf_counter() - t
    times["denoise_s"] = sum(timer.step_s)
    times["decode_and_postprocess_s"] = times["pipeline_call_s"] - times["denoise_s"]
    cap.remove()
    rss.mark("pipeline_done")

    sched = pipe.scheduler
    rec["schedule"] = {"timesteps": [float(x) for x in sched.timesteps],
                       "sigmas": [float(x) for x in sched.sigmas]}
    rec["dit_calls"] = cap.calls
    rec["cfg"] = {"true_cfg": False, "dit_forwards_per_step": cap.calls / a.steps,
                  "guidance_scale_passed": a.guidance_scale, "guidance_used_by_dit": rec["transformer"]["guidance_embeds"]}
    if cap.calls != a.steps:
        raise RuntimeError(f"{cap.calls} DiT calls for {a.steps} steps: CFG on?")
    if reduced:
        rec["injection"] = C.injection_check_cast(cap.captured["hidden_states"], initial, prep.value)
        rec["first_dit_timestep"] = {"value": cap.captured["timestep"].tolist(),
                                     "dtype": str(cap.captured["timestep"].dtype)}
        inj_ok = rec["injection"]["ok"]
    else:
        rec["injection"] = C.injection_check(cap.captured["hidden_states"], initial)
        inj_ok = rec["injection"]["bit_identical_to_fp32_upcast"]

    final = timer.final
    if final is None or not torch.equal(final, finals["at_unpack"]):
        raise RuntimeError("callback final latents != latents at _unpack_latents")
    torch.save(final, a.out_dir / "final.pt")
    if image.ndim != 4 or image.shape[0] != 1:
        raise RuntimeError(f"unexpected image tensor {tuple(image.shape)}")
    torch.save(image.contiguous(), a.out_dir / "out.pt")
    from PIL import Image

    Image.fromarray(to_uint8(image[0]).permute(1, 2, 0).contiguous().numpy()).save(a.out_dir / "out.png")
    rec["outputs"] = {
        "final.pt": {**C.tensor_rec(final), "file_sha256": C.file_sha(a.out_dir / "final.pt")},
        "out.pt": {"dtype": str(image.dtype), "shape": list(image.shape), "min": float(image.min()),
                   "max": float(image.max()), "file_sha256": C.file_sha(a.out_dir / "out.pt")},
        "out.png": {"file_sha256": C.file_sha(a.out_dir / "out.png"), "uint8_rule": "quality_metrics.to_uint8"},
    }
    rec["timing"] = {**times, **C.step_summary(timer.step_s, cap.forward_s, 50)}
    rec["wall_s"] = time.perf_counter() - t_start
    rec["memory"] = rss.stop()
    rec["finished_utc"] = C.utc()
    rec["cpu_only"] = cpu_only.report()
    C.write_json(a.out_dir / "reference.json", rec)
    print(f"[cpu-ref] DONE wall {rec['wall_s']:.1f} s peak RSS {rec['memory']['peak_rss_gib_sampled']} GiB "
          f"injection bit-identical {inj_ok} dtype {a.dtype} vae {a.vae_dtype} "
          f"cpu_only {rec['cpu_only']}", flush=True)
    if not inj_ok:
        sys.exit(3)
    if not rec["cpu_only"]["ok"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
