"""Frozen CPU fp32 quality references (design §7, quality).

``difflet bench reference --from <result.json>`` reruns the result's recipe in diffusers on the host
CPU in float32, starting from the exact initial latents the device run dumped (a fixed seed does not
make two implementations draw the same noise), and freezes the final latents and the pre-encode
frames under ``$DIFFLET_BENCH_ROOT/references/<id>/`` with a manifest of every file's sha256 and
size. The id hashes everything that decides the output: model and revision, shape, steps, prompt,
guidance, the initial latents' hashes and the reference implementation's version. A reference is
never rewritten; hosts share one by copying the directory, and the manifest proves it is the same.

Supported (the settings ``scripts/cpu_fp32_reference.py`` validated): Qwen-Image with or without
true CFG (diffusers' ``true_cfg_scale`` and negative prompt), and LTX-2 at guidance 1.0. Anything else is refused with the reason, not approximated.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from difflet.bench import host as H

#: Bump when the reference computation changes; part of every reference id.
IMPLEMENTATION = "diffusers-cpu-fp32/1"
MODELS = {"Qwen/Qwen-Image": "qwen_image", "Lightricks/LTX-2": "ltx_2"}


class ReferenceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spec_from_args(args: dict[str, Any]) -> dict[str, Any]:
    """The sampling settings a reference must reproduce, from a recipe's args."""
    model = MODELS.get(str(args.get("model-id")))
    if model is None:
        raise ReferenceError(f"no CPU fp32 reference implementation for {args.get('model-id')!r}; "
                             f"supported: {sorted(MODELS)}")
    spec = {"model": model, "model_id": args["model-id"], "revision": args["revision"],
            "prompt": args["prompt"], "steps": int(args.get("steps") or 50),
            "height": args.get("height"), "width": args.get("width"),
            "num_frames": args.get("num-frames"), "negative_prompt": args.get("negative-prompt")}
    if model == "qwen_image":
        true_cfg = args.get("true-cfg-scale")
        if true_cfg not in (None, False) and float(true_cfg) > 1.0:
            # Only under true CFG, so the unguided references keep their ids.
            spec["true_cfg_scale"] = float(true_cfg)
            if spec["negative_prompt"] is None:
                spec["negative_prompt"] = " "  # the engine's default (qwen_image._true_cfg_kwargs)
    if model == "ltx_2" and float(args.get("guidance-scale") or 0) != 1.0:
        raise ReferenceError("the LTX-2 reference runs at guidance 1.0; pin guidance-scale: 1.0 "
                             "in the recipe (the engine's own default may differ)")
    for key in ("height", "width"):
        if spec[key] is None:
            raise ReferenceError(f"the recipe must pin {key} for a reference")
    if model == "ltx_2" and spec["num_frames"] is None:
        raise ReferenceError("the recipe must pin num-frames for an LTX-2 reference")
    return spec


def initial_hashes(dump_dir: Path) -> dict[str, str]:
    return {p.stem: _sha256(p) for p in sorted(Path(dump_dir).glob("initial*.pt"))}


def reference_id(spec: dict[str, Any], initial: dict[str, str]) -> str:
    payload = json.dumps({"spec": spec, "initial": initial, "impl": IMPLEMENTATION},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def references_root() -> Path:
    return H.bench_root() / "references"


def verify(directory: Path) -> dict[str, Any]:
    """A reference's manifest, after checking every file against it."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise ReferenceError(f"{directory} has no manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for name, info in manifest["files"].items():
        path = directory / name
        if not path.exists() or path.stat().st_size != info["bytes"] or _sha256(path) != info["sha256"]:
            raise ReferenceError(f"{path} does not match the reference manifest")
    return manifest


def find_reference(plan, dump_dir: Path | None = None) -> Path | None:
    """The frozen reference for ``plan``'s recipe and the initial latents in ``dump_dir``."""
    if dump_dir is None:
        return None
    try:
        spec = spec_from_args(plan.recipe.args)
    except ReferenceError:
        return None
    ref_id = reference_id(spec, initial_hashes(dump_dir))
    directory = references_root() / f"{spec['model']}-{ref_id[:16]}"
    if not directory.exists():
        return None
    verify(directory)
    return directory


def build(result: dict[str, Any], *, cpus: list[int]) -> Path:
    """Build and freeze the reference for a result's quality dump. CPU-only, pinned to ``cpus``."""
    args = (result.get("recipe") or {}).get("args") or {}
    spec = spec_from_args(args)
    dump = (((result.get("metrics") or {}).get("quality") or {}).get("dump") or {}).get("dir")
    if not dump or not Path(dump).is_dir():
        raise ReferenceError("the result has no quality dump on this host; run "
                             "`difflet bench run --metrics quality --quality-ref none` first")
    initial = initial_hashes(Path(dump))
    if not initial:
        raise ReferenceError(f"no initial*.pt latents in {dump}")
    ref_id = reference_id(spec, initial)
    final = references_root() / f"{spec['model']}-{ref_id[:16]}"
    if final.exists():
        verify(final)
        return final  # frozen: never rebuilt
    idle = H.check_idle([], cpus)
    if not idle.idle:
        raise ReferenceError("CPUs not idle: " + "; ".join(idle.reasons()))
    os.sched_setaffinity(0, cpus)
    model_dir = H.snapshot_dir(str(spec["model_id"]), str(spec["revision"]))
    staging = references_root() / f".{final.name}.{os.getpid()}.partial"
    staging.mkdir(parents=True, exist_ok=False)
    started = time.time()
    if spec["model"] == "qwen_image":
        step_seconds = _run_qwen(spec, model_dir, Path(dump), staging)
    else:
        step_seconds = _run_ltx2(spec, model_dir, Path(dump), staging)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)}
             for p in sorted(staging.iterdir()) if p.is_file()}
    from difflet.bench import provenance

    packages = provenance.python_packages()
    manifest = {"schema_version": 1, "reference_id": ref_id, "implementation": IMPLEMENTATION,
                "spec": spec, "initial_latents_sha256": initial, "files": files,
                "built_from_run": result.get("run_id"), "seconds": time.time() - started,
                "step_seconds": step_seconds, "cpus": H.format_cpu_list(cpus),
                "toolchain": provenance.toolchain(packages),
                "git": provenance.git_state(H.source_root())}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    os.replace(staging, final)
    return final


class _Recorder:
    def __init__(self, num_steps: int) -> None:
        self.num_steps = num_steps
        self.step_times: list[float] = []
        self.final = None
        self._last = time.perf_counter()

    def __call__(self, pipe, step, timestep, callback_kwargs):
        now = time.perf_counter()
        self.step_times.append(now - self._last)
        self._last = now
        if step == self.num_steps - 1:
            self.final = callback_kwargs["latents"].detach().cpu().clone()
        return callback_kwargs


def _fixed(tensor):
    import torch

    fixed = tensor.to(torch.float32)

    def prepare(*_args, **_kwargs):
        return fixed.clone()

    return prepare


def _run_qwen(spec: dict[str, Any], model_dir: Path, initial_dir: Path, out: Path) -> list[float]:
    import torch
    from diffusers import QwenImagePipeline

    from difflet.utils.latent_dump import write_frames

    encoder = QwenImagePipeline.from_pretrained(str(model_dir), transformer=None, vae=None,
                                                torch_dtype=torch.float32)
    embeds, mask = encoder.encode_prompt(prompt=spec["prompt"], device="cpu")
    scale = spec.get("true_cfg_scale", 1.0)
    neg_embeds, neg_mask = (encoder.encode_prompt(prompt=spec["negative_prompt"], device="cpu")
                            if scale > 1.0 else (None, None))
    del encoder
    gc.collect()
    pipe = QwenImagePipeline.from_pretrained(str(model_dir), text_encoder=None, tokenizer=None,
                                             torch_dtype=torch.float32)
    pipe.prepare_latents = _fixed(torch.load(initial_dir / "initial.pt", map_location="cpu"))
    recorder = _Recorder(spec["steps"])
    image = pipe(prompt_embeds=embeds, prompt_embeds_mask=mask, negative_prompt_embeds=neg_embeds,
                 negative_prompt_embeds_mask=neg_mask, true_cfg_scale=scale,
                 height=int(spec["height"]), width=int(spec["width"]),
                 num_inference_steps=spec["steps"], callback_on_step_end=recorder,
                 output_type="pt").images[0]
    torch.save(recorder.final, out / "final.pt")
    # The device writer quantizes the decoded image with floor(x*255+0.5) (qwen _save_image_pil).
    write_frames(out, image.clamp(0, 1), rule="round_half_up", layout="CHW")
    return recorder.step_times


def _run_ltx2(spec: dict[str, Any], model_dir: Path, initial_dir: Path, out: Path) -> list[float]:
    import torch
    from diffusers import LTX2Pipeline

    from difflet.utils.latent_dump import write_frames

    encoder = LTX2Pipeline.from_pretrained(str(model_dir), transformer=None, vae=None,
                                           audio_vae=None, vocoder=None, connectors=None,
                                           torch_dtype=torch.float32)
    embeds, mask, _, _ = encoder.encode_prompt(prompt=spec["prompt"],
                                               do_classifier_free_guidance=False, device="cpu")
    del encoder
    gc.collect()
    pipe = LTX2Pipeline.from_pretrained(str(model_dir), text_encoder=None, tokenizer=None,
                                        torch_dtype=torch.float32)
    pipe.prepare_latents = _fixed(torch.load(initial_dir / "initial_video.pt", map_location="cpu"))
    pipe.prepare_audio_latents = _fixed(torch.load(initial_dir / "initial_audio.pt",
                                                   map_location="cpu"))
    recorder = _Recorder(spec["steps"])
    output = pipe(prompt_embeds=embeds, prompt_attention_mask=mask,
                  height=int(spec["height"]), width=int(spec["width"]),
                  num_frames=int(spec["num_frames"]), frame_rate=24.0,
                  num_inference_steps=spec["steps"], guidance_scale=1.0,
                  callback_on_step_end=recorder)
    torch.save(recorder.final, out / "final_video.pt")
    # The device writer hands export_to_video float [F,H,W,C] in [0,1] (it truncates).
    write_frames(out, output.frames[0], rule="truncate", layout="FHWC")
    return recorder.step_times
