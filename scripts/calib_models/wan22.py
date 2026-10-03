"""Static FP8 activation-scale calibration for Wan 2.2 A14B (two experts), CPU.

Plugin for ``scripts/ptq_calibrate_activations.py --model-type wan22``: the driver
loads this file and calls ``run(args, install_hooks)``. Wan 2.2 T2V-A14B has two
DiTs with the Wan 2.1 architecture (40 blocks, dim 5120):

* ``<snapshot>/transformer``   -- high-noise expert, runs while t >= boundary
* ``<snapshot>/transformer_2`` -- low-noise expert, runs while t <  boundary

with ``boundary = boundary_ratio (model_index.json, 0.875) * 1000 = 875``. Both are
loaded in bf16 and driven by the real ``WanOrchestrator`` loop (its UniPC scheduler,
``_select_transformer`` / ``_select_guidance_scale``), so each expert only sees the
steps it serves in production (at 20 steps the high-noise expert serves the first
few steps, the low-noise one the rest).

Naming convention of the output JSON (``layers``)
-------------------------------------------------
The driver keys records by dotted module name and both experts have identical
module names, so:

* high-noise expert (``transformer/``):    plain names, e.g. ``blocks.0.attn1.to_q``
* low-noise expert  (``transformer_2/``):  ``transformer_2.``-prefixed names, e.g.
  ``transformer_2.blocks.0.attn1.to_q``

(the low-noise expert is hooked through a throwaway ``nn.Module`` whose child is
named ``transformer_2``; ``QuantSpec.matches`` accepts the prefixed names because
suffix targets match ``*.<target>``). NOTE: ``difflet/quant/checkpoint.py::
calibrated_amax`` currently looks names up WITHOUT a subfolder prefix, so as is
both experts' checkpoints would get the high-noise expert's scales; the
transformer_2 checkpoint build needs a subfolder-aware lookup
(``"transformer_2." + name``) before this JSON is used for static scales.

``--max-steps``: the driver's ``install_hooks`` puts a stop hook on the first
target linear of every model it is given, all sharing one call counter. Both
experts are hooked, so the counter counts DiT forwards across the expert switch
(= denoise steps at guidance 1, 2 per step under CFG) and ``--max-steps N`` stops
after N forwards whichever expert runs them. With N <= the number of high-noise
steps the low-noise expert records nothing (a warning is printed).

Driver argument ``--model-type wan22`` needs ``QuantSpec.for_model("wan22")``
to resolve to the Wan target set (``difflet.quant.targets.TARGETS_BY_MODEL``).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

EXPERTS = ("transformer", "transformer_2")  # high-noise, low-noise


def _shard_files(expert_dir: Path) -> list[Path]:
    for index_name in ("diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json"):
        index = expert_dir / index_name
        if index.exists():
            names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            return [expert_dir / n for n in names]
    for single in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        if (expert_dir / single).exists():
            return [expert_dir / single]
    raise FileNotFoundError(f"no safetensors checkpoint in {expert_dir}")


def _load_expert(expert_dir: Path, dtype: torch.dtype = torch.bfloat16):
    """One Wan expert in ``dtype``, built on meta and filled shard by shard.

    Each fp32 shard is renamed and cast to ``dtype`` before the next one is read,
    so the peak is the ``dtype`` model plus one fp32 shard (~5 GB), never a full
    fp32 copy. The only non-checkpoint tensors (the RoPE tables) are recomputed.
    """
    from safetensors.torch import load_file

    import difflet.models.wan.modeling_wan as wan
    from difflet.models.wan.checkpoint.backbone import convert_backbone_state_dict

    config = wan.WanTransformerConfig.from_diffusers_dict(json.loads((expert_dir / "config.json").read_text()))
    with torch.device("meta"):
        model = wan.WanTransformer3DModel(config, dtype=dtype)
    model = model.to(dtype)
    for shard in _shard_files(expert_dir):
        state = convert_backbone_state_dict(load_file(str(shard)))
        state = {k: v.to(dtype) for k, v in state.items()}
        model.load_state_dict(state, strict=False, assign=True)
        del state
    rope = model.rope
    fresh = wan.WanRotaryPosEmbed(attention_head_dim=config.attention_head_dim, patch_size=config.patch_size,
                                  max_seq_len=config.rope_max_seq_len, theta=config.rope_theta)
    for name, buf in fresh.named_buffers():
        # same dtype as the driver's Wan 2.1 path (model.to(dtype) casts these too)
        setattr(rope, name, buf.to(dtype))
    missing = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if missing:
        raise RuntimeError(f"{expert_dir}: {len(missing)} tensors not in the checkpoint, e.g. {missing[:5]}")
    return model.eval()


def _hook_expert(install_hooks, model, prefix: str | None) -> int:
    """Hook ``model``; with ``prefix`` its records are keyed ``<prefix>.<name>``."""
    if prefix is None:
        return install_hooks(model)
    holder = torch.nn.Module()
    holder.add_module(prefix, model)  # hooks live on model's modules; holder is discarded
    return install_hooks(holder)


def _embeds(args, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(prompt_embeds, negative_prompt_embeds or None when CFG is off)."""
    import ptq_calibrate_activations as driver

    cfg = float(args.guidance_scale) > 1.0
    text_pt = getattr(args, "text_pt", None)
    if text_pt:
        data = torch.load(text_pt, map_location="cpu")
        if isinstance(data, torch.Tensor):
            data = {"prompt_embeds": data}
        pos = data["prompt_embeds"].to(dtype)
        neg = data.get("negative_prompt_embeds")
        if cfg and neg is None:
            neg = driver._prompt_embeds(args.model_dir, "", args.text_seq_len, dtype)
        return pos, (neg.to(dtype) if cfg else None)
    pos = driver._prompt_embeds(args.model_dir, args.prompt, args.text_seq_len, dtype)
    # Under CFG: the empty-prompt UMT5 encode, as WanOrchestrator._negative_prompt_embeds does.
    neg = driver._prompt_embeds(args.model_dir, "", args.text_seq_len, dtype) if cfg else None
    return pos, neg


def run(args, install_hooks) -> float:
    from difflet.models.wan.pipeline import WanOrchestrator

    dtype = torch.bfloat16
    model_dir = Path(args.model_dir)
    # Text first: the UMT5 encoder is freed before ~2x28 GB of experts arrive.
    started = time.perf_counter()
    embeds, neg = _embeds(args, dtype)
    print(f"[calib/wan22] prompt embeds {tuple(embeds.shape)} (cfg={'on' if neg is not None else 'off'}) "
          f"in {time.perf_counter() - started:.1f}s", flush=True)

    experts = {}
    for sub in EXPERTS:
        started = time.perf_counter()
        experts[sub] = _load_expert(model_dir / sub, dtype)
        print(f"[calib/wan22] {sub} loaded in {time.perf_counter() - started:.1f}s", flush=True)
    high, low = experts["transformer"], experts["transformer_2"]

    n_high = _hook_expert(install_hooks, high, None)
    n_low = _hook_expert(install_hooks, low, "transformer_2")
    print(f"[calib/wan22] hooked high-noise={n_high} (plain names), low-noise={n_low} (transformer_2.*)", flush=True)

    orch = WanOrchestrator(model_path=str(model_dir), transformer=high, transformer_2=low, dtype=dtype,
                           height=args.height, width=args.width, num_frames=args.num_frames,
                           max_text_length=args.text_seq_len)
    if orch.boundary_ratio is None:
        raise SystemExit(f"{model_dir}/model_index.json has no boundary_ratio; not a two-expert Wan 2.2 snapshot")
    if orch.scheduler is not None:
        orch.scheduler.set_timesteps(args.steps)
        boundary = orch.boundary_ratio * float(getattr(orch.scheduler.config, "num_train_timesteps", 1000))
        n_high_steps = sum(float(t) >= boundary for t in orch.scheduler.timesteps)
        print(f"[calib/wan22] boundary t={boundary:g}: high-noise expert serves {n_high_steps}/{args.steps} steps",
              flush=True)
        if args.max_steps is not None and args.max_steps <= n_high_steps:
            print("[calib/wan22] WARNING: --max-steps stops inside the high-noise range; "
                  "the low-noise expert will record nothing", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    latent_frames = (args.num_frames - 1) // 4 + 1
    latents = torch.randn(1, 16, latent_frames, args.height // 8, args.width // 8, generator=g)
    started = time.perf_counter()
    with torch.no_grad():
        orch(prompt_embeds=embeds, negative_prompt_embeds=neg, latents=latents, height=args.height,
             width=args.width, num_frames=args.num_frames, num_inference_steps=args.steps,
             guidance_scale=args.guidance_scale, output_type="latent")
    return time.perf_counter() - started
