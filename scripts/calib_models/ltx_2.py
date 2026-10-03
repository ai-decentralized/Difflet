"""LTX-2 calibration loop for ``scripts/ptq_calibrate_activations.py`` (``--model-type ltx_2``).

Records the per-call input absmax of every FP8 target linear of the LTX-2 dual-stream DiT
(``difflet.quant.targets.LTX_2_TARGETS``: 6 attentions x {to_q,to_k,to_v,to_out.0} + video/audio
FFN in/out = 28 linears per block, 48 blocks = 1344) while running the real denoise loop on the
CPU, mirroring the device path the FP8 checkpoint serves (``--transformer-mode single``):

* conditioning: Gemma3 text encoder + connectors through the diffusers ``LTX2Pipeline``
  (``_load_ltx_2_host_pipeline`` without decoders) and ``LTX2Orchestrator.prepare_conditioning``
  — exactly the CLI's host prompt path (left padding, ``max_sequence_length=text_seq_len``,
  ``[negative "", positive]`` stacked when guidance > 1);
* transformer: diffusers ``LTX2VideoTransformer3DModel.from_pretrained(<snap>/transformer)`` in
  bf16 with Difflet's split-RoPE patch (``segmented._patch_ltx2_rope``), i.e.
  ``_load_cpu_transformer``. Module names are the HF checkpoint keys
  (``transformer_blocks.3.attn1.to_q``) — no ``transformer.`` prefix — which is what
  ``difflet.quant.checkpoint.calibrated_amax`` looks up;
* loop: ``LTX2Orchestrator`` (its scheduler, latent init ``randn(generator=manual_seed(seed))``
  in bf16, video then audio, as ``generate`` does) driving the model through
  ``_DiffusersAdapter``, which makes the same call the single-mode Neuron wrapper makes
  (``audio_timestep=timestep``, ``audio_sigma=sigma``, fps 24, no STG / modality isolation).
  Guidance > 1 runs the uncond and cond halves as two batch-1 passes per step (the non
  cfg-parallel device behaviour), so every layer then records 2 values per step.

The device default text length is 1024 (``LTX_2_DEFAULT_TEXT_SEQ_LEN``); the driver's
``--text-seq-len`` defaults to 512 (Wan), so pass ``--text-seq-len 1024`` to mirror the device.

``--text-pt`` (optional) skips Gemma. Accepted formats:

* the ``.safetensors`` bundle written by ``scripts/ltx_2_cache_dit_inputs.py`` (its latents /
  coords / timesteps are ignored — the loop draws its own latents from ``--seed``, as the device
  does); it is batch 1, i.e. guidance 1 only;
* a ``torch.save`` dict (``.pt``) with ``encoder_hidden_states`` [B, L, 3840],
  ``audio_encoder_hidden_states`` [B, L, 3840], ``encoder_attention_mask`` [B, L] (bool/int) and
  optionally ``audio_encoder_attention_mask`` (defaults to the video mask), with B = 1 at
  guidance 1 and B = 2 ``[uncond, cond]`` at guidance > 1, and L = ``--text-seq-len``.

When the conditioning is computed here it is saved as ``<out stem>_conditioning.pt`` (the
``.pt`` format above) so a rerun can pass it as ``--text-pt``.

Memory (124 GB host): Gemma3-12B (46 GB fp32 on disk) loads as ~24 GB bf16 and is freed
before the transformer (37.8 GB bf16) loads, so the peak is ~40-45 GB (transformer +
activations at 480x704x49: 2310 video + 51 audio tokens). Time: ~1 min per DiT forward on 12
vCPUs (estimate), ~20-25 min for 20 steps at guidance 1, plus ~5-10 min of loading.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch

DTYPE = torch.bfloat16
FRAME_RATE = 24.0  # LTX2Orchestrator / application default; the CLI does not override it
DEVICE_TEXT_SEQ_LEN = 1024  # difflet.models.ltx_2.application.LTX_2_DEFAULT_TEXT_SEQ_LEN

COND_KEYS = (
    "encoder_hidden_states",
    "audio_encoder_hidden_states",
    "encoder_attention_mask",
    "audio_encoder_attention_mask",
)


# --------------------------------------------------------------------------- loaders
# Module-level so the unit tests can monkeypatch them (no real weights in tests).


def load_transformer(model_dir: Path, dtype: torch.dtype = DTYPE) -> torch.nn.Module:
    """The real LTX-2 DiT on the CPU, as ``NeuronLTX2TransformerApplication._load_cpu_transformer``."""
    from difflet.backends.trainium.ltx_2.segmented import _patch_ltx2_rope

    _patch_ltx2_rope()
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    model = LTX2VideoTransformer3DModel.from_pretrained(str(Path(model_dir) / "transformer"), torch_dtype=dtype)
    return model.to(dtype=dtype).eval()


def load_scheduler(model_dir: Path) -> Any:
    from difflet.models.ltx_2.pipeline import _load_scheduler

    scheduler = _load_scheduler(str(model_dir))
    if scheduler is None:
        raise SystemExit(f"LTX-2 scheduler config missing under {model_dir}/scheduler")
    return scheduler


def compute_conditioning(args, dtype: torch.dtype = DTYPE) -> dict[str, torch.Tensor]:
    """Gemma3 + connectors via the diffusers host pipeline, then free them."""
    from difflet.models.ltx_2.application import _load_ltx_2_host_pipeline
    from difflet.models.ltx_2.pipeline import LTX2Orchestrator

    started = time.perf_counter()
    pipe = _load_ltx_2_host_pipeline(model_path=str(args.model_dir), dtype=dtype, device="cpu",
                                     load_decode_components=False)
    print(f"[calib] LTX-2 text encoder + connectors loaded in {time.perf_counter() - started:.1f}s", flush=True)
    orch = LTX2Orchestrator(model_path=str(args.model_dir), host_pipeline=pipe, dtype=dtype,
                            height=args.height, width=args.width, num_frames=args.num_frames,
                            text_seq_len=args.text_seq_len, frame_rate=FRAME_RATE)
    started = time.perf_counter()
    with torch.no_grad():
        # The device passes max(guidance, audio_guidance); audio_guidance defaults to guidance.
        values = orch.prepare_conditioning(prompt=args.prompt, guidance_scale=float(args.guidance_scale),
                                           num_videos_per_prompt=1, max_sequence_length=args.text_seq_len)
    cond = {k: v.detach().cpu().clone() for k, v in zip(COND_KEYS, values)}
    print(f"[calib] conditioning {tuple(cond['encoder_hidden_states'].shape)} in "
          f"{time.perf_counter() - started:.1f}s", flush=True)
    del orch, pipe, values
    gc.collect()
    return cond


def load_conditioning_file(path: Path) -> dict[str, torch.Tensor]:
    """``--text-pt``: a ltx_2_cache_dit_inputs.py ``.safetensors`` bundle or a ``.pt`` dict."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        tensors = load_file(str(path), device="cpu")
    else:
        tensors = torch.load(str(path), map_location="cpu")
    missing = [k for k in COND_KEYS[:3] if k not in tensors]
    if missing:
        raise KeyError(f"--text-pt {path} lacks {missing} (needs {', '.join(COND_KEYS)})")
    cond = {k: tensors[k] for k in COND_KEYS if k in tensors}
    cond.setdefault("audio_encoder_attention_mask", cond["encoder_attention_mask"])
    return cond


def _validate_conditioning(cond: dict[str, torch.Tensor], *, guidance_scale: float, text_seq_len: int) -> None:
    batch = 2 if float(guidance_scale) > 1.0 else 1
    for key in COND_KEYS:
        t = cond[key]
        if t.shape[0] != batch or t.shape[1] != text_seq_len:
            raise ValueError(f"conditioning {key} has shape {tuple(t.shape)}; guidance {guidance_scale} "
                             f"and --text-seq-len {text_seq_len} need batch {batch} x {text_seq_len}")


def expected_hook_count(model: torch.nn.Module) -> int:
    from difflet.quant.spec import QuantSpec

    return int(model.config.num_layers) * len(QuantSpec.for_model("ltx_2").targets)


# --------------------------------------------------------------------------- adapter


class _DiffusersAdapter:
    """What ``LTX2Orchestrator`` calls (``transformer(bundle)``) on the plain diffusers model.

    Same kwargs as the single-mode Neuron wrapper (``backends/trainium/ltx_2/transformer.py``).
    """

    supports_ltx_2_extra_kwargs = False
    cfg_parallel_enabled = False

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype = DTYPE) -> None:
        self.model = model
        self.dtype = dtype
        self.geometry: dict[str, Any] = {}
        self.calls = 0

    def __call__(self, bundle):
        self.calls += 1
        dt = self.dtype
        return self.model(
            hidden_states=bundle.hidden_states.to(dt),
            audio_hidden_states=bundle.audio_hidden_states.to(dt),
            encoder_hidden_states=bundle.encoder_hidden_states.to(dt),
            audio_encoder_hidden_states=bundle.audio_encoder_hidden_states.to(dt),
            timestep=bundle.timestep.to(dt),
            audio_timestep=bundle.timestep.to(dt),
            sigma=bundle.sigma.to(dt),
            audio_sigma=bundle.sigma.to(dt),
            encoder_attention_mask=bundle.encoder_attention_mask,
            audio_encoder_attention_mask=bundle.audio_encoder_attention_mask,
            video_coords=bundle.video_coords,
            audio_coords=bundle.audio_coords,
            isolate_modalities=False,
            spatio_temporal_guidance_blocks=None,
            perturbation_mask=None,
            use_cross_timestep=False,
            return_dict=False,
            **self.geometry,
        )


# --------------------------------------------------------------------------- entry point


def run(args, install_hooks) -> float:
    from difflet.models.ltx_2.pipeline import LTX2Orchestrator

    torch.set_num_threads(int(args.threads))
    if int(args.text_seq_len) != DEVICE_TEXT_SEQ_LEN:
        print(f"[calib] WARNING: --text-seq-len {args.text_seq_len}; the LTX-2 device path encodes "
              f"{DEVICE_TEXT_SEQ_LEN} tokens", flush=True)

    if args.text_pt:
        cond = load_conditioning_file(args.text_pt)
        print(f"[calib] conditioning from {args.text_pt}", flush=True)
    else:
        cond = compute_conditioning(args)
        saved = Path(args.out).with_name(Path(args.out).stem + "_conditioning.pt")
        try:
            saved.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cond, saved)
            print(f"[calib] conditioning saved to {saved} (reuse with --text-pt)", flush=True)
        except OSError as exc:  # a convenience only
            print(f"[calib] could not save conditioning ({exc})", flush=True)
    _validate_conditioning(cond, guidance_scale=args.guidance_scale, text_seq_len=int(args.text_seq_len))

    started = time.perf_counter()
    model = load_transformer(Path(args.model_dir), DTYPE)
    print(f"[calib] LTX-2 transformer loaded in {time.perf_counter() - started:.1f}s", flush=True)
    hooked = install_hooks(model)
    expected = expected_hook_count(model)
    if hooked != expected:
        raise RuntimeError(f"hooked {hooked} LTX-2 target linears, expected {expected} "
                           f"({model.config.num_layers} blocks x 28)")

    adapter = _DiffusersAdapter(model, DTYPE)
    orch = LTX2Orchestrator(model_path=str(args.model_dir), transformer=adapter, dtype=DTYPE,
                            height=args.height, width=args.width, num_frames=args.num_frames,
                            text_seq_len=int(args.text_seq_len), frame_rate=FRAME_RATE,
                            scheduler=load_scheduler(Path(args.model_dir)))
    # The single-mode wrapper's static geometry (LTX2TransformerInferenceConfig).
    adapter.geometry = dict(num_frames=orch.latent_num_frames, height=orch.latent_height,
                            width=orch.latent_width, fps=FRAME_RATE,
                            audio_num_frames=orch.inferred_audio_num_frames)
    audio_channels = int(model.config.audio_in_channels) // orch.latent_mel_bins
    print(f"[calib] latents {orch.latent_num_frames}x{orch.latent_height}x{orch.latent_width} "
          f"({orch.video_seq_len} video tokens), {orch.audio_seq_len} audio tokens, "
          f"{args.steps} steps, guidance {args.guidance_scale}", flush=True)

    started = time.perf_counter()
    try:
        with torch.no_grad():
            orch(encoder_hidden_states=cond["encoder_hidden_states"],
                 audio_encoder_hidden_states=cond["audio_encoder_hidden_states"],
                 encoder_attention_mask=cond["encoder_attention_mask"].to(torch.bool),
                 audio_encoder_attention_mask=cond["audio_encoder_attention_mask"].to(torch.bool),
                 video_channels=int(model.config.in_channels), audio_channels=audio_channels,
                 num_inference_steps=int(args.steps), guidance_scale=float(args.guidance_scale),
                 generator=torch.Generator().manual_seed(int(args.seed)), output_type="latent")
    finally:
        print(f"[calib] {adapter.calls} DiT passes in {time.perf_counter() - started:.1f}s", flush=True)
    return time.perf_counter() - started
