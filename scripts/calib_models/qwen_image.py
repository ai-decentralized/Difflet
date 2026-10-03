"""Qwen-Image activation calibration loop for ``scripts/ptq_calibrate_activations.py``.

``run(args, install_hooks) -> elapsed_seconds`` drives the real Qwen-Image denoise loop
on the CPU in bf16, reproducing what the device ``generate`` stage feeds the DiT:

* Model: the stock diffusers ``QwenImageTransformer2DModel.from_pretrained(<snapshot>/transformer)``.
  Its ``named_modules()`` names equal the HF checkpoint keys (``transformer_blocks.3.attn.to_q``,
  ``transformer_blocks.3.img_mlp.net.2``), which is what ``difflet.quant.checkpoint.calibrated_amax``
  looks up. Difflet's ``_QwenImageTransformerTraceModule`` would prefix every name with
  ``transformer.`` and carries TP-only replacements, so it is not used here.
* Conditioning: the diffusers ``QwenImagePipeline.encode_prompt`` (Qwen2.5-VL, the chat template,
  first 34 template tokens dropped, last hidden state after the final norm -- the same tensor the
  CLI's ``_stage_text`` captures at ``norm``), zero-padded to the device's ``TEXT_SEQ_LEN`` (1024).
  Like the device trace module, the text mask is DROPPED: the DiT sees all 1024 text rows, the
  zero padding included, unmasked (``encoder_hidden_states_mask=None``, so RoPE spans 1024 text
  positions as in the device's static RoPE).
* Loop: Difflet's ``QwenImageOrchestrator`` (same timestep/1000, same scheduler step) through a
  thin adapter, with the CLI's timesteps: ``sigmas = linspace(1, 1/steps, steps)`` and the
  resolution-dependent ``mu`` shift from the scheduler config. Noise: ``torch.manual_seed(seed)``
  then the orchestrator's bf16 ``randn`` -- the CLI's order.
* Guidance: Qwen-Image has ``guidance_embeds=False`` and the device path runs NO true CFG (one DiT
  call per step, no negative prompt), so ``--guidance-scale`` does not change any activation; it
  is forwarded as the (ignored) guidance tensor only. One hooked call per step.

``--text-pt`` (optional): a ``torch.save``'d dict in the CLI text stage's ``{work_dir}/text.pt``
format -- ``encoder_hidden_states`` (1, L, 3584) and ``encoder_hidden_states_mask`` (1, L) bool.
When given, Qwen2.5-VL is not loaded; the embeds are zero-padded/truncated to 1024 and the mask is
ignored (dropped, as on device). ``--text-seq-len`` is pinned to the device's 1024 (a different
value is reported and overridden, since the driver's default of 512 is Wan's). ``--num-frames``
is ignored.

Memory: Qwen2.5-VL (~16 GB bf16) is loaded first and freed before the ~41 GB transformer loads.
"""

from __future__ import annotations

import gc
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

DTYPE = torch.bfloat16
SCRIPTS = Path(__file__).resolve().parents[1]


def _device_text_seq_len() -> int:
    from difflet.common.orchestrators.qwen_image import TEXT_SEQ_LEN

    return TEXT_SEQ_LEN


def _cache_inputs_module():
    """``scripts/qwen_image_cache_dit_inputs.py`` (the text-pipeline loader and normalisation)."""
    path = SCRIPTS / "qwen_image_cache_dit_inputs.py"
    spec = importlib.util.spec_from_file_location("qwen_image_cache_dit_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pad(embeds: torch.Tensor, mask: torch.Tensor | None, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _cache_inputs_module().normalize_prompt_embeds(embeds, mask, max_sequence_length=seq_len,
                                                          pad_to_max_sequence_length=True)


def _encode_prompt(model_dir: Path, prompt: str, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Real Qwen2.5-VL conditioning, zero-padded to ``seq_len``; the encoder is freed on return."""
    helpers = _cache_inputs_module()
    pipe = helpers._load_text_pipeline(model_id=str(model_dir), dtype=DTYPE, device=torch.device("cpu"),
                                       revision=None, local_files_only=True)
    with torch.no_grad():
        embeds, mask = pipe.encode_prompt(prompt=prompt, device=torch.device("cpu"), num_images_per_prompt=1,
                                          max_sequence_length=seq_len)
    del pipe
    gc.collect()
    return _pad(embeds.to(DTYPE), mask, seq_len)


def _load_conditioning(args, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(args, "text_pt", None):
        text = torch.load(args.text_pt, map_location="cpu")
        return _pad(text["encoder_hidden_states"].to(DTYPE), text.get("encoder_hidden_states_mask"), seq_len)
    return _encode_prompt(Path(args.model_dir), args.prompt, seq_len)


def _load_transformer(model_dir: Path):
    from diffusers import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_pretrained(str(Path(model_dir) / "transformer"),
                                                      torch_dtype=DTYPE).eval()


class _ModelAdapter:
    """What ``QwenImageOrchestrator`` calls: ``transformer(bundle)``, mask dropped as on device."""

    def __init__(self, model, packed_hw: tuple[int, int]):
        self.model = model
        self.dtype = DTYPE
        self.packed_hw = packed_hw

    def __call__(self, bundle):
        ph, pw = self.packed_hw
        return self.model(hidden_states=bundle.hidden_states, timestep=bundle.timestep,
                          encoder_hidden_states=bundle.encoder_hidden_states, encoder_hidden_states_mask=None,
                          img_shapes=[[(1, ph, pw)]] * int(bundle.hidden_states.shape[0]),
                          return_dict=False)[0]


def _mu(scheduler, height: int, width: int) -> float:
    """The CLI ``_stage_generate`` resolution shift."""
    sc = scheduler.config
    image_seq_len = (height // 16) * (width // 16)
    slope = (sc.max_shift - sc.base_shift) / (sc.max_image_seq_len - sc.base_image_seq_len)
    return image_seq_len * slope + (sc.base_shift - slope * sc.base_image_seq_len)


def run(args, install_hooks) -> float:
    from difflet.models.qwen_image.pipeline import QwenImageOrchestrator

    torch.set_num_threads(int(args.threads))
    seq_len = _device_text_seq_len()
    if int(getattr(args, "text_seq_len", seq_len)) != seq_len:
        print(f"[calib] qwen_image: --text-seq-len {args.text_seq_len} overridden by the device's {seq_len}",
              flush=True)
    height, width, steps = int(args.height), int(args.width), int(args.steps)

    started = time.perf_counter()
    embeds, mask = _load_conditioning(args, seq_len)
    print(f"[calib] text conditioning {tuple(embeds.shape)} ({int(mask.sum())} real tokens, mask dropped) "
          f"in {time.perf_counter() - started:.1f}s", flush=True)

    started = time.perf_counter()
    model = _load_transformer(Path(args.model_dir))
    print(f"[calib] transformer loaded in {time.perf_counter() - started:.1f}s", flush=True)
    expected = int(model.config.num_layers) * 12
    hooked = install_hooks(model)
    if hooked != expected:
        raise RuntimeError(f"hooked {hooked} target linears, expected {expected} "
                           f"({model.config.num_layers} blocks x 12)")

    orch = QwenImageOrchestrator(model_path=str(args.model_dir), transformer=None, dtype=DTYPE,
                                 height=height, width=width, text_seq_len=seq_len)
    if orch.scheduler is None:
        raise RuntimeError(f"no scheduler/scheduler_config.json under {args.model_dir}")
    orch.transformer = _ModelAdapter(model, (orch.latent_height // 2, orch.latent_width // 2))
    sigmas = np.linspace(1.0, 1.0 / steps, steps).tolist()
    orch.scheduler.set_timesteps(sigmas=sigmas, mu=_mu(orch.scheduler, height, width), device="cpu")
    guidance = torch.full([1], float(args.guidance_scale), dtype=DTYPE)  # ignored: guidance_embeds=False
    torch.manual_seed(int(args.seed))
    started = time.perf_counter()
    with torch.no_grad():
        orch(encoder_hidden_states=embeds, encoder_hidden_states_mask=mask, guidance=guidance,
             timesteps=orch.scheduler.timesteps, num_inference_steps=steps, output_type="latent")
    return time.perf_counter() - started
