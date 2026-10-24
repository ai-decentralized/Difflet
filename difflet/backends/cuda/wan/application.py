"""Wan 2.2 on the GPU flow: both experts, UMT5 and the VAE on one GPU, driven by the
backend-neutral ``WanOrchestrator``.

* The two DiT experts are ``CudaWanTransformerApplication`` (the neutral
  ``WanTransformer3DModel`` on ``difflet.ops``), each loaded from its own diffusers directory
  and compiled into its own artifact subdirectory.
* UMT5 (``transformers.UMT5EncoderModel``) sits behind the ``(input_ids, attention_mask) →
  embeds`` callable ``WanOrchestrator.encode_prompt`` expects.
* diffusers' ``AutoencoderKLWan`` sits behind a decoder callable whose ``.config`` carries
  ``latents_mean``/``latents_std``, so the orchestrator denormalises as ``WanPipeline`` does.

**Expert residency.** Wan 2.2 A14B's two 14B experts are ~57 GB in bf16 together; with UMT5
(~11 GB) and the VAE they fit a 96 GB card, so ``enable_transformer_2`` defaults to True here
(the Trainium/TPU default stays False). Timesteps at or above ``boundary_ratio × 1000``
(``model_index.json``) run ``transformer``, below it ``transformer_2``.

**CFG.** ``guidance_scale > 1`` is the orchestrator's sequential cond + uncond passes per step;
``guidance_scale_2`` applies below the boundary. Loop latents stay on the host in fp32 —
UniPC's corrector collapses in bf16 — and the DiT/VAE callables move tensors to the device per
call and return host tensors.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from difflet.backends.gpu.application import GpuApplication
from difflet.models.wan.dit_config import WanDiTConfig
from difflet.models.wan.pipeline import _read_boundary_ratio
from difflet.models.wan.spec import SPEC

DEFAULT_STEPS = SPEC.default_steps
DEFAULT_GUIDANCE_SCALE = SPEC.default_guidance_scale  # the high-noise stage
DEFAULT_GUIDANCE_SCALE_2 = 3.0  # Wan 2.2's low-noise stage (below the boundary)


def _timestep_dtype(value: Any, model_dtype: torch.dtype) -> torch.dtype:
    """``"float32"`` (the default: diffusers' exact timestep) or ``"model"`` (the model-dtype
    cast, kept for A/B comparisons). ``None`` means unset, i.e. the default."""
    if value is None or value == "float32":
        return torch.float32
    if value == "model":
        return model_dtype
    raise ValueError(
        f"Unsupported timestep_dtype for Wan: {value!r}; expected 'float32' or 'model'"
    )


class _CudaTextEncoder:
    """UMT5 on the device behind ``WanOrchestrator.encode_prompt``'s callable.

    The orchestrator tokenizes to ``text_seq_len`` on the host; the padded tail is trimmed
    before the model runs and zero-filled after, returning ``(B, text_seq_len, text_dim)`` on
    the host in the model dtype. T5 attention is masked, so real tokens never attend to
    padding and the result is the same tensor for a fraction of the work.
    """

    def __init__(self, model_path: str, text_dim: int, text_seq_len: int, dtype, device):
        from transformers import UMT5EncoderModel

        self.text_dim = int(text_dim)
        self.text_seq_len = int(text_seq_len)
        self.dtype = dtype
        self.device = torch.device(device)
        model = UMT5EncoderModel.from_pretrained(
            os.path.join(model_path, "text_encoder"), dtype=dtype
        )
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)

    def __call__(self, input_ids, attention_mask):
        batch = int(input_ids.shape[0])
        valid = max(int(attention_mask.sum(dim=-1).max()), 1)
        mask = attention_mask[:, :valid].to(self.device, torch.int64)
        with torch.no_grad():
            hidden = self.model(
                input_ids[:, :valid].to(self.device, torch.int64), mask
            ).last_hidden_state
        # Zero past each row's own mask too: with batch > 1 the trimmed prefix is the
        # longest row's.
        hidden = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).to(self.dtype).cpu()
        embeds = torch.zeros(batch, self.text_seq_len, self.text_dim, dtype=self.dtype)
        embeds[:, :valid] = hidden
        return embeds


class _CudaDiT:
    """One expert behind the orchestrator's DiT callable.

    The orchestrator holds the loop latents on the host in fp32; each call moves
    the latents and text to the device in the model dtype, the timestep in ``timestep_dtype``
    (fp32 by default: the exact integer diffusers passes; the sinusoidal projection computes
    in fp32 anyway), and the output back to the host. ``dtype``, ``timestep_dtype`` and
    ``config`` are what the orchestrator reads off a component.
    """

    def __init__(self, app, dtype, device, timestep_dtype=torch.float32):
        self.app = app
        self.dtype = dtype
        self.timestep_dtype = timestep_dtype
        self.device = torch.device(device)
        self.config = getattr(app, "dit_config", None)

    def __call__(self, hidden_states, timestep, encoder_hidden_states):
        with torch.no_grad():
            out = self.app(
                hidden_states.to(self.device, self.dtype),
                timestep.to(self.device, self.timestep_dtype),
                encoder_hidden_states.to(self.device, self.dtype),
            )
        return out.cpu()


class _CudaVaeDecoder:
    """``AutoencoderKLWan`` on the device behind the orchestrator's decoder callable.

    ``.config`` is the VAE's own, so ``WanOrchestrator._decode_latents`` denormalises with
    ``latents_mean``/``latents_std``; this decodes and does nothing else: the result is
    diffusers' ``(B, 3, F, H, W)`` float32 tensor in ``[-1, 1]``, on the host.
    """

    def __init__(self, vae):
        self.vae = vae
        self.config = vae.config
        self.dtype = vae.dtype
        self.device = vae.device

    def __call__(self, latents):
        with torch.no_grad():
            frames = self.vae.decode(latents.to(self.device, self.dtype), return_dict=False)[0]
        return frames.float().cpu()


class CudaWanApplication(GpuApplication):
    spec = SPEC
    expert_subdirectories = True

    def _configure(self, kwargs: dict[str, Any]) -> None:
        self.timestep_dtype = _timestep_dtype(kwargs.get("timestep_dtype"), self.dtype)
        self.text_seq_len = int(kwargs.get("text_seq_len", 512))
        self.batch_size = int(kwargs.get("batch_size", 1))
        # See the module docstring: both experts fit a 96 GB card, so on the GPU flow the
        # low-noise expert is resident by default.
        self.enable_transformer_2 = bool(kwargs.get("enable_transformer_2", True))
        self.load_text_encoder = bool(kwargs.get("load_text_encoder", True))
        self.load_vae = bool(kwargs.get("load_vae", True))
        self.transformer_2_path = os.path.join(self.model_path, "transformer_2")
        self.transformer_2 = None
        self.text_encoder = None
        self.vae_decoder = None
        self.pipeline = None

    def _build_transformers(self) -> None:
        from difflet.backends.cuda.wan.transformer import CudaWanTransformerApplication

        self.config = self._expert_config(self.transformer_path)
        self.transformer = CudaWanTransformerApplication(
            model_path=self.transformer_path,
            config=self.config,
            device=self.device,
            torch_compile=self.torch_compile,
        )
        if not self.enable_transformer_2:
            return
        if os.path.exists(os.path.join(self.transformer_2_path, "config.json")):
            # Both A14B experts share one config, but read the second one's own config.json
            # anyway: a variant where they diverge would otherwise load against the wrong
            # geometry.
            self.transformer_2 = CudaWanTransformerApplication(
                model_path=self.transformer_2_path,
                config=self._expert_config(self.transformer_2_path),
                device=self.device,
                torch_compile=self.torch_compile,
            )
        elif _read_boundary_ratio(self.model_path) is not None:
            # boundary_ratio marks a two-expert (Wan 2.2) snapshot; without transformer_2
            # the low-noise steps would silently run on the high-noise expert.
            raise FileNotFoundError(
                f"{self.transformer_2_path!r} is missing: {self.model_path!r} is a two-expert "
                "Wan 2.2 snapshot (model_index.json has boundary_ratio), so this is a "
                "partial download. Re-download the snapshot, or pass "
                "enable_transformer_2=False to run the high-noise expert alone."
            )

    def _expert_config(self, transformer_path: str) -> WanDiTConfig:
        parallel = self.parallel
        cfg_parallel = bool(getattr(parallel, "cfg_parallel_enabled", False))
        return WanDiTConfig.from_pretrained(
            transformer_path,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            text_seq_len=self.text_seq_len,
            # CFG-parallel: the orchestrator's cfg branch hands the expert the
            # [uncond, cond] stack and the modeling asserts batch 2, so the warm-up and
            # the input contract are batch 2 too.
            batch_size=2 if cfg_parallel else self.batch_size,
            tp_degree=int(getattr(parallel, "tp_degree", 1) or 1),
            torch_dtype=self.dtype,
            context_parallel_enabled=int(getattr(parallel, "cp_degree", 1) or 1) > 1,
            cfg_parallel_enabled=cfg_parallel,
            sp_enabled=bool(getattr(parallel, "sp_enabled", False)),
        )

    def named_transformers(self):
        named = [("transformer", self.transformer)]
        if self.transformer_2 is not None:
            named.append(("transformer_2", self.transformer_2))
        return named

    # ------------------------------------------------ encoders / VAE / loop

    def load_companions(self) -> None:
        """UMT5 and the VAE on the device (or the injected callables), then the orchestrator."""
        from difflet.models.wan.pipeline import WanOrchestrator

        if self.load_text_encoder:
            self.text_encoder = _CudaTextEncoder(
                self.model_path,
                int(self.config.text_dim),
                self.text_seq_len,
                self.dtype,
                self.device,
            )
        else:
            self.text_encoder = self.kwargs.get("text_encoder")
        if self.load_vae:
            from diffusers import AutoencoderKLWan

            vae = AutoencoderKLWan.from_pretrained(
                os.path.join(self.model_path, "vae"), torch_dtype=self.dtype
            )
            vae = vae.to(self.device).eval()
            vae.requires_grad_(False)
            self.vae_decoder = _CudaVaeDecoder(vae)
        else:
            self.vae_decoder = self.kwargs.get("vae_decoder")

        transformer_2 = None
        if self.transformer_2 is not None:
            transformer_2 = _CudaDiT(
                self.transformer_2, self.dtype, self.device, self.timestep_dtype
            )
        self.pipeline = WanOrchestrator(
            model_path=self.model_path,
            text_encoder=self.text_encoder,
            transformer=_CudaDiT(self.transformer, self.dtype, self.device, self.timestep_dtype),
            transformer_2=transformer_2,
            vae_decoder=self.vae_decoder,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            max_text_length=self.text_seq_len,
        )

    # ------------------------------------------------------------- DiT API

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        cfg = self._require_transformer("input contract").dit_config
        batch = int(cfg.batch_size)
        return {
            "hidden_states": {
                "shape": (
                    batch,
                    int(cfg.in_channels),
                    cfg.latent_frames,
                    cfg.latent_height,
                    cfg.latent_width,
                ),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch,), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (batch, int(cfg.text_seq_len), int(cfg.text_dim)),
                "dtype": self.dtype,
            },
        }

    # ---------------------------------------------------------- generation

    def generate(
        self,
        prompt: str | list[str],
        *,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        guidance_scale_2: float = DEFAULT_GUIDANCE_SCALE_2,
        negative_prompt: str | list[str] | None = None,
        seed: int | None = None,
        latents: torch.Tensor | None = None,
        num_frames: int | None = None,
        output_type: str = "pt",
    ):
        """Text → video through the orchestrator's denoise loop with both experts.

        ``latents`` is ``(B, C, F', H', W')`` initial noise used verbatim (the end-to-end
        oracle feeds the same tensor to stock diffusers); otherwise ``seed`` draws it with a
        host generator (``prepare_latents`` draws on the host). One shape is loaded, so a
        different ``num_frames`` is refused.

        Returns the ``(B, 3, F, H, W)`` frames tensor in ``[-1, 1]`` for ``"pt"``, the
        latents for ``"latent"``.
        """
        if self.pipeline is None:
            raise RuntimeError("application is not loaded; call load() first")
        if not isinstance(prompt, str) and len(prompt) > 1:
            raise NotImplementedError(
                f"Wan generates one prompt at a time on the GPU flow, got {len(prompt)}; "
                "batch > 1 is not supported yet"
            )
        loaded_frames = self.shape["num_frames"]
        if num_frames is not None and int(num_frames) != loaded_frames:
            raise ValueError(
                f"Wan was loaded for num_frames={loaded_frames}, got num_frames={num_frames}; "
                "one shape per process (more is not supported yet)"
            )
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            guidance_scale_2=None if guidance_scale_2 is None else float(guidance_scale_2),
            latents=latents,
            channels=int(self.config.in_channels),
            num_frames=loaded_frames,
            generator=generator,
            output_type=output_type,
            return_dict=False,
        )[0]


__all__ = [
    "CudaWanApplication",
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_GUIDANCE_SCALE_2",
    "DEFAULT_STEPS",
]
