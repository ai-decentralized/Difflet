"""Neuron application for Wan's VAE decoder.

Wraps the backend-neutral ``WanVAEDecoderModel`` (``difflet/models/wan/vae/modeling_vae.py``),
which mirrors diffusers' decoder key for key (``post_quant_conv.*``, ``decoder.*``) and decodes
the latent frame by frame with diffusers' causal convolution cache. The checkpoint's encoder
tensors match no parameter and are skipped.

Precision (``default_vae_dtype``) follows the Trainium path, clip length by clip length: bf16 up to
``BF16_MAX_FRAMES`` frames, where it decodes on the device in its pipeline dtype (bf16), and fp32
beyond, where its compiled decoder exceeds the compiler's instruction limit and it decodes with
diffusers' VAE on the host in fp32 (as Wan's model cards load it). This application runs eagerly,
so long clips also decode on the device, at the same fp32. An explicit ``dtype`` (fp32 or bf16;
the composed application's ``vae_dtype``, the CLI's ``--vae-dtype``) overrides that choice. In
bf16 the numerically sensitive steps stay fp32 (``difflet/models/wan/vae/modeling_vae.py``): the
RMS norms normalise in fp32, as HunyuanVideo's bf16 decoder needed for its GroupNorm statistics
(F2); the written-out attention's scores and softmax are fp32; and the output head
(``fp32_output_head=True``: ``conv_out`` and the clamp) runs in fp32, so the frames come back fp32
instead of rounded to bf16. It has no tensor-parallel layers, so every rank that runs it holds the
whole 0.5 GB decoder (0.25 GB in bf16); the CLI runs it on one rank, in a launch after the
transformer stage (its fp32 activations are about 12 GiB per core at 480x832), as the Trainium
path runs its VAE stage.

Execution (``exec_mode``): ``"eager"`` by default (unchanged): the decode runs op by op and the
device builds a small per-op kernel for each op and shape. ``"compile"`` is the opt-in (the CLI's
``--vae-exec-mode compile``; phase 7 of the 2026-10-26 native-vs-AoT plan, after HunyuanVideo's
compiled decoder, F1a ``1c10dfc``): the decode runs through ``WanVAERegionDecoder``
(``difflet/models/wan/vae/regions.py``), the same weights as fixed-shape regions (per resnet a
norm + SiLU, a convolution, a norm + SiLU and a convolution + residual; the attention; per
upsampler the repeat and its convolution; the output head), each compiled on its own with
``compile.compile_region``. The causal frame cache is threaded through the regions as explicit
inputs and outputs, so one set of graphs serves the first latent frame and one every later
frame, whatever the clip's length; the frame is not tiled (the decode is the reference's). The
latents and the decoded frames stay on the host between the regions' calls, so ``forward``
returns host frames. The warm-up (``load(skip_warmup=False)``) decodes at most two latent frames
of zeros, which builds or loads every graph; the CLI's generate-mode vae stage skips it, and its
first decode builds or loads them.
"""

from __future__ import annotations

import json
import os

import torch

#: The longest clip the Trainium path decodes on the device (in bf16); longer clips decode on its
#: host fallback in fp32.
BF16_MAX_FRAMES = 9


def default_vae_dtype(num_frames: int) -> torch.dtype:
    """The VAE's precision for a clip of ``num_frames`` frames, matching the Trainium path."""
    return torch.bfloat16 if int(num_frames) <= BF16_MAX_FRAMES else torch.float32

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.backends.neuron.wan.compile_options import apply_ccop_overlap


class NeuronWanVaeDecoderApplication(TorchNeuronApplicationBase):
    """The whole Wan VAE decoder on each rank, at one static latent shape, eager (default) or
    compiled region by region, in fp32 or bf16 (the callers pick ``default_vae_dtype`` unless
    told otherwise).

    ``model_path`` is the VAE's directory (``<model>/vae``); ``latent_shape`` is the
    ``(batch, z_dim, latent_frames, latent_height, latent_width)`` the warm-up decodes. Its
    input is ``dtype``; its output is fp32 in either dtype (the fp32 output head).
    ``compile_backend`` overrides the ``torch.compile`` backend of the compile mode (default
    ``"neuron"`` on the device; the CPU tests pass ``"aot_eager"``). ``ccop_overlap=True`` (Wan's
    callers only) applies Wan's ccop overlap compiler flag on the neuron device
    (``wan/compile_options.py``); Qwen-Image, which reuses this decoder, leaves it False.
    """

    EXEC_MODES = ("eager", "compile")
    #: Dynamo's recompile limit in the compile mode: regions of one class share a ``forward``,
    #: and each takes one shape per decoder level and kind of latent frame (first or later):
    #: about a dozen for the convolution region at Wan's 480x832.
    RECOMPILE_LIMIT = 64
    #: The compile mode's warm-up decodes this many latent frames at most: the first frame and
    #: one later frame build every graph a decode of any length runs.
    WARMUP_FRAMES = 2

    block_attrs = ()
    #: The checkpoint's encoder half, unused by decoding.
    skipped_checkpoint_prefixes = ("encoder.", "quant_conv.")

    def __init__(
        self,
        *,
        model_path,
        latent_shape: tuple[int, ...],
        parallel=None,
        dtype=torch.float32,
        exec_mode: str | None = "eager",
        device="neuron",
        compile_backend=None,
        ccop_overlap: bool = False,
        **kwargs,
    ) -> None:
        exec_mode = "eager" if exec_mode is None else exec_mode
        if exec_mode not in self.EXEC_MODES:
            raise ValueError(
                f"the Wan VAE decoder runs eager or compile, got exec_mode={exec_mode!r}")
        super().__init__(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            exec_mode=exec_mode,
            device=device,
            **kwargs,
        )
        # Before anything compiles: Wan's ccop overlap flag (wan/compile_options.py), only for the
        # Wan callers: Qwen-Image builds this decoder too and keeps its compile flags.
        if ccop_overlap:
            apply_ccop_overlap(self.device, self)
        if compile_backend is not None:
            self.compile_backend = compile_backend
        if self.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"the Wan VAE decoder runs in fp32 or bf16, got {self.dtype}")
        self.latent_shape = tuple(int(n) for n in latent_shape)
        self.vae_config = self._read_config()
        if len(self.latent_shape) != 5 or self.latent_shape[1] != int(self.vae_config.z_dim):
            raise ValueError(
                f"latent_shape {self.latent_shape} must be (batch, z_dim={self.vae_config.z_dim}, "
                "frames, height, width)"
            )

    def _read_config(self):
        from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig

        path = os.path.join(str(self.model_path), "config.json")
        with open(path, encoding="utf-8") as f:
            return WanVAEDecoderConfig.from_diffusers_dict(json.load(f))

    def build_module(self) -> torch.nn.Module:
        from difflet.models.wan.vae.modeling_vae import WanVAEDecoderModel

        # The fp32 output head is a no-op in fp32 (the fp32 decode is unchanged).
        return WanVAEDecoderModel(self.vae_config, fp32_output_head=True)

    def checkpoint_key(self, name: str) -> str:
        return name  # diffusers' decoder layout, key for key

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        shape = self.latent_shape
        if self.exec_mode == "compile":  # the first and one later latent frame build every graph
            shape = (*shape[:2], min(shape[2], self.WARMUP_FRAMES), *shape[3:])
        return (torch.zeros(shape, dtype=self.dtype),)

    def load(self, compiled_model_path=None, start_rank_id=None, local_ranks_size=None,
             skip_warmup: bool = False) -> None:
        """The base load; in the compile mode the loaded model is then wrapped in
        ``WanVAERegionDecoder`` (``self.module``), its regions compiled, and the warm-up decodes
        ``get_example_inputs`` through them."""
        if self.exec_mode != "compile" or self.is_loaded:
            super().load(compiled_model_path, start_rank_id, local_ranks_size,
                         skip_warmup=skip_warmup)
            return
        # The base compiles nothing here (block_attrs is empty) and skips its warm-up.
        super().load(compiled_model_path, start_rank_id, local_ranks_size, skip_warmup=True)
        import torch._dynamo as dynamo

        from difflet.backends.neuron.compile import compile_region
        from difflet.models.wan.vae.regions import WanVAERegionDecoder

        with self._phase("compile regions"):
            decoder = WanVAERegionDecoder(self.module).eval().requires_grad_(False)
            dynamo.config.recompile_limit = max(dynamo.config.recompile_limit,
                                                self.RECOMPILE_LIMIT)
            regions = decoder.regions()
            for region in regions:
                compile_region(region, backend=self._compile_backend())
            self.compiled_blocks = [f"region {i}" for i in range(len(regions))]
            self.module = decoder
        if not skip_warmup:
            self.warmup()
            # Every later latent frame reuses the graphs the warm-up built, so the request's
            # shape is warm too (no unwarmed-shape warning at the first decode).
            self._compiled_shapes.add((tuple(self.latent_shape),))


__all__ = ["NeuronWanVaeDecoderApplication"]
