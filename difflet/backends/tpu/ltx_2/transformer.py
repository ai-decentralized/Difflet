"""TPU application for the LTX-2 audiovisual DiT.

Builds diffusers' ``LTX2VideoTransformer3DModel`` and shards it with the same
recipe the Trainium wrapper uses (``difflet/models/ltx_2/modeling_ltx_2.py``, through
``backends/tpu/ltx_2/sharding.py``: head-sharded attention with a global qk RMS norm, per-rank RoPE, column/row
parallel FFN), then loads each rank's slices lazily from the safetensors
checkpoint. The checkpoint keys are diffusers' own, so no renames.

Fourth model on the TPU backend; see
docs/plans/2026-09-11-tpu-port-hunyuan-ltx2-flux.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from difflet.backends.tpu.core.application_base import TpuApplicationBase
from difflet.backends.tpu.core.checkpoint import load_checkpoint_into
from difflet.backends.tpu.core.weights import materialize_meta_
from difflet.backends.tpu.ops_impl import parallel_mesh
from difflet.pipeline.parallel_mesh import MeshSpec

logger = logging.getLogger(__name__)


def _tpu_apply_split_rotary_emb(x, freqs):
    """``modeling_ltx_2._difflet_apply_split_rotary_emb``, rewritten for TorchTPU's compiler.

    The shared copy (diffusers' code) rotates in a ``[B, H, T, D]`` view of the
    activations -- ``swapaxes(1, 2)`` there and back -- with in-place
    ``addcmul_`` on slices. Under ``torch.compile`` the closing
    ``swapaxes(1, 2).reshape(b, t, -1)`` reached TorchTPU's fx_to_mlir as a
    ``view`` of a non-contiguous tensor: "cannot create a view of shape
    [1, 2310, 1024] from ... [1, 2310, 8, 128] and strides
    [2365440, 128, 295680, 1]" (2026-10-28). An explicit ``.contiguous()`` was
    dropped by the compiler and failed the same way, so the fake and real
    strides disagree there -- most likely a TorchTPU issue.

    This rotates in the activations' own ``[B, T, H, D]`` layout instead: only
    the small cos/sin are transposed, nothing is updated in place, and the
    final reshape is of a contiguous tensor. Same math --
    ``(x1 cos - x2 sin, x2 cos + x1 sin)`` over the two halves of the last dim,
    in fp32. TPU-only so the shared module, and the graphs Trainium traces from
    it, stay as they are.
    """
    cos, sin = freqs
    x_dtype = x.dtype
    flatten = x.ndim != 4 and cos.ndim == 4
    if flatten:
        batch_size, num_heads, seq_len, _ = cos.shape
        x = x.reshape(batch_size, seq_len, num_heads, -1)
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(f"Expected x.shape[-1] to be even for split rotary, got {last}.")
    rotary_dim = last // 2
    first = x[..., :rotary_dim].float()
    second = x[..., rotary_dim:].float()
    out = torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)
    if flatten:
        out = out.reshape(batch_size, seq_len, -1)
    return out.to(dtype=x_dtype)


class TpuLTX2TransformerApplication(TpuApplicationBase):
    """LTX-2's DiT, sharded across the TP group."""

    def __init__(self, *, model_path, config):
        super().__init__(config=config)
        self.model_path = Path(model_path)

    def _init_runtime(self) -> None:
        from difflet.backends.tpu.ops_impl.platform import configure_matmul_precision

        configure_matmul_precision()
        parallel_mesh.init_parallel_mesh(
            MeshSpec(tp=int(getattr(self.config, "tp_degree", 1) or 1))
        )

    def build_module(self) -> torch.nn.Module:
        from difflet.backends.tpu.ltx_2.sharding import build_ltx2_transformer, shard_ltx2_transformer
        from difflet.ops import SPMDRank

        try:
            from accelerate import init_empty_weights
        except ImportError:  # pragma: no cover - accelerate is a hard dep
            init_empty_weights = None

        tp = int(getattr(self.config, "tp_degree", 1) or 1)

        def _build():
            transformer = build_ltx2_transformer(self.config)
            # After build_ltx2_transformer, which installs the shared copy: the
            # attention processor reads this module attribute at call time.
            import diffusers.models.transformers.transformer_ltx2 as ltx2_transformer

            ltx2_transformer.apply_split_rotary_emb = _tpu_apply_split_rotary_emb
            if tp > 1:
                # SPMDRank on TPU answers with this rank's Python int, which is
                # what the per-rank RoPE / qk-norm-weight slices need. The TPU
                # processor honors the text padding mask (see sharding.py).
                shard_ltx2_transformer(transformer, tp, SPMDRank(tp))
            return transformer

        if init_empty_weights is None:
            return _build()
        # include_buffers=False: LTX-2's rope modules keep no persistent state
        # (cos/sin are computed per call), but the pattern matches the other
        # ports and keeps any computed buffer real.
        with init_empty_weights(include_buffers=False):
            return _build()

    def _prepare_module(self) -> torch.nn.Module:
        if self.module is not None:
            return self.module
        try:
            parallel_mesh.get_mesh_spec()
        except RuntimeError:
            self._init_runtime()
        module = self.build_module()
        spec = self._mesh()
        dtype = getattr(self.config, "torch_dtype", None)
        materialize_meta_(module, dtype=dtype)
        load_checkpoint_into(
            module,
            self.model_path,
            tp_size=spec.tp,
            tp_rank=parallel_mesh.get_tp_rank(),
            dtype=dtype,
            strict=True,
        )
        module.requires_grad_(False)
        self.module = module.eval()
        return self.module

    def get_example_inputs(self) -> tuple:
        """Fixed-shape inputs matching ``LTX2DiTInputBundle`` (10 tensors)."""
        cfg = self.config
        dtype = getattr(cfg, "torch_dtype", torch.bfloat16)
        batch = int(cfg.batch_size)
        return (
            torch.zeros(batch, cfg.video_seq_len, int(cfg.in_channels), dtype=dtype),
            torch.zeros(batch, cfg.audio_seq_len, int(cfg.audio_in_channels), dtype=dtype),
            torch.zeros(batch, int(cfg.text_seq_len), int(cfg.video_text_dim), dtype=dtype),
            torch.zeros(batch, int(cfg.audio_text_seq_len), int(cfg.audio_text_dim), dtype=dtype),
            torch.full((batch,), 500.0, dtype=dtype),
            torch.full((batch,), 0.5, dtype=dtype),
            torch.ones(batch, int(cfg.text_seq_len), dtype=torch.bool),
            torch.ones(batch, int(cfg.audio_text_seq_len), dtype=torch.bool),
            torch.zeros(batch, 3, cfg.video_seq_len, 2, dtype=torch.float32),
            torch.zeros(batch, 1, cfg.audio_seq_len, 2, dtype=torch.float32),
        )

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        sigma,
        encoder_attention_mask,
        audio_encoder_attention_mask,
        video_coords,
        audio_coords,
    ):
        """The bundle's ten tensors → ``(video_out, audio_out)``.

        Same call as the Trainium trace module's forward (minus its CFG-parallel
        scatter/gather, which the TPU backend does not run).
        """
        if self.module is None:
            self._init_runtime()
            self._prepare_module()
        cfg = self.config
        return self.module(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            audio_timestep=timestep,
            sigma=sigma,
            audio_sigma=sigma,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=int(cfg.latent_num_frames),
            height=int(cfg.latent_height),
            width=int(cfg.latent_width),
            fps=float(cfg.frame_rate),
            audio_num_frames=int(cfg.audio_num_frames),
            video_coords=video_coords,
            audio_coords=audio_coords,
            isolate_modalities=False,
            spatio_temporal_guidance_blocks=None,
            perturbation_mask=None,
            use_cross_timestep=bool(getattr(cfg, "use_cross_timestep", False)),
            return_dict=False,
        )


__all__ = ["TpuLTX2TransformerApplication"]
