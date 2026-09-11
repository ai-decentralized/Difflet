"""HunyuanVideo on the TPU backend: the DiT on the chips, the rest on the host.

Third model on the TPU backend, built on the Wan port's structure
(``difflet/models/wan/tpu_application.py``):

* the DiT is ``difflet/backends/tpu/hunyuan_video/transformer.py`` — the same
  backend-neutral modeling as Trainium, sharded per rank at load time, run
  eagerly under torch_xla;
* the Llama-3 text encoder runs on the host on XLA ordinal 0 and its hidden
  state is broadcast to the other ranks (``TpuBroadcastLlamaEncoder``); one
  fp32 copy is ~32 GiB where four would not fit beside the DiT shards on a
  188 GiB host;
* CLIP-L and the causal 3D VAE stay on the host in the serving adapter, as
  they do on Trainium with ``--host-vae``.

The orchestrator is the same backend-neutral ``HunyuanVideoOrchestrator`` the
Trainium application builds, so the serving stage runners need no TPU branch
for the denoise.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any

import torch

from difflet.models.hunyuan_video.application import HunyuanVideoDiTInputBundle

logger = logging.getLogger(__name__)

_ENCODE_OK = 0
_ENCODE_FAILED = 1


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    text = str(dtype).lower().removeprefix("torch.")
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


class TpuBroadcastLlamaEncoder:
    """Llama-3 hidden state on one rank, broadcast to the rest.

    Presents the call surface the serving Llama stage already uses for the
    Neuron app — ``app(input_ids=, attention_mask=, position_ids=,
    sampling_params=)`` returning an object with ``captured_tensors[0]`` — so
    ``HunyuanVideoLlamaStageRunner`` runs unchanged. ``capture_layer`` is the
    index of the decoder layer whose *output* is captured (Trainium captures
    ``layers.29``); in HF's ``output_hidden_states`` indexing that is
    ``hidden_states[capture_layer + 1]``, i.e. the pre-final-norm state
    diffusers takes with ``num_hidden_layers_to_skip=2``.

    Only XLA ordinal 0 holds the model (see the module docstring); it must be
    the XLA ordinal, not the replica index, because the collective broadcasts
    from an ordinal. Every rank calls this — the stage runs on all replicas.
    """

    def __init__(self, model_path: str, *, seq_len: int, capture_layer: int, dtype):
        import torch_xla.runtime as xr

        self.seq_len = int(seq_len)
        self.capture_layer = int(capture_layer)
        self.dtype = dtype
        self.is_encoder = int(xr.global_ordinal()) == 0
        self.model = None
        encoder_path = os.path.join(model_path, "text_encoder")
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(encoder_path)
        self.hidden_size = int(config.hidden_size)
        if self.is_encoder:
            from transformers import LlamaModel

            # fp32: bf16 is emulated on the EPYC host and slower (see Wan).
            self.model = LlamaModel.from_pretrained(encoder_path, dtype=torch.float32).eval()
            self.model.requires_grad_(False)

    def __call__(self, *, input_ids, attention_mask, position_ids=None, sampling_params=None):
        import torch_xla
        import torch_xla.core.xla_model as xm

        del position_ids, sampling_params  # Neuron-app arguments; HF derives both
        device = torch_xla.device()
        hidden = torch.zeros(1, self.seq_len, self.hidden_size, dtype=self.dtype)
        status = _ENCODE_OK
        if self.is_encoder:
            # Must not raise before the collective: the other ranks are
            # already committed to reaching it.
            try:
                with torch.no_grad():
                    out = self.model(
                        input_ids=input_ids.to(torch.int64),
                        attention_mask=attention_mask.to(torch.int64),
                        output_hidden_states=True,
                    )
                hidden = out.hidden_states[self.capture_layer + 1].to(self.dtype)
            except Exception:  # noqa: BLE001 - re-raised below on every rank
                logger.exception("hunyuan_video.tpu_llama_encode_failed")
                status = _ENCODE_FAILED

        payload = [hidden.to(device), torch.tensor([status], dtype=torch.int32).to(device)]
        xm.collective_broadcast(payload, root_ordinal=0)
        xm.mark_step()
        if int(payload[1].cpu()[0]) != _ENCODE_OK:
            raise RuntimeError("HunyuanVideo Llama encoding failed on the encoder rank")
        return SimpleNamespace(captured_tensors=[payload[0].cpu()])


class _OnDeviceTransformer:
    """Move a DiT call onto the chips and its result back.

    The orchestrator keeps the latents on the host (its Euler step runs
    there), so each step is one round trip; measured on Wan as ~2 ms/step,
    not worth a device-resident loop for v1.
    """

    def __init__(self, module, config, dtype):
        self.module = module
        self.config = config
        self.dtype = dtype

    def __call__(self, bundle: HunyuanVideoDiTInputBundle):
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = torch_xla.device()
        out = self.module(
            bundle.hidden_states.to(self.dtype).to(device),
            bundle.timestep.to(self.dtype).to(device),
            bundle.encoder_hidden_states.to(self.dtype).to(device),
            bundle.encoder_attention_mask.to(device),
            bundle.pooled_projections.to(self.dtype).to(device),
            bundle.guidance.to(self.dtype).to(device),
            return_dict=False,
        )[0]
        xm.mark_step()
        return out.cpu()


class TpuHunyuanVideoApplication(torch.nn.Module):
    def __init__(
        self,
        *,
        model_path: str,
        parallel,
        dtype: Any,
        shape: dict[str, int | None],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from difflet.backends.tpu.hunyuan_video.config import TpuHunyuanVideoConfig
        from difflet.backends.tpu.hunyuan_video.transformer import (
            TpuHunyuanVideoTransformerApplication,
        )

        self.model_path = model_path
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.shape = {
            "height": int(shape.get("height") or 320),
            "width": int(shape.get("width") or 512),
            "num_frames": int(shape.get("num_frames") or 61),
        }
        self.kwargs = kwargs
        self.text_seq_len = int(kwargs.get("text_seq_len", 256))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.model_version = "1.0"

        self.transformer_path = os.path.join(model_path, "transformer")
        self.transformer = None
        self.pipeline = None

        if not os.path.exists(os.path.join(self.transformer_path, "config.json")):
            return

        self.config = TpuHunyuanVideoConfig.from_pretrained(
            self.transformer_path,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            text_seq_len=self.text_seq_len,
            batch_size=self.batch_size,
            tp_degree=int(parallel.tp_degree),
            torch_dtype=self.dtype,
            context_parallel_enabled=int(getattr(parallel, "cp_degree", 1)) > 1,
            cfg_parallel_enabled=bool(getattr(parallel, "cfg_parallel", False)),
            sp_enabled=bool(getattr(parallel, "sp_enabled", False)),
            cp_mode=getattr(parallel, "cp_mode", "gather_kv"),
        )
        self.transformer = TpuHunyuanVideoTransformerApplication(
            model_path=self.transformer_path, config=self.config
        )

    # --------------------------------------------------- DiffletPipeline API

    def _require_transformer(self, action: str):
        if self.transformer is None:
            raise NotImplementedError(
                f"HunyuanVideo {action} requires transformer/config.json under "
                f"{self.model_path!r}"
            )
        return self.transformer

    def compile(self, compiled_model_path: str, debug: bool = False) -> None:
        self._require_transformer("compile").compile(compiled_model_path, debug=debug)

    def load(
        self,
        compiled_model_path: str,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        self._require_transformer("load").load(
            compiled_model_path,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
            skip_warmup=skip_warmup,
        )

    def load_eager(self) -> None:
        """Bring the DiT up for serving with no compiled artifact, then build
        the orchestrator — the path ``difflet serve`` uses (see Wan)."""
        import torch_xla
        import torch_xla.core.xla_model as xm

        from difflet.models.hunyuan_video.pipeline import HunyuanVideoOrchestrator

        transformer = self._require_transformer("load")
        device = torch_xla.device()
        module = transformer._prepare_module().to(device)
        xm.mark_step()
        xm.wait_device_ops()
        # Compile the request-shaped graph now, a few ranks at a time (see
        # warmup_eager on why): the first execution costs ~100 s and ~43 GB
        # of host RAM per rank on a v5e, and the serving smoke would
        # otherwise trigger it on all four ranks at once.
        seconds = transformer.warmup_eager(module, device)
        logger.info("hunyuan_video.tpu_warmup seconds=%.1f", seconds)
        print(f"[hunyuan_video] tpu warmup (first compile) in {seconds:.1f}s", flush=True)

        self.pipeline = HunyuanVideoOrchestrator(
            model_path=self.model_path,
            transformer=_OnDeviceTransformer(module, self.config, self.dtype),
            vae=None,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            teacache_cadence=self.kwargs.get("teacache_cadence"),
            teacache_online_delta_alpha=self.kwargs.get("teacache_online_delta_alpha"),
        )

    def has_compiled_artifacts(self, compiled_model_path: str) -> bool:
        if self.transformer is None:
            return False
        return self.transformer.has_compiled_artifacts(compiled_model_path)

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        cfg = self._require_transformer("input contract").config
        batch = int(cfg.batch_size)
        text = int(cfg.text_seq_len)
        return {
            "hidden_states": {
                "shape": (
                    batch, int(cfg.in_channels), cfg.latent_frames,
                    cfg.latent_height, cfg.latent_width,
                ),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch,), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (batch, text, int(cfg.text_embed_dim)),
                "dtype": self.dtype,
            },
            "encoder_attention_mask": {"shape": (batch, text), "dtype": torch.int64},
            "pooled_projections": {
                "shape": (batch, int(cfg.pooled_projection_dim)),
                "dtype": self.dtype,
            },
            "guidance": {"shape": (batch,), "dtype": self.dtype},
        }

    def forward_dit(self, bundle: HunyuanVideoDiTInputBundle):
        if self.pipeline is not None:
            return self.pipeline.transformer(bundle)
        return self._require_transformer("forward")(*bundle.as_model_inputs())

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and isinstance(args[0], HunyuanVideoDiTInputBundle):
            return self.forward_dit(args[0])
        return self._require_transformer("forward")(*args, **kwargs)


__all__ = ["TpuBroadcastLlamaEncoder", "TpuHunyuanVideoApplication"]
