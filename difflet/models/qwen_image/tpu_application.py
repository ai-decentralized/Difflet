"""Qwen-Image TPU application.

Mirrors ``NeuronQwenImageApplication``'s outward contract — the shape
``DiffletPipeline`` and the orchestrators call — but composes the TPU
component lifecycle underneath instead of ``MultiComponentApplication``.

Scoped to the transformer component only, following the plan's
recommendation to prove the single-component path before porting the
multi-component compile/load algorithm. The text encoder, scheduler loop and
VAE stay host-side, exactly as the Trainium M4a closure does.

Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from difflet.models.qwen_image.contract import (
    QwenImageDiTInputBundle,
    normalize_dtype,
)


class TpuQwenImageApplication(torch.nn.Module):
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
        from difflet.backends.tpu.qwen_image.config import TpuQwenImageConfig
        from difflet.backends.tpu.qwen_image.transformer import (
            TpuQwenImageTransformerApplication,
        )

        self.model_path = model_path
        self.parallel = parallel
        self.dtype = normalize_dtype(dtype)
        self.shape = {
            "height": int(shape.get("height") or 1024),
            "width": int(shape.get("width") or 1024),
            "num_frames": None,
        }
        self.kwargs = kwargs
        self.transformer_path = os.path.join(model_path, "transformer")
        self.text_seq_len = int(kwargs.get("text_seq_len", 1024))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.transformer = None

        if not os.path.exists(os.path.join(self.transformer_path, "config.json")):
            return

        self.config = TpuQwenImageConfig.from_pretrained(
            self.transformer_path,
            height=self.shape["height"],
            width=self.shape["width"],
            text_seq_len=self.text_seq_len,
            batch_size=self.batch_size,
            tp_degree=int(parallel.tp_degree),
            torch_dtype=self.dtype,
            context_parallel_enabled=int(getattr(parallel, "cp_degree", 1)) > 1,
            cp_mode=getattr(parallel, "cp_mode", "gather_kv"),
        )
        self.transformer = TpuQwenImageTransformerApplication(
            model_path=self.transformer_path, config=self.config
        )

    # --------------------------------------------------- DiffletPipeline API

    def _require_transformer(self, action: str):
        if self.transformer is None:
            raise NotImplementedError(
                f"Qwen-Image {action} requires transformer/config.json under "
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

    def has_compiled_artifacts(self, compiled_model_path: str) -> bool:
        if self.transformer is None:
            return False
        return self.transformer.has_compiled_artifacts(compiled_model_path)

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        cfg = self._require_transformer("input contract").config
        batch = int(cfg.batch_size)
        return {
            "hidden_states": {
                "shape": (batch, int(cfg.image_seq_len), int(cfg.in_channels)),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch,), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (batch, int(cfg.text_seq_len), int(cfg.joint_attention_dim)),
                "dtype": self.dtype,
            },
            "encoder_hidden_states_mask": {
                "shape": (batch, int(cfg.text_seq_len)),
                "dtype": torch.bool,
            },
            "guidance": {"shape": (batch,), "dtype": self.dtype},
        }

    def forward_dit(self, bundle: QwenImageDiTInputBundle):
        return self._require_transformer("forward")(*bundle.as_model_inputs())

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and isinstance(args[0], QwenImageDiTInputBundle):
            return self.forward_dit(args[0])
        if kwargs and not args:
            return self.forward_dit(QwenImageDiTInputBundle(**kwargs))
        return self._require_transformer("forward")(*args, **kwargs)


__all__ = ["TpuQwenImageApplication"]
