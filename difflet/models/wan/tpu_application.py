"""Wan TPU application.

Mirrors ``NeuronWanApplication``'s outward contract — the shape
``DiffletPipeline`` and ``WanOrchestrator`` call — but composes the TPU
component lifecycle underneath instead of ``MultiComponentApplication``.

Scoped to the DiT experts. The umT5 text encoder and the VAE stay host-side,
exactly as the Qwen-Image TPU path does: the stages are sequential, so they
never need to co-reside, and on a 16 GB chip the DiT's transient footprint
does not leave room for a passenger.

**Expert residency.** Wan 2.2 A14B ships two 14B experts selected by a
timestep boundary (``boundary_ratio`` in ``model_index.json``, 0.875 here:
timesteps above it use ``transformer``, below it ``transformer_2``). Each is
7.2 GiB per chip at tp=4, so one fits comfortably and two do not — 14.3 GiB
of the ~15.4 GiB usable, less than the DiT forward itself needs. So
``enable_transformer_2`` defaults to False, matching what Trainium serving
already does (``difflet/serving/models/wan.py``), and the high-noise expert
runs every step.

Follows Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md, extended to
Wan.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype.replace("torch.", ""), None)
        if isinstance(resolved, torch.dtype):
            return resolved
    return torch.bfloat16


class TpuWanApplication(torch.nn.Module):
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
        from difflet.backends.tpu.wan.config import TpuWanConfig
        from difflet.backends.tpu.wan.transformer import TpuWanTransformerApplication

        self.model_path = model_path
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.shape = {
            "height": int(shape.get("height") or 480),
            "width": int(shape.get("width") or 832),
            "num_frames": int(shape.get("num_frames") or 9),
        }
        self.kwargs = kwargs
        self.text_seq_len = int(kwargs.get("text_seq_len", 512))
        self.batch_size = int(kwargs.get("batch_size", 1))
        # See the module docstring: two resident experts do not fit a v5e chip
        # at tp=4, so this is opt-in rather than the Trainium default of True.
        self.enable_transformer_2 = bool(kwargs.get("enable_transformer_2", False))

        self.transformer_path = os.path.join(model_path, "transformer")
        self.transformer_2_path = os.path.join(model_path, "transformer_2")
        self.transformer = None
        self.transformer_2 = None

        if not os.path.exists(os.path.join(self.transformer_path, "config.json")):
            return

        self.config = TpuWanConfig.from_pretrained(
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
        self.transformer = TpuWanTransformerApplication(
            model_path=self.transformer_path, config=self.config
        )

        if self.enable_transformer_2 and os.path.exists(
            os.path.join(self.transformer_2_path, "config.json")
        ):
            # Both experts share the A14B config, but read the second one's own
            # config.json anyway — a variant where they diverge would otherwise
            # load silently against the wrong geometry.
            config_2 = TpuWanConfig.from_pretrained(
                self.transformer_2_path,
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
            self.transformer_2 = TpuWanTransformerApplication(
                model_path=self.transformer_2_path, config=config_2
            )

    # --------------------------------------------------- DiffletPipeline API

    def _experts(self) -> list:
        return [e for e in (self.transformer, self.transformer_2) if e is not None]

    def _require_transformer(self, action: str):
        if self.transformer is None:
            raise NotImplementedError(
                f"Wan {action} requires transformer/config.json under {self.model_path!r}"
            )
        return self.transformer

    def compile(self, compiled_model_path: str, debug: bool = False) -> None:
        self._require_transformer("compile")
        for name, expert in self._named_experts():
            expert.compile(os.path.join(compiled_model_path, name), debug=debug)

    def load(
        self,
        compiled_model_path: str,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        self._require_transformer("load")
        for name, expert in self._named_experts():
            expert.load(
                os.path.join(compiled_model_path, name),
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
                skip_warmup=skip_warmup,
            )

    def has_compiled_artifacts(self, compiled_model_path: str) -> bool:
        if self.transformer is None:
            return False
        return all(
            expert.has_compiled_artifacts(os.path.join(compiled_model_path, name))
            for name, expert in self._named_experts()
        )

    def _named_experts(self) -> list[tuple[str, Any]]:
        # Each expert owns its own artifact subdirectory: they are separate
        # 14B checkpoints, so one shared directory would have the second
        # overwrite the first's per-rank graphs.
        named = [("transformer", self.transformer)]
        if self.transformer_2 is not None:
            named.append(("transformer_2", self.transformer_2))
        return named

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        cfg = self._require_transformer("input contract").config
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

    def __call__(self, *args: Any, **kwargs: Any):
        return self._require_transformer("forward")(*args, **kwargs)


__all__ = ["TpuWanApplication"]
