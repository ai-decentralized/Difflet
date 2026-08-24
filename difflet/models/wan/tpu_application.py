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

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: Encode outcomes broadcast alongside the embeddings so every rank fails the
#: same way. A raise on the encoding rank alone would hang the others on the
#: collective.
_ENCODE_OK = 0
_ENCODE_FAILED = 1


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype.replace("torch.", ""), None)
        if isinstance(resolved, torch.dtype):
            return resolved
    return torch.bfloat16


class _BroadcastTextEncoder:
    """umT5 on one rank, broadcast to the rest.

    ``WanOrchestrator.encode_prompt`` tokenizes on every rank -- cheap, CPU
    only, and deterministic -- then calls this with ``(input_ids,
    attention_mask)`` padded to the compiled text length. Only XLA ordinal 0
    holds the model:

    * three of the four encodes were computing the same tensor from the same
      prompt, and
    * one fp32 copy is ~11 GiB where four are ~48 GiB apiece in practice and
      OOM-kill a 188 GiB host, which is what would otherwise force bf16 -- and
      bf16 is emulated on this AMD EPYC host, 4.5x slower than fp32.

    It must be the XLA *ordinal*, not the worker index: the two are a scrambled
    mapping, and broadcasting from the wrong one ships a zero placeholder to
    everyone without failing.

    The padded positions are trimmed before the model runs and re-padded after.
    T5 attention is masked, so real tokens never attend to padding and those
    positions are zeroed downstream either way -- the same tensor for a
    fraction of the work.
    """

    def __init__(self, model_path: str, text_dim: int, seq_len: int, dtype):
        import torch_xla.runtime as xr

        self.text_dim = int(text_dim)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.is_encoder = int(xr.global_ordinal()) == 0
        self.model = None
        if self.is_encoder:
            from transformers import UMT5EncoderModel

            self.model = UMT5EncoderModel.from_pretrained(
                os.path.join(model_path, "text_encoder"), dtype=torch.float32
            ).eval()
            self.model.requires_grad_(False)

    def __call__(self, input_ids, attention_mask):
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = torch_xla.device()
        embeds = torch.zeros(1, self.seq_len, self.text_dim, dtype=self.dtype)
        status = _ENCODE_OK
        if self.is_encoder:
            # Must not raise before the collective: every other rank is
            # already committed to reaching it and would block until the
            # engine's cancel timeout.
            try:
                valid = max(int(attention_mask.sum()), 1)
                with torch.no_grad():
                    hidden = self.model(
                        input_ids[:, :valid].to(torch.int64),
                        attention_mask[:, :valid].to(torch.int32),
                    ).last_hidden_state
                embeds[:, :valid] = hidden.to(self.dtype)
            except Exception:  # noqa: BLE001 - re-raised below on every rank
                logger.exception("wan.tpu_encode_failed")
                status = _ENCODE_FAILED

        payload = [embeds.to(device), torch.tensor([status], dtype=torch.int32).to(device)]
        xm.collective_broadcast(payload, root_ordinal=0)
        xm.mark_step()
        if int(payload[1].cpu()[0]) != _ENCODE_OK:
            raise RuntimeError("Wan prompt encoding failed on the encoder rank")
        return payload[0].cpu()


class _OnDeviceTransformer:
    """Move a DiT call onto the chips and its result back.

    The orchestrator holds the latents on the host in fp32 because UniPC's
    order-2 corrector collapses in bf16. Keeping them on device instead was
    measured and is worth ~2 ms/step, so the round-trip stays.
    """

    def __init__(self, module, config, dtype):
        self.module = module
        self.config = config
        self.dtype = dtype

    def __call__(self, hidden_states, timestep, encoder_hidden_states):
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = torch_xla.device()
        out = self.module(
            hidden_states.to(self.dtype).to(device),
            timestep.to(self.dtype).to(device),
            encoder_hidden_states.to(self.dtype).to(device),
        )
        xm.mark_step()
        return out.cpu()


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
        self.text_encoder = None
        self.pipeline = None

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

    def load_eager(self) -> None:
        """Bring the model up for serving, with no compiled artifact.

        Direction A (torch.export -> StableHLO) exists in
        ``TpuApplicationBase`` but has never been exercised on a real model,
        and an artifact would save tracing rather than compilation, so serving
        does not depend on it. This is the path ``difflet serve`` uses.

        Builds the orchestrator afterwards, so ``pipeline`` is the same
        backend-neutral ``WanOrchestrator`` the Trainium adapter hands to the
        stage runners -- which is why those runners need no TPU branch.
        """
        import torch_xla
        import torch_xla.core.xla_model as xm

        from difflet.models.wan.pipeline import WanOrchestrator

        expert = self._require_transformer("load")
        module = expert._prepare_module().to(torch_xla.device())
        xm.mark_step()
        xm.wait_device_ops()

        transformer = _OnDeviceTransformer(module, self.config, self.dtype)
        transformer_2 = None
        if self.transformer_2 is not None:
            module_2 = self.transformer_2._prepare_module().to(torch_xla.device())
            xm.mark_step()
            xm.wait_device_ops()
            transformer_2 = _OnDeviceTransformer(
                module_2, self.transformer_2.config, self.dtype
            )

        self.text_encoder = _BroadcastTextEncoder(
            self.model_path, int(self.config.text_dim), self.text_seq_len, self.dtype
        )
        self.pipeline = WanOrchestrator(
            model_path=self.model_path,
            text_encoder=self.text_encoder,
            transformer=transformer,
            transformer_2=transformer_2,
            # The VAE decodes on the host: AutoencoderKLWan raises an
            # unsupported-negative-index error under torch_xla, and the serving
            # stage runner owns that decode anyway.
            vae_decoder=None,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            max_text_length=self.text_seq_len,
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
