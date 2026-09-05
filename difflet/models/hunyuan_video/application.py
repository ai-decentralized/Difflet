"""HunyuanVideo Trainium application skeleton.

M3 starts with registration, shape/cache plumbing, and explicit scope guards.
The transformer/text/VAE components land after the dual-stream attention
reference path is covered by CPU tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

try:
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.core.multi_component_application import (
        ComponentSpec,
        MultiComponentApplication,
    )
except (ImportError, FileNotFoundError) as exc:
    _TRAINIUM_IMPORT_ERROR = exc

    class NeuronConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Trainium runtime dependencies are unavailable.") from (
                _TRAINIUM_IMPORT_ERROR
            )

    @dataclass(frozen=True)
    class ComponentSpec:  # type: ignore[no-redef]
        name: str
        component: Any

    class MultiComponentApplication:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Trainium runtime dependencies are unavailable.") from (
                _TRAINIUM_IMPORT_ERROR
            )


def _load_diffusers_config(path: str):
    from difflet.utils.diffusers_adapter import load_diffusers_config

    return load_diffusers_config(path)


@dataclass(frozen=True)
class HunyuanVideoDiTInputBundle:
    """Host-side contract for one HunyuanVideo DiT call.

    M3 v0 keeps Llama3, CLIP, scheduler setup, and VAE decode outside the
    Trainium graph. The Trainium boundary is exactly this tuple.
    """

    hidden_states: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    pooled_projections: torch.Tensor
    guidance: torch.Tensor

    def as_model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.timestep,
            self.encoder_hidden_states,
            self.encoder_attention_mask,
            self.pooled_projections,
            self.guidance,
        )


@dataclass(frozen=True)
class HunyuanVideo15DiTInputBundle:
    """Host-side contract for one HunyuanVideo 1.5 DiT call.

    HunyuanVideo 1.5 uses Qwen2.5-VL embeddings, ByT5 glyph embeddings, and
    image-semantic embeddings in addition to the latent tensor.
    """

    hidden_states: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    timestep_r: torch.Tensor
    encoder_hidden_states_2: torch.Tensor
    encoder_attention_mask_2: torch.Tensor
    image_embeds: torch.Tensor

    def as_model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.timestep,
            self.encoder_hidden_states,
            self.encoder_attention_mask,
            self.timestep_r,
            self.encoder_hidden_states_2,
            self.encoder_attention_mask_2,
            self.image_embeds,
        )


def allowed_hunyuan_video_latent_shapes(config: Any) -> list[tuple[int, ...]]:
    """One latent shape per compiled bucket, largest first."""

    batch_size = int(getattr(config.neuron_config, "batch_size", 1))
    compile_shapes = getattr(config, "compile_shapes", None)
    if not compile_shapes:
        # Single-bucket path: honor the config's latent_* properties directly
        # (some callers provide latent dims without height/width).
        return [
            (
                batch_size,
                int(config.in_channels),
                int(config.latent_frames),
                int(config.latent_height),
                int(config.latent_width),
            )
        ]
    from difflet.backends.trainium.core.bucketing import canonicalize_shapes

    shapes = []
    for height, width, num_frames in canonicalize_shapes(compile_shapes):
        shapes.append(
            (
                batch_size,
                int(config.in_channels),
                (int(num_frames) - 1) // 4 + 1,
                int(height) // 8,
                int(width) // 8,
            )
        )
    return shapes


def validate_hunyuan_video_dit_inputs(
    bundle: HunyuanVideoDiTInputBundle,
    *,
    config: Any,
    dtype: torch.dtype,
) -> None:
    """Validate the M3 v0 embedding/latent contract before Trainium dispatch."""

    batch_size = int(getattr(config.neuron_config, "batch_size", 1))
    text_seq_len = int(getattr(config, "text_seq_len", 256))
    allowed_latent_shapes = allowed_hunyuan_video_latent_shapes(config)
    if tuple(bundle.hidden_states.shape) not in set(allowed_latent_shapes):
        raise ValueError(
            f"HunyuanVideo DiT input 'hidden_states' has shape {tuple(bundle.hidden_states.shape)}, "
            f"expected one of the compiled bucket shapes: {allowed_latent_shapes}."
        )
    if bundle.hidden_states.dtype != dtype:
        raise TypeError(
            f"HunyuanVideo DiT input 'hidden_states' has dtype {bundle.hidden_states.dtype}, "
            f"expected {dtype}."
        )
    expected = {
        "timestep": ((batch_size,), dtype),
        "encoder_hidden_states": (
            (batch_size, text_seq_len, int(config.text_embed_dim)),
            dtype,
        ),
        "encoder_attention_mask": ((batch_size, text_seq_len), torch.int64),
        "pooled_projections": ((batch_size, int(config.pooled_projection_dim)), dtype),
        "guidance": ((batch_size,), dtype),
    }
    tensors = {
        "timestep": bundle.timestep,
        "encoder_hidden_states": bundle.encoder_hidden_states,
        "encoder_attention_mask": bundle.encoder_attention_mask,
        "pooled_projections": bundle.pooled_projections,
        "guidance": bundle.guidance,
    }
    for name, tensor in tensors.items():
        shape, tensor_dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"HunyuanVideo DiT input {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {shape}."
            )
        if tensor.dtype != tensor_dtype:
            raise TypeError(
                f"HunyuanVideo DiT input {name!r} has dtype {tensor.dtype}, "
                f"expected {tensor_dtype}."
            )


def validate_hunyuan_video15_dit_inputs(
    bundle: HunyuanVideo15DiTInputBundle,
    *,
    config: Any,
    dtype: torch.dtype,
) -> None:
    """Validate the HunyuanVideo 1.5 embedding/latent contract before dispatch."""

    batch_size = int(getattr(config.neuron_config, "batch_size", 1))
    text_seq_len = int(getattr(config, "text_seq_len", 1000))
    text_seq_len_2 = int(getattr(config, "text_seq_len_2", 256))
    image_seq_len = int(getattr(config, "image_seq_len", 729))
    latent_shape = (
        batch_size,
        int(config.in_channels),
        int(config.latent_frames),
        int(config.latent_height),
        int(config.latent_width),
    )
    expected = {
        "hidden_states": (latent_shape, dtype),
        "timestep": ((batch_size,), dtype),
        "encoder_hidden_states": (
            (batch_size, text_seq_len, int(config.text_embed_dim)),
            dtype,
        ),
        "encoder_attention_mask": ((batch_size, text_seq_len), torch.int64),
        "timestep_r": ((batch_size,), dtype),
        "encoder_hidden_states_2": (
            (batch_size, text_seq_len_2, int(config.text_embed_2_dim)),
            dtype,
        ),
        "encoder_attention_mask_2": ((batch_size, text_seq_len_2), torch.int64),
        "image_embeds": ((batch_size, image_seq_len, int(config.image_embed_dim)), dtype),
    }
    tensors = {
        "hidden_states": bundle.hidden_states,
        "timestep": bundle.timestep,
        "encoder_hidden_states": bundle.encoder_hidden_states,
        "encoder_attention_mask": bundle.encoder_attention_mask,
        "timestep_r": bundle.timestep_r,
        "encoder_hidden_states_2": bundle.encoder_hidden_states_2,
        "encoder_attention_mask_2": bundle.encoder_attention_mask_2,
        "image_embeds": bundle.image_embeds,
    }
    for name, tensor in tensors.items():
        shape, tensor_dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"HunyuanVideo 1.5 DiT input {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {shape}."
            )
        if tensor.dtype != tensor_dtype:
            raise TypeError(
                f"HunyuanVideo 1.5 DiT input {name!r} has dtype {tensor.dtype}, "
                f"expected {tensor_dtype}."
            )


def create_hunyuan_video_backbone_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    text_seq_len: int = 256,
    batch_size: int = 1,
    context_parallel_enabled: bool = False,
    cp_mode: str = "gather_kv",
    sp_enabled: bool = False,
    compile_shapes=None,
):
    from difflet.backends.trainium.hunyuan_video.backbone import (
        HunyuanVideoBackboneInferenceConfig,
    )

    transformer_path = os.path.join(model_path, "transformer")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
    )
    extra = {}
    if compile_shapes:
        extra["compile_shapes"] = compile_shapes
    return HunyuanVideoBackboneInferenceConfig(
        neuron_config=neuron_config,
        load_config=_load_diffusers_config(transformer_path),
        height=height,
        width=width,
        num_frames=num_frames,
        text_seq_len=text_seq_len,
        context_parallel_enabled=context_parallel_enabled,
        cp_mode=cp_mode,
        sp_enabled=sp_enabled,
        **extra,
    )


def create_hunyuan_video15_backbone_config(
    *,
    transformer_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    text_seq_len: int = 1000,
    text_seq_len_2: int = 256,
    image_seq_len: int = 729,
    batch_size: int = 1,
):
    from difflet.backends.trainium.hunyuan_video.backbone15 import (
        HunyuanVideo15BackboneInferenceConfig,
    )

    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
    )
    return HunyuanVideo15BackboneInferenceConfig(
        neuron_config=neuron_config,
        load_config=_load_diffusers_config(transformer_path),
        height=height,
        width=width,
        num_frames=num_frames,
        text_seq_len=text_seq_len,
        text_seq_len_2=text_seq_len_2,
        image_seq_len=image_seq_len,
    )


def create_hunyuan_video_vae_decoder_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    batch_size: int = 1,
    compile_shapes=None,
):
    from difflet.backends.trainium.hunyuan_video.vae import HunyuanVideoVAEDecoderInferenceConfig

    vae_path = os.path.join(model_path, "vae")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
    )
    extra = {}
    if compile_shapes:
        extra["compile_shapes"] = compile_shapes
    return HunyuanVideoVAEDecoderInferenceConfig(
        neuron_config=neuron_config,
        load_config=_load_diffusers_config(vae_path),
        height=height,
        width=width,
        num_frames=num_frames,
        **extra,
    )


def create_hunyuan_video15_vae_decoder_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    batch_size: int = 1,
    tile_sample_min_height: int = 256,
    tile_sample_min_width: int = 256,
    tile_overlap_factor: float = 0.25,
):
    from difflet.backends.trainium.hunyuan_video.vae15 import (
        HunyuanVideo15VAEDecoderInferenceConfig,
    )

    vae_path = os.path.join(model_path, "vae")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
    )
    return HunyuanVideo15VAEDecoderInferenceConfig(
        neuron_config=neuron_config,
        load_config=_load_diffusers_config(vae_path),
        height=height,
        width=width,
        num_frames=num_frames,
        tile_sample_min_height=tile_sample_min_height,
        tile_sample_min_width=tile_sample_min_width,
        tile_overlap_factor=tile_overlap_factor,
    )


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported HunyuanVideo dtype: {dtype!r}")


class NeuronHunyuanVideoApplication(MultiComponentApplication):
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
        self.model_path = model_path
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.model_version = str(kwargs.get("model_version", "1.0"))
        default_shape = (
            {"height": 480, "width": 848, "num_frames": 121}
            if self.model_version == "1.5"
            else {"height": 320, "width": 512, "num_frames": 61}
        )
        self.shape = {
            "height": int(shape.get("height") or default_shape["height"]),
            "width": int(shape.get("width") or default_shape["width"]),
            "num_frames": int(shape.get("num_frames") or default_shape["num_frames"]),
        }
        # Optional bucket shape set: kwargs["shapes"] is a list of shape dicts /
        # (h, w, f) tuples compiled into ONE artifact (K DiT NEFFs, one weight
        # copy). self.shape becomes the largest (priority) shape.
        self.compile_shapes = None
        raw_shapes = kwargs.get("shapes")
        if raw_shapes:
            from difflet.backends.trainium.core.bucketing import canonicalize_shapes

            self.compile_shapes = canonicalize_shapes(raw_shapes)
            if self.model_version == "1.5" and len(self.compile_shapes) > 1:
                raise NotImplementedError(
                    "Multi-shape bucketing is not supported for HunyuanVideo 1.5 yet."
                )
            largest = self.compile_shapes[0]
            self.shape = {
                "height": largest[0],
                "width": largest[1],
                "num_frames": int(largest[2]),
            }
        self.kwargs = kwargs
        transformer_subfolder = str(kwargs.get("transformer_subfolder", "transformer"))
        self.transformer_path = os.path.join(model_path, transformer_subfolder)
        self.vae_decoder_path = os.path.join(model_path, "vae")
        self.transformer = None
        self.vae_decoder = None
        self.teacache_probe = None
        self.teacache_probe_fused = False
        # host-refiner (cclog 86): the HV-1.5 token refiner self-attends over the sparse
        # mllm stream (<128 valid keys), which NaNs the Neuron flash kernel; run it on host
        # (eager, exact) and feed refined embeds into the NEFF instead.
        self._host_refiner = None
        self._host_refiner_on = (
            os.environ.get("DIFFLET_HUNYUAN15_HOST_REFINER") == "1"
            and str(kwargs.get("model_version", "1.0")) == "1.5"
        )
        # refined embeds depend only on (raw_mllm, timestep, mask); the prompt is fixed
        # within a generation and the timesteps repeat across the gate/baseline/teacache
        # trajectories, so memoize by timestep (keyed under a cheap prompt fingerprint).
        # Lazy (not eager-precompute-all): a TeaCache-skipped step never computes it.
        self._refiner_cache: dict[float, torch.Tensor] = {}
        self._refiner_fp = None
        self.pipeline = None
        self.text_seq_len = int(kwargs.get("text_seq_len", 256))
        self.text_seq_len_2 = int(kwargs.get("text_seq_len_2", 256))
        self.image_seq_len = int(kwargs.get("image_seq_len", 729))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.transformer_runtime = str(
            kwargs.get(
                "transformer_runtime",
                os.environ.get("DIFFLET_HUNYUAN15_TRANSFORMER_RUNTIME", "monolithic"),
            )
        )

        enable_transformer = bool(kwargs.get("enable_transformer", True))
        enable_vae_decoder = bool(kwargs.get("enable_vae_decoder", False))
        transformer_config_path = os.path.join(self.transformer_path, "config.json")
        self.transformer_config = None
        if os.path.exists(transformer_config_path):
            self.transformer_config = _load_diffusers_config(self.transformer_path)
        if self.model_version == "1.5" and enable_transformer and os.path.exists(transformer_config_path):
            from difflet.backends.trainium.hunyuan_video.backbone15 import (
                NeuronHunyuanVideo15BackboneApplication,
            )

            config = create_hunyuan_video15_backbone_config(
                transformer_path=self.transformer_path,
                world_size=parallel.tp_degree,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=self.shape["height"],
                width=self.shape["width"],
                num_frames=self.shape["num_frames"],
                text_seq_len=int(kwargs.get("text_seq_len", 1000)),
                text_seq_len_2=self.text_seq_len_2,
                image_seq_len=self.image_seq_len,
                batch_size=self.batch_size,
            )
            teacache_fused_15 = bool(kwargs.get("teacache_fused", False))
            if self.transformer_runtime == "segmented":
                if teacache_fused_15:
                    raise NotImplementedError(
                        "HunyuanVideo 1.5 TeaCache is not supported with the segmented "
                        "runtime: per-block process loading has no single graph for the "
                        "persistent prev_mod Parameter. Use transformer_runtime='monolithic'."
                    )
                from difflet.backends.trainium.hunyuan_video.segmented15 import (
                    DEFAULT_ATTENTION_COMPILER_ARGS,
                    DEFAULT_BLOCK_COMPILER_ARGS,
                    HunyuanVideo15SegmentedTransformerApplication,
                )

                self.transformer = HunyuanVideo15SegmentedTransformerApplication(
                    model_path=self.transformer_path,
                    config=config,
                    query_tile_size=int(kwargs.get("segmented_query_tile_size", 2051)),
                    key_tile_size=int(kwargs.get("segmented_key_tile_size", 2051)),
                    block_load_mode=str(kwargs.get("segmented_block_load_mode", "all")),
                    block_compiler_args=str(
                        kwargs.get("segmented_block_compiler_args", DEFAULT_BLOCK_COMPILER_ARGS)
                    ),
                    attention_compiler_args=str(
                        kwargs.get(
                            "segmented_attention_compiler_args",
                            DEFAULT_ATTENTION_COMPILER_ARGS,
                        )
                    ),
                )
            elif self.transformer_runtime == "monolithic":
                self.transformer = NeuronHunyuanVideo15BackboneApplication(
                    model_path=self.transformer_path,
                    config=config,
                )
                if teacache_fused_15:
                    # cclog 86: HV-1.5 block-0 modulation is timestep-only
                    # (Qwen-weak); the probe is mounted to MEASURE the signal.
                    from difflet.backends.trainium.hunyuan_video.teacache_probe15 import (
                        NeuronHunyuanVideo15TeacacheProbeFusedApplication,
                    )

                    self.teacache_probe = NeuronHunyuanVideo15TeacacheProbeFusedApplication(
                        model_path=self.transformer_path,
                        config=config,
                    )
                    self.teacache_probe_fused = True
            else:
                raise ValueError(
                    "HunyuanVideo 1.5 transformer_runtime must be 'monolithic' or "
                    f"'segmented', got {self.transformer_runtime!r}."
                )
        elif enable_transformer and os.path.exists(transformer_config_path):
            from difflet.backends.trainium.hunyuan_video.backbone import (
                NeuronHunyuanVideoBackboneApplication,
            )

            config = create_hunyuan_video_backbone_config(
                model_path=model_path,
                world_size=parallel.world_size,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=self.shape["height"],
                width=self.shape["width"],
                num_frames=self.shape["num_frames"],
                text_seq_len=self.text_seq_len,
                batch_size=self.batch_size,
                context_parallel_enabled=parallel.cp_degree > 1,
                cp_mode=parallel.cp_mode,
                sp_enabled=bool(getattr(parallel, "sp_enabled", False)),
                compile_shapes=self.compile_shapes,
            )
            self.transformer = NeuronHunyuanVideoBackboneApplication(
                model_path=self.transformer_path,
                config=config,
            )

            # The probe is a sibling sub-app with its own NEFF (compiled and
            # loaded as the "teacache_probe" component). Only the adaptive
            # TeaCache modes need it; callers that never run them (the CLI's
            # plain / probe-free runs) opt out so no probe NEFF is compiled or
            # loaded. Default True keeps the Python API unchanged.
            teacache_fused = bool(kwargs.get("teacache_fused", False))
            enable_teacache_probe = bool(kwargs.get("enable_teacache_probe", True))
            if not enable_teacache_probe:
                pass
            elif teacache_fused:
                from difflet.backends.trainium.hunyuan_video.teacache_probe import (
                    NeuronHunyuanVideoTeacacheProbeFusedApplication,
                )

                self.teacache_probe = NeuronHunyuanVideoTeacacheProbeFusedApplication(
                    model_path=self.transformer_path,
                    config=config,
                )
                self.teacache_probe_fused = True
            else:
                from difflet.backends.trainium.hunyuan_video.teacache_probe import (
                    NeuronHunyuanVideoTeacacheProbeApplication,
                )

                self.teacache_probe = NeuronHunyuanVideoTeacacheProbeApplication(
                    model_path=self.transformer_path,
                    config=config,
                )
                self.teacache_probe_fused = False

        vae_config_path = os.path.join(self.vae_decoder_path, "config.json")
        if enable_vae_decoder and os.path.exists(vae_config_path):
            if self.model_version == "1.5":
                from difflet.backends.trainium.hunyuan_video.vae15 import (
                    NeuronHunyuanVideo15VAEDecoderApplication,
                )

                vae_config = create_hunyuan_video15_vae_decoder_config(
                    model_path=model_path,
                    world_size=1,
                    tp_degree=1,
                    dtype=self.dtype,
                    height=self.shape["height"],
                    width=self.shape["width"],
                    num_frames=self.shape["num_frames"],
                    batch_size=self.batch_size,
                    tile_sample_min_height=int(kwargs.get("vae_tile_sample_min_height", 256)),
                    tile_sample_min_width=int(kwargs.get("vae_tile_sample_min_width", 256)),
                    tile_overlap_factor=float(kwargs.get("vae_tile_overlap_factor", 0.25)),
                )
                self.vae_decoder = NeuronHunyuanVideo15VAEDecoderApplication(
                    model_path=self.vae_decoder_path,
                    config=vae_config,
                )
            else:
                from difflet.backends.trainium.hunyuan_video.vae import (
                    NeuronHunyuanVideoVAEDecoderApplication,
                )

                # world_size must be the PROCESS world (tp*cp*cfg), not tp. The
                # VAE is not tensor-parallel (its convolutions are not TP-aware),
                # so tp_degree stays 1 and it is replicated across the ranks of
                # the process communicator the DiT establishes. One NxD process
                # has exactly one world: a VAE claiming a smaller world in the
                # same process segfaults the Neuron runtime at weight init (on
                # device, tp2cp2 ulysses 2026-08-30: DiT w4 + VAE w2 -> SIGSEGV),
                # and a world-1 VAE co-resident with world-4 components was
                # rejected by three runtime experiments on Qwen (see
                # docs/design/qwen_trn2_topology/03_adaptation_assessment.md;
                # the validated one-world/mixed-TP topology is in
                # 05_flux_runtime_validation.md). world_check.py enforces this
                # at load time. HunyuanVideo's DiT and VAE share the generate
                # stage process, unlike Wan/Qwen whose VAE has its own 1-core
                # stage — that is why this VAE cannot be world_size=1.
                vae_config = create_hunyuan_video_vae_decoder_config(
                    model_path=model_path,
                    world_size=parallel.world_size,
                    tp_degree=1,
                    dtype=self.dtype,
                    height=self.shape["height"],
                    width=self.shape["width"],
                    num_frames=self.shape["num_frames"],
                    batch_size=self.batch_size,
                    compile_shapes=self.compile_shapes,
                )
                self.vae_decoder = NeuronHunyuanVideoVAEDecoderApplication(
                    model_path=self.vae_decoder_path,
                    config=vae_config,
                )
        from difflet.models.hunyuan_video.pipeline import HunyuanVideoOrchestrator

        self.pipeline = HunyuanVideoOrchestrator(
            model_path=model_path,
            transformer=self if self.transformer is not None else None,
            vae=self.vae_decoder,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            teacache_speedup=kwargs.get("teacache_speedup"),
            teacache_calibration_path=kwargs.get("teacache_calibration_path"),
            # Probe-free modes: runtime-only, never part of the artifact identity.
            teacache_cadence=kwargs.get("teacache_cadence"),
            teacache_online_delta_alpha=kwargs.get("teacache_online_delta_alpha"),
        )

    def components(self) -> list[ComponentSpec]:
        components: list[ComponentSpec] = []
        if self.transformer is not None:
            if hasattr(self.transformer, "component_specs"):
                components.extend(self.transformer.component_specs(prefix="transformer"))
            else:
                components.append(ComponentSpec("transformer", self.transformer))
        if self.teacache_probe is not None:
            components.append(ComponentSpec("teacache_probe", self.teacache_probe))
        if self.vae_decoder is not None:
            if hasattr(self.vae_decoder, "component_specs"):
                components.extend(self.vae_decoder.component_specs(prefix="vae_decoder"))
            else:
                components.append(ComponentSpec("vae_decoder", self.vae_decoder))
        return components

    def load(
        self,
        compiled_model_path: str,
        start_rank_id: int | None = None,
        local_ranks_size: int | None = None,
        skip_warmup: bool = False,
        select=None,
    ) -> None:
        if (
            self.model_version == "1.5"
            and self.transformer is not None
            and getattr(self.transformer, "block_load_mode", None) == "process"
        ):
            self.transformer.set_compiled_model_path(str(compiled_model_path))
            if self.vae_decoder is not None:
                self.vae_decoder.load(
                    os.path.join(str(compiled_model_path), "vae_decoder"),
                    start_rank_id=0 if start_rank_id is not None else None,
                    local_ranks_size=1,
                    skip_warmup=skip_warmup,
                )
            return
        return super().load(
            compiled_model_path,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
            skip_warmup=skip_warmup,
            select=select,
        )

    def no_components_message(self, action: str) -> str:
        if self.model_version == "1.5":
            return (
                "HunyuanVideo 1.5 compile/load requires a transformer variant "
                "config under the selected transformer_subfolder, for example "
                "'transformer' for community Diffusers repos or 'transformer/480p_t2v' "
                "for the Tencent original layout."
            )
        if action == "compile":
            return (
                "HunyuanVideo compile requires transformer/config.json. "
                "The current app has no active compile component."
            )
        if action == "load":
            return "HunyuanVideo load requires compiled component artifacts"
        return super().no_components_message(action)

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        if self.transformer is None:
            raise NotImplementedError("HunyuanVideo DiT contract requires an active transformer.")
        config = self.transformer.config
        batch_size = int(getattr(config.neuron_config, "batch_size", 1))
        if self.model_version == "1.5":
            text_seq_len = int(getattr(config, "text_seq_len", 1000))
            text_seq_len_2 = int(getattr(config, "text_seq_len_2", 256))
            image_seq_len = int(getattr(config, "image_seq_len", 729))
            return {
                "hidden_states": {
                    "shape": (
                        batch_size,
                        int(config.in_channels),
                        int(config.latent_frames),
                        int(config.latent_height),
                        int(config.latent_width),
                    ),
                    "dtype": self.dtype,
                },
                "timestep": {"shape": (batch_size,), "dtype": self.dtype},
                "encoder_hidden_states": {
                    "shape": (batch_size, text_seq_len, int(config.text_embed_dim)),
                    "dtype": self.dtype,
                },
                "encoder_attention_mask": {
                    "shape": (batch_size, text_seq_len),
                    "dtype": torch.int64,
                },
                "timestep_r": {"shape": (batch_size,), "dtype": self.dtype},
                "encoder_hidden_states_2": {
                    "shape": (batch_size, text_seq_len_2, int(config.text_embed_2_dim)),
                    "dtype": self.dtype,
                },
                "encoder_attention_mask_2": {
                    "shape": (batch_size, text_seq_len_2),
                    "dtype": torch.int64,
                },
                "image_embeds": {
                    "shape": (batch_size, image_seq_len, int(config.image_embed_dim)),
                    "dtype": self.dtype,
                },
            }
        text_seq_len = int(getattr(config, "text_seq_len", 256))
        return {
            "hidden_states": {
                "shape": (
                    batch_size,
                    int(config.in_channels),
                    int(config.latent_frames),
                    int(config.latent_height),
                    int(config.latent_width),
                ),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch_size,), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (batch_size, text_seq_len, int(config.text_embed_dim)),
                "dtype": self.dtype,
            },
            "encoder_attention_mask": {
                "shape": (batch_size, text_seq_len),
                "dtype": torch.int64,
            },
            "pooled_projections": {
                "shape": (batch_size, int(config.pooled_projection_dim)),
                "dtype": self.dtype,
            },
            "guidance": {"shape": (batch_size,), "dtype": self.dtype},
        }

    def _get_host_refiner(self):
        """Build (once) a CPU fp32 copy of the HV-1.5 token refiner (context_embedder).

        Runs on host to avoid the Neuron flash <128-valid-keys NaN (cclog 86). Loads the
        ``context_embedder.*`` weights from the transformer checkpoint; uses a key-only
        attention mask (no all-(-inf) query row -> finite on CPU; valid-row output is
        identical to the symmetric mask, and padding-query rows are masked/zeroed
        downstream in the NEFF).
        """
        if self._host_refiner is not None:
            return self._host_refiner
        import glob
        import types

        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15TokenRefiner,
        )
        from safetensors.torch import load_file

        cfg = self.transformer.config
        refiner = HunyuanVideo15TokenRefiner(
            in_channels=int(cfg.text_embed_dim),
            num_attention_heads=int(cfg.num_attention_heads),
            attention_head_dim=int(cfg.attention_head_dim),
            num_layers=int(cfg.num_refiner_layers),
            mlp_ratio=float(getattr(cfg, "mlp_ratio", 4.0)),
        )
        prefix = "context_embedder."
        state: dict[str, torch.Tensor] = {}
        for f in sorted(glob.glob(os.path.join(self.transformer_path, "*.safetensors"))):
            for k, v in load_file(f).items():
                if k.startswith(prefix):
                    state[k[len(prefix):]] = v
        missing, unexpected = refiner.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"host refiner weight mismatch: missing={missing[:4]} unexpected={unexpected[:4]}"
            )
        refiner = refiner.to(torch.float32).eval()

        def _host_keyonly(rself, hs, temb, am=None):
            mask = None
            if am is not None:
                bs, seq = int(am.shape[0]), int(am.shape[1])
                neg = torch.finfo(hs.dtype).min
                mask = torch.zeros((bs, 1, 1, seq), dtype=hs.dtype).masked_fill(
                    ~am.bool().view(bs, 1, 1, seq), neg
                )
            for blk in rself.refiner_blocks:
                hs = blk(hs, temb, mask)
            return hs

        refiner.token_refiner.forward = types.MethodType(_host_keyonly, refiner.token_refiner)
        self._host_refiner = refiner
        return refiner

    def forward_dit(self, bundle: HunyuanVideoDiTInputBundle):
        if self.transformer is None:
            raise NotImplementedError("HunyuanVideo forward_dit requires an active transformer.")
        if isinstance(bundle, HunyuanVideo15DiTInputBundle):
            validate_hunyuan_video15_dit_inputs(
                bundle,
                config=self.transformer.config,
                dtype=self.dtype,
            )
            inputs = list(bundle.as_model_inputs())
            if self._host_refiner_on:
                # inputs = (hidden_states, timestep, encoder_hidden_states[mllm],
                #           encoder_attention_mask, timestep_r, ...) — refine the raw mllm
                # on host (fp32) and feed the inner_dim result into the NEFF.
                raw, ts, mask = inputs[2], inputs[1], inputs[3]
                # cheap prompt fingerprint (one channel across tokens + valid count); reset
                # the per-timestep cache when the prompt changes.
                fp = (tuple(raw.shape), round(float(raw[0, :, 0].float().sum()), 3), int(mask.sum()))
                if fp != self._refiner_fp:
                    self._refiner_cache = {}
                    self._refiner_fp = fp
                tkey = round(float(ts.flatten()[0]), 6)
                refined = self._refiner_cache.get(tkey)
                if refined is None:
                    refiner = self._get_host_refiner()
                    with torch.no_grad():
                        refined = refiner(raw.to(torch.float32), ts.to(torch.float32), mask).to(self.dtype)
                    self._refiner_cache[tkey] = refined
                inputs[2] = refined
            return self.transformer(*inputs)
        validate_hunyuan_video_dit_inputs(
            bundle,
            config=self.transformer.config,
            dtype=self.dtype,
        )
        return self.transformer(*bundle.as_model_inputs())

    def teacache_mod_input(self, bundle: HunyuanVideoDiTInputBundle) -> torch.Tensor:
        """Calibration entry — returns ``mod_input`` only.

        For Trainium backbone this dispatches the standalone probe NEFF
        (``NeuronHunyuanVideoTeacacheProbeApplication``) which is a separate
        compiled artifact from the DiT NEFF. For CPU model paths the wrapped
        ``HunyuanVideoTransformer3DModel.teacache_mod_input`` is called directly.
        """
        if self.teacache_probe is not None:
            return self.teacache_probe.teacache_mod_input(*bundle.as_model_inputs())
        if self.transformer is None:
            raise NotImplementedError("HunyuanVideo TeaCache requires an active transformer.")
        hook = getattr(self.transformer, "teacache_mod_input", None)
        if hook is None:
            raise NotImplementedError(
                "The active HunyuanVideo transformer does not expose teacache_mod_input."
            )
        return hook(*bundle.as_model_inputs())

    def teacache_delta(self, bundle: HunyuanVideoDiTInputBundle) -> torch.Tensor:
        """fused-A entry (cclog 80): returns ONLY the scalar delta. prev_mod is
        a persistent on-device Parameter updated in place via alias — no host
        handle. Requires the fused probe (teacache_fused=True)."""
        if not self.teacache_probe_fused or self.teacache_probe is None:
            raise NotImplementedError(
                "teacache_delta requires the fused probe (teacache_fused=True)."
            )
        return self.teacache_probe.teacache_delta(*bundle.as_model_inputs())

    def teacache_mod_input_with_delta(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
        prev_mod_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """T1 production probe entry: returns ``(delta_scalar, mod_input_handle)``.

        Dispatches the standalone probe NEFF. ``prev_mod_input`` is a device
        tensor kept on HBM across denoise steps (Python reference swap); only
        the 4-byte ``delta_scalar`` crosses PCIe per step.
        """
        if self.teacache_probe is None:
            raise NotImplementedError(
                "HunyuanVideo TeaCache probe NEFF is not loaded. The active "
                "transformer path may be CPU or HV-1.5 segmented — see cclog 72/73."
            )
        return self.teacache_probe.teacache_mod_input_with_delta(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
            prev_mod_input,
        )

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and isinstance(args[0], (HunyuanVideoDiTInputBundle, HunyuanVideo15DiTInputBundle)):
            return self.forward_dit(args[0])
        if self.model_version == "1.5":
            direct_keys_15 = {
                "hidden_states",
                "timestep",
                "encoder_hidden_states",
                "encoder_attention_mask",
                "timestep_r",
                "encoder_hidden_states_2",
                "encoder_attention_mask_2",
                "image_embeds",
            }
            if not args and direct_keys_15.issubset(kwargs):
                bundle = HunyuanVideo15DiTInputBundle(**kwargs)
                return self.forward_dit(bundle)
        direct_keys = {
            "hidden_states",
            "timestep",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "pooled_projections",
            "guidance",
        }
        if not args and direct_keys.issubset(kwargs):
            bundle = HunyuanVideoDiTInputBundle(**kwargs)
            return self.forward_dit(bundle)
        if self.transformer is not None and args:
            return self.transformer(*args, **kwargs)
        if self.pipeline is not None and self.pipeline.has_runtime_components():
            return self.pipeline(*args, **kwargs)
        del args, kwargs
        raise NotImplementedError("HunyuanVideo end-to-end inference is not implemented yet")
