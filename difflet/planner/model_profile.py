"""Normalized per-model geometry: heads, layers, and derived sequence lengths.

The cost model needs the quantities that set communication volume -- token count,
head count, hidden size, block count -- and each model's ``InferenceConfig``
spells them differently (``image_seq_len`` on Qwen, ``video_seq_len`` on LTX-2,
inline ``num_patches`` arithmetic on Flux). This module is the one place that
translates.

Dimensions are a static table rather than a read of ``transformer/config.json``
so ``difflet plan`` works before anything is downloaded -- planning a
configuration is exactly what you want to do *before* pulling 30 GB of weights.
The table is verified against the shipped configs by
``tests/unit/planner/test_model_profile.py``, which reads the real files when
they happen to be present and skips when they are not.
"""

from __future__ import annotations

from dataclasses import dataclass

from difflet.registry import ModelCapabilities, resolve_model


@dataclass(frozen=True)
class ModelDimensions:
    """Backbone geometry, in the terms the cost model uses.

    ``patch_size`` and the VAE compression ratios describe how a pixel-space
    shape becomes a token count. They are the *effective* values for that
    reduction, which is not always what ``config.json`` reports: Flux's config
    says ``patch_size: 1`` because its 2x2 latent packing happens in the
    pipeline rather than the transformer, so the effective patch size here is 2.
    """

    num_attention_heads: int
    attention_head_dim: int
    num_layers: int
    num_single_layers: int = 0
    text_seq_len: int = 0
    vae_spatial_compression: int = 8
    vae_temporal_compression: int = 1  # 1 marks an image model
    patch_size: int = 2
    patch_size_t: int = 1
    # Set when the geometry has not been checked against a real run. Only
    # HunyuanVideo 1.5 today, which is a download-only scaffold.
    unverified: bool = False

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def total_blocks(self) -> int:
        return self.num_layers + self.num_single_layers


@dataclass(frozen=True)
class SequenceLengths:
    image: int
    text: int

    @property
    def joint(self) -> int:
        """MMDiT attends over the concatenated [text || image] sequence."""

        return self.image + self.text


# Read from each model's shipped transformer/config.json; see the test module.
DIMENSIONS: dict[str, ModelDimensions] = {
    "flux": ModelDimensions(
        num_attention_heads=24, attention_head_dim=128,
        num_layers=19, num_single_layers=38,
        text_seq_len=512,
        vae_spatial_compression=8, patch_size=2,  # 2x2 packing lives in the pipeline
    ),
    "qwen_image": ModelDimensions(
        num_attention_heads=24, attention_head_dim=128,
        num_layers=60,
        text_seq_len=1024,
        vae_spatial_compression=8, patch_size=2,
    ),
    "wan": ModelDimensions(
        num_attention_heads=40, attention_head_dim=128,
        num_layers=40,
        text_seq_len=512,
        vae_spatial_compression=8, vae_temporal_compression=4, patch_size=2,
    ),
    "hunyuan_video": ModelDimensions(
        num_attention_heads=24, attention_head_dim=128,
        num_layers=20, num_single_layers=40,
        text_seq_len=256,
        vae_spatial_compression=8, vae_temporal_compression=4, patch_size=2,
    ),
    "hunyuan_video_15": ModelDimensions(
        num_attention_heads=16, attention_head_dim=128,
        num_layers=54,
        text_seq_len=256,
        vae_spatial_compression=16, vae_temporal_compression=4, patch_size=1,
        unverified=True,  # scaffold: the geometry has never been exercised
    ),
    "ltx_2": ModelDimensions(
        num_attention_heads=32, attention_head_dim=128,
        num_layers=48,
        text_seq_len=1024,
        vae_spatial_compression=32, vae_temporal_compression=8, patch_size=1,
    ),
}


@dataclass(frozen=True)
class WeightFootprint:
    """bf16 weight bytes, from a header-only safetensors scan of the real checkpoints.

    Used to estimate device HBM pressure. **The estimate is advisory, not a
    feasibility rule**, because the residency semantics are not settled: staged
    models (Wan, Qwen, HunyuanVideo) run their components as sequential
    subprocesses so the components are never all resident, and Wan 2.2's
    ``transformer``/``transformer_2`` pair is a timestep-boundary switch whose
    co-residency depends on per-stage enable flags.

    The naive reading -- ``dp * cfg * cp`` full copies of ``total_bytes`` against
    the device's HBM -- says Wan 2.2 ``tp2cfg`` cannot fit on a trn2.3xlarge,
    and ``scripts/verify_cli.py`` records that cell passing on device. So the
    naive reading is an over-estimate somewhere, and until a run measures peak
    HBM per configuration the planner reports the number and declines to reject
    on it. See the benchmark plan in the design doc.
    """

    total_bytes: int
    # True when the CLI runs this model's components as separate stage
    # subprocesses, so peak residency is one stage rather than the sum.
    staged: bool


# Measured with difflet/cli/dp/hbm_check.py::component_weight_bytes over the
# shipped checkpoints (bf16, header-only scan).
WEIGHTS: dict[str, WeightFootprint] = {
    "flux": WeightFootprint(total_bytes=33_700_000_000, staged=False),
    "qwen_image": WeightFootprint(total_bytes=57_700_000_000, staged=True),
    "wan": WeightFootprint(total_bytes=68_800_000_000, staged=True),
    "hunyuan_video": WeightFootprint(total_bytes=41_400_000_000, staged=True),
    "hunyuan_video_15": WeightFootprint(total_bytes=0, staged=True),  # scaffold, no weights
    "ltx_2": WeightFootprint(total_bytes=67_700_000_000, staged=False),
}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    capabilities: ModelCapabilities
    dims: ModelDimensions
    default_shape: dict[str, int | None]
    weights: WeightFootprint

    def sequence_lengths(
        self,
        *,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
    ) -> SequenceLengths:
        """Token counts for a pixel-space shape.

        One formula covers every model here: compress by the VAE ratios, then
        divide by the patch size. Image models have ``vae_temporal_compression``
        of 1 and no frame axis, which collapses the temporal term to 1.
        """

        shape = self.resolve_shape(height=height, width=width, num_frames=num_frames)
        resolved_height = shape["height"]
        resolved_width = shape["width"]
        if resolved_height is None or resolved_width is None:
            raise ValueError(f"model {self.name!r} has no resolved height/width")

        dims = self.dims
        latent_h = int(resolved_height) // dims.vae_spatial_compression
        latent_w = int(resolved_width) // dims.vae_spatial_compression
        frames = shape.get("num_frames")
        if frames is None or dims.vae_temporal_compression <= 1:
            latent_frames = 1
        else:
            latent_frames = (int(frames) - 1) // dims.vae_temporal_compression + 1

        image_tokens = (
            (latent_frames // dims.patch_size_t)
            * (latent_h // dims.patch_size)
            * (latent_w // dims.patch_size)
        )
        return SequenceLengths(image=max(image_tokens, 1), text=dims.text_seq_len)

    def resolve_shape(
        self,
        *,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
    ) -> dict[str, int | None]:
        shape = dict(self.default_shape)
        if height is not None:
            shape["height"] = height
        if width is not None:
            shape["width"] = width
        if num_frames is not None:
            shape["num_frames"] = num_frames
        return shape


def load_profile(model_id: str, *, model_type: str | None = None) -> ModelProfile:
    entry = resolve_model(model_id, model_type=model_type)
    try:
        dims = DIMENSIONS[entry.name]
    except KeyError as exc:
        raise ValueError(f"no planner dimensions registered for model {entry.name!r}") from exc
    capabilities = entry.require_capabilities()
    if capabilities.num_attention_heads != dims.num_attention_heads:
        raise ValueError(
            f"{entry.name}: registry declares {capabilities.num_attention_heads} heads "
            f"but planner dimensions say {dims.num_attention_heads}"
        )
    return ModelProfile(
        name=entry.name,
        model_id=model_id,
        capabilities=capabilities,
        dims=dims,
        default_shape=dict(entry.default_shape),
        weights=WEIGHTS.get(entry.name, WeightFootprint(total_bytes=0, staged=False)),
    )


def device_weight_bytes(profile: ModelProfile, parallel) -> int:
    """Upper-bound weight bytes resident across the whole device.

    Tensor parallelism shards one copy across its group; cp, cfg and dp each
    replicate that copy. So the device holds ``dp * cfg * cp`` copies regardless
    of ``tp``, which is why adding cp or dp on a fixed core count buys latency at
    the cost of memory. Reported, never enforced -- see :class:`WeightFootprint`.
    """

    copies = (
        parallel.dp_degree
        * (2 if parallel.cfg_parallel_enabled else 1)
        * parallel.cp_degree
    )
    return copies * profile.weights.total_bytes
