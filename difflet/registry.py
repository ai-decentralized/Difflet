"""Model registry for DiffletPipeline."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

from difflet.pipeline.parallel_config import DiffletParallelConfig

ApplicationFactory = Callable[..., Any]
Detector = Callable[[str], bool]

_REGISTRY: dict[str, "ModelEntry"] = {}
_BUILTINS_LOADED = False


@dataclass(frozen=True)
class ModelEntry:
    name: str
    application_factory: ApplicationFactory | str
    hf_paths: tuple[str, ...] = ()
    detector: Detector | None = None
    default_parallel: DiffletParallelConfig = field(default_factory=DiffletParallelConfig)
    default_shape: dict[str, int | None] = field(default_factory=dict)
    backends: tuple[str, ...] = ("trainium",)
    # Per-model HF download allow-list. ``None`` (the default) lets
    # ``resolve_model_path`` use ``DEFAULT_DIFFUSERS_PATTERNS``. Override only
    # when a model needs files outside the standard diffusers layout (e.g. a
    # model that ships custom code in a non-standard directory).
    download_patterns: tuple[str, ...] | None = None

    def matches(self, model_id: str) -> bool:
        normalized = model_id.rstrip("/")
        if normalized in self.hf_paths:
            return True
        if self.detector is not None:
            return bool(self.detector(model_id))
        return False

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

    def create_application(
        self,
        *,
        model_path: str,
        parallel: DiffletParallelConfig,
        dtype: Any,
        shape: dict[str, int | None],
        backend: str,
        application_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        factory = _resolve_factory(self.application_factory)
        return factory(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            backend=backend,
            **(application_kwargs or {}),
        )

    def require_backend(self, backend: str) -> None:
        if backend not in self.backends:
            supported = ", ".join(self.backends)
            raise ValueError(
                f"model {self.name!r} does not support backend {backend!r}; "
                f"supported backends: {supported}"
            )


def register_model(
    *,
    name: str,
    application_factory: ApplicationFactory | str | None = None,
    hf_paths: list[str] | tuple[str, ...] = (),
    detector: Detector | None = None,
    default_parallel: DiffletParallelConfig | None = None,
    default_shape: dict[str, int | None] | None = None,
    backends: list[str] | tuple[str, ...] = ("trainium",),
    download_patterns: list[str] | tuple[str, ...] | None = None,
) -> Callable[[type], type]:
    """Register a model entry.

    Can be used as a decorator, or called with a dummy class for metadata-only
    entries whose application factory is a lazy import string.
    """

    def decorator(cls: type) -> type:
        factory = application_factory or getattr(cls, "application_factory", None)
        if factory is None:
            raise ValueError(f"model {name!r} is missing an application_factory")
        _REGISTRY[name] = ModelEntry(
            name=name,
            application_factory=factory,
            hf_paths=tuple(hf_paths),
            detector=detector,
            default_parallel=default_parallel or DiffletParallelConfig(),
            default_shape=default_shape or {},
            backends=tuple(backends),
            download_patterns=tuple(download_patterns) if download_patterns is not None else None,
        )
        return cls

    return decorator


def resolve_model(model_id: str, model_type: str | None = None) -> ModelEntry:
    _ensure_builtin_models_registered()
    if model_type is not None:
        try:
            return _REGISTRY[model_type]
        except KeyError as exc:
            raise ValueError(f"unknown model_type {model_type!r}") from exc

    matches = [entry for entry in _REGISTRY.values() if entry.matches(model_id)]
    if not matches:
        names = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(f"could not resolve model type for {model_id!r}; registered: {names}")
    if len(matches) > 1:
        names = ", ".join(entry.name for entry in matches)
        raise ValueError(f"model id {model_id!r} matched multiple entries: {names}")
    return matches[0]


def registered_models() -> tuple[ModelEntry, ...]:
    _ensure_builtin_models_registered()
    return tuple(_REGISTRY.values())


def _resolve_factory(factory: ApplicationFactory | str) -> ApplicationFactory:
    if callable(factory):
        return factory
    module_name, sep, attr_name = factory.partition(":")
    if sep != ":" or not module_name or not attr_name:
        raise ValueError(f"invalid application factory reference {factory!r}")
    module = importlib.import_module(module_name)
    resolved = getattr(module, attr_name)
    if not callable(resolved):
        raise TypeError(f"application factory {factory!r} is not callable")
    return resolved


def _ensure_builtin_models_registered() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    _register_builtin_flux()
    _register_builtin_wan()
    _register_builtin_hunyuan_video_15()
    _register_builtin_hunyuan_video()
    _register_builtin_qwen_image()
    _register_builtin_ltx_2()


def _register_builtin_flux() -> None:
    def is_flux(model_id: str) -> bool:
        value = model_id.lower()
        return "flux" in value or "black-forest-labs/flux" in value

    @register_model(
        name="flux",
        application_factory="difflet.models.flux.entry:create_flux_application",
        hf_paths=(
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
        ),
        detector=is_flux,
        default_parallel=DiffletParallelConfig(tp_degree=8),
        default_shape={"height": 1024, "width": 1024, "num_frames": None},
    )
    class _FluxRegistration:
        pass


def _register_builtin_wan() -> None:
    def is_wan(model_id: str) -> bool:
        value = model_id.lower()
        return "wan" in value or "wan-ai/" in value

    @register_model(
        name="wan",
        application_factory="difflet.models.wan.entry:create_wan_application",
        hf_paths=(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        ),
        detector=is_wan,
        default_parallel=DiffletParallelConfig(tp_degree=4),
        default_shape={"height": 480, "width": 832, "num_frames": 9},
        backends=("trainium",),
    )
    class _WanRegistration:
        pass


def _register_builtin_hunyuan_video() -> None:
    def is_hunyuan_video(model_id: str) -> bool:
        value = model_id.lower()
        return (
            ("hunyuanvideo" in value or "hunyuan-video" in value or "hunyuan_video" in value)
            and not _is_hunyuan_video_15(value)
        )

    @register_model(
        name="hunyuan_video",
        application_factory="difflet.models.hunyuan_video.entry:create_hunyuan_video_application",
        hf_paths=(
            "hunyuanvideo-community/HunyuanVideo",
            "tencent/HunyuanVideo",
        ),
        detector=is_hunyuan_video,
        default_parallel=DiffletParallelConfig(tp_degree=4),
        default_shape={"height": 320, "width": 512, "num_frames": 61},
        backends=("trainium",),
    )
    class _HunyuanVideoRegistration:
        pass


def _register_builtin_hunyuan_video_15() -> None:
    def is_hunyuan_video_15(model_id: str) -> bool:
        return _is_hunyuan_video_15(model_id.lower())

    @register_model(
        name="hunyuan_video_15",
        application_factory="difflet.models.hunyuan_video.entry:create_hunyuan_video15_application",
        hf_paths=(
            "tencent/HunyuanVideo-1.5",
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        ),
        detector=is_hunyuan_video_15,
        default_parallel=DiffletParallelConfig(tp_degree=4),
        default_shape={"height": 480, "width": 848, "num_frames": 121},
        backends=("trainium",),
    )
    class _HunyuanVideo15Registration:
        pass


def _is_hunyuan_video_15(value: str) -> bool:
    return any(
        marker in value
        for marker in (
            "hunyuanvideo-1.5",
            "hunyuanvideo_1.5",
            "hunyuanvideo1.5",
            "hunyuanvideo15",
            "hunyuan-video-1.5",
            "hunyuan_video_15",
        )
    )


def _register_builtin_qwen_image() -> None:
    def is_qwen_image(model_id: str) -> bool:
        value = model_id.lower()
        return "qwen-image" in value or "qwen/image" in value or "qwen_image" in value

    @register_model(
        name="qwen_image",
        application_factory="difflet.models.qwen_image.entry:create_qwen_image_application",
        hf_paths=(
            "Qwen/Qwen-Image",
        ),
        detector=is_qwen_image,
        default_parallel=DiffletParallelConfig(tp_degree=4),
        default_shape={"height": 1024, "width": 1024, "num_frames": None},
        backends=("trainium",),
    )
    class _QwenImageRegistration:
        pass


def _register_builtin_ltx_2() -> None:
    def is_ltx_2(model_id: str) -> bool:
        value = model_id.lower()
        return (
            "ltx-2" in value
            or "ltx2" in value
            or "ltx_2" in value
            or "lightricks/ltx" in value
        )

    @register_model(
        name="ltx_2",
        application_factory="difflet.models.ltx_2.entry:create_ltx_2_application",
        hf_paths=(
            "Lightricks/LTX-2",
        ),
        detector=is_ltx_2,
        default_parallel=DiffletParallelConfig(tp_degree=4),
        default_shape={"height": 512, "width": 768, "num_frames": 121},
        backends=("trainium",),
        download_patterns=(
            "*.json",
            "*.txt",
            "*.md",
            "LICENSE",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model*.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model*.safetensors",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/*",
            "scheduler/scheduler_config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "audio_vae/config.json",
            "audio_vae/diffusion_pytorch_model.safetensors",
            "vocoder/config.json",
            "vocoder/diffusion_pytorch_model.safetensors",
            "connectors/config.json",
            "connectors/diffusion_pytorch_model.safetensors",
        ),
    )
    class _LTX2Registration:
        pass
