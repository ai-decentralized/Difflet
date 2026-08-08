# >>> NxDI fork banner — managed by scripts/add_fork_banner.py >>>
# Forked from neuronx-distributed-inference v0.9.17334+ced6ae4e
# Original path: neuronx_distributed_inference/models/diffusers/flux/application.py
# Fork date: 2026-05-08
# Modifications:
#   2026-05-08 _compile_component:
#     * check ``model.pt`` (the post-compile end product) instead of the
#       containing directory; ``neuron_config.json`` is written *during*
#       compile and would otherwise short-circuit the skip check
#     * insert ``torch.distributed.barrier()`` calls so all ranks enter and
#       leave the per-component compile in lockstep — without this, one rank
#       would early-exit while peers were still SPMD-tracing, dropping the
#       trace silently and leaving an empty cache subdir
#   2026-05-08 load:
#     * load each component in rank lockstep for the same reason; component
#       load performs Neuron runtime initialization and weight transfer, and
#       ranks must not race into the next component while peers still load the
#       current one
#   2026-05-08 create_flux_config:
#     * recompile a component if its saved neuron_config rank layout differs
#       from the current config, even when model.pt already exists
#   2026-05-08 _compile_component/load:
#     * in torchrun, single-core replicated components are compiled only by
#       rank 0 and then shared; all ranks still load them with local rank 0
#   2026-05-11 lifecycle:
#     * move shared race-safe component compile/load orchestration into
#       difflet.backends.trainium.core.multi_component_application
# <<< NxDI fork banner <<<
import inspect
import logging
import os
from typing import Any, Optional

import torch

from difflet.backends.trainium.core.config import InferenceConfig, NeuronConfig
from difflet.backends.trainium.core.multi_component_application import (
    ComponentSpec,
    MultiComponentApplication,
)
from difflet.models.flux.clip.modeling_clip import (
    CLIPInferenceConfig,
    NeuronClipApplication,
)
from difflet.backends.trainium.flux.teacache_probe_fused import (
    NeuronFluxTeacacheProbeFusedApplication,
)
from difflet.models.flux.modeling_flux import (
    FluxBackboneInferenceConfig,
    NeuronFluxBackboneApplication,
)
from difflet.models.flux.pipeline import NeuronFluxPipeline
from difflet.models.flux.vae.modeling_vae import (
    NeuronVAEDecoderApplication,
    VAEDecoderInferenceConfig,
)
from difflet.models.flux.t5.modeling_t5 import (
    NeuronT5Application,
    T5InferenceConfig,
)
from difflet.utils.diffusers_adapter import load_diffusers_config
from difflet.utils.hf_adapter import load_pretrained_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_flux_parallelism_config(
    backbone_tp_degree: int, cp_degree: int = 1, cfg_parallel_enabled: bool = False
) -> int:
    """
    Get the world_size based on backbone_tp_degree and parallelism settings.

    Args:
        backbone_tp_degree: The tensor parallelism degree for the backbone model
        cp_degree: Context parallelism degree (1 = disabled, default: 1)
        cfg_parallel_enabled: Whether CFG parallelism is enabled (default: False)

    Returns:
        int: world_size (backbone_tp_degree × cp_degree, or × 2 for CFG parallel)

    Note:
        cp_degree > 1 and cfg_parallel_enabled are mutually exclusive.
    """
    if cp_degree > 1 and cfg_parallel_enabled:
        raise ValueError(
            "cp_degree > 1 and cfg_parallel_enabled are mutually exclusive. "
            "Only one can be set at a time."
        )

    if cfg_parallel_enabled:
        return backbone_tp_degree * 2

    return backbone_tp_degree * cp_degree


def create_flux_config(
    model_path,
    world_size,
    backbone_tp_degree,
    dtype,
    height,
    width,
    inpaint=False,
    cfg_parallel_enabled=False,
    context_parallel_enabled=False,
    cp_mode="gather_kv",
    sp_enabled=False,
):
    text_encoder_path = os.path.join(model_path, "text_encoder")
    text_encoder_2_path = os.path.join(model_path, "text_encoder_2")
    backbone_path = os.path.join(model_path, "transformer")
    vae_decoder_path = os.path.join(model_path, "vae")

    clip_neuron_config = NeuronConfig(
        tp_degree=1,
        world_size=world_size,
        torch_dtype=dtype,
    )
    clip_config = CLIPInferenceConfig(
        neuron_config=clip_neuron_config,
        load_config=load_pretrained_config(text_encoder_path),
    )

    t5_neuron_config = NeuronConfig(
        tp_degree=world_size,  # T5: TP degree = world_size
        world_size=world_size,
        torch_dtype=dtype,
    )
    t5_config = T5InferenceConfig(
        neuron_config=t5_neuron_config,
        load_config=load_pretrained_config(text_encoder_2_path),
    )

    backbone_neuron_config = NeuronConfig(
        tp_degree=backbone_tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
    )
    backbone_config = FluxBackboneInferenceConfig(
        cfg_parallel_enabled=cfg_parallel_enabled,
        context_parallel_enabled=context_parallel_enabled,
        cp_mode=cp_mode,
        sp_enabled=sp_enabled,
        neuron_config=backbone_neuron_config,
        load_config=load_diffusers_config(backbone_path),
        height=height,
        width=width,
    )

    decoder_neuron_config = NeuronConfig(
        tp_degree=1,
        world_size=world_size,
        torch_dtype=dtype,
    )
    if inpaint:
        decoder_config = VAEDecoderInferenceConfig(
            neuron_config=decoder_neuron_config,
            load_config=load_diffusers_config(vae_decoder_path),
            height=height,
            width=width,
        )
    else:
        decoder_config = VAEDecoderInferenceConfig(
            neuron_config=decoder_neuron_config,
            load_config=load_diffusers_config(vae_decoder_path),
            height=height,
            width=width,
            transformer_in_channels=backbone_config.in_channels,
        )

    setattr(backbone_config, "vae_scale_factor", decoder_config.vae_scale_factor)

    return (clip_config, t5_config, backbone_config, decoder_config)


class NeuronFluxApplication(MultiComponentApplication):
    def __init__(
        self,
        model_path: str,
        text_encoder_config: InferenceConfig,
        text_encoder2_config: InferenceConfig,
        backbone_config: InferenceConfig,
        decoder_config: InferenceConfig,
        text_encoder_path: Optional[str] = None,
        text_encoder_2_path: Optional[str] = None,
        vae_decoder_path: Optional[str] = None,
        transformer_path: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        pipeline_class=NeuronFluxPipeline,
        teacache_fused: bool = False,
        teacache_speedup: Optional[float] = None,
        teacache_calibration=None,
        teacache_calibration_path: Optional[str] = None,
        teacache_cadence: Optional[int] = None,
        teacache_online_delta_alpha: Optional[float] = None,
        cache_plan_file: Optional[str] = None,
        cache_mask_file: Optional[str] = None,
        cache_predictor: Optional[str] = None,
        cache_predictor_order: Optional[int] = None,
        cache_predictor_coord: Optional[str] = None,
        cache_recovery_warmup_steps: Optional[int] = None,
        cache_recovery_cooldown_steps: Optional[int] = None,
        cache_recovery_max_consecutive: Optional[int] = None,
        cache_recovery_steps: Optional[int] = None,
        cache_require_final_anchor: Optional[bool] = None,
        cache_profile_file: Optional[str] = None,
        cache_profile_qualification_file: Optional[str] = None,
        cache_runtime_model_id: Optional[str] = None,
        cache_runtime_model_revision: Optional[str] = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.text_encoder_path = text_encoder_path or os.path.join(model_path, "text_encoder")
        self.text_encoder_2_path = text_encoder_2_path or os.path.join(model_path, "text_encoder_2")
        self.transformer_path = transformer_path or os.path.join(model_path, "transformer")
        self.vae_decoder_path = vae_decoder_path or os.path.join(model_path, "vae")

        self.height = height
        self.width = width
        self.max_sequence_length = 512
        self._cache_plan = None
        self._cache_mask = None
        self._cache_predictor_spec = None
        self._cache_recovery_config = None
        self._probe_free_recovery_config = None
        self._qualified_cache_profile = None
        self._cache_runtime_model_id = cache_runtime_model_id
        self._cache_runtime_model_revision = cache_runtime_model_revision
        self._teacache_cadence = teacache_cadence
        self._teacache_online_delta_alpha = teacache_online_delta_alpha

        plan_selected = cache_plan_file is not None
        mask_selected = cache_mask_file is not None
        qualified_profile_selected = cache_profile_file is not None
        if (cache_profile_qualification_file is not None) != qualified_profile_selected:
            raise ValueError(
                "cache_profile_file and cache_profile_qualification_file must be provided together"
            )
        predictor_options = {
            "cache_predictor": cache_predictor,
            "cache_predictor_order": cache_predictor_order,
            "cache_predictor_coord": cache_predictor_coord,
        }
        recovery_options = {
            "cache_recovery_warmup_steps": cache_recovery_warmup_steps,
            "cache_recovery_cooldown_steps": cache_recovery_cooldown_steps,
            "cache_recovery_max_consecutive": cache_recovery_max_consecutive,
            "cache_recovery_steps": cache_recovery_steps,
            "cache_require_final_anchor": cache_require_final_anchor,
        }
        recovery_selected = any(value is not None for value in recovery_options.values())
        probe_free_selected = (
            teacache_cadence is not None or teacache_online_delta_alpha is not None
        )
        old_cache_selected = any(
            value is not None
            for value in (
                teacache_speedup,
                teacache_calibration,
                teacache_calibration_path,
                teacache_cadence,
                teacache_online_delta_alpha,
            )
        ) or bool(teacache_fused)
        if qualified_profile_selected and (
            plan_selected
            or mask_selected
            or old_cache_selected
            or any(value is not None for value in predictor_options.values())
            or recovery_selected
        ):
            raise ValueError(
                "a qualified cache profile cannot be combined with legacy cache options"
            )
        if qualified_profile_selected and (
            not isinstance(cache_runtime_model_id, str)
            or not cache_runtime_model_id
            or not isinstance(cache_runtime_model_revision, str)
            or not cache_runtime_model_revision
        ):
            raise ValueError(
                "qualified cache profiles require a resolved runtime model id and revision"
            )
        legacy_modes = sum(
            (
                teacache_speedup is not None,
                teacache_cadence is not None,
                teacache_online_delta_alpha is not None,
            )
        )
        if legacy_modes > 1:
            raise ValueError(
                "teacache_speedup, teacache_cadence, and "
                "teacache_online_delta_alpha are mutually exclusive"
            )
        if teacache_cadence is not None and (
            isinstance(teacache_cadence, bool)
            or not isinstance(teacache_cadence, int)
            or teacache_cadence <= 0
        ):
            raise ValueError("teacache_cadence must be a positive integer")
        if teacache_online_delta_alpha is not None and float(teacache_online_delta_alpha) <= 0.0:
            raise ValueError("teacache_online_delta_alpha must be positive")
        if plan_selected and (
            mask_selected
            or any(value is not None for value in predictor_options.values())
            or recovery_selected
        ):
            raise ValueError(
                "cache_plan_file is a complete cache configuration and cannot be "
                "combined with cache_mask_file, predictor, or recovery overrides"
            )
        if plan_selected and old_cache_selected:
            raise ValueError(
                "cache_plan_file and legacy teacache configuration are mutually exclusive"
            )
        if mask_selected and old_cache_selected:
            raise ValueError(
                "cache_mask_file and legacy teacache configuration are mutually exclusive"
            )
        if not mask_selected and any(value is not None for value in predictor_options.values()):
            raise ValueError("cache predictor overrides require cache_mask_file")
        if recovery_selected and (
            teacache_speedup is not None
            or teacache_calibration is not None
            or teacache_calibration_path is not None
        ):
            raise ValueError("cache recovery overrides are not supported by calibrated TeaCache")
        if not mask_selected and recovery_selected and not probe_free_selected:
            raise ValueError(
                "cache recovery overrides require cache_mask_file, "
                "teacache_cadence, or teacache_online_delta_alpha"
            )
        if mask_selected and cache_predictor is None:
            raise ValueError("cache_mask_file requires cache_predictor")

        if qualified_profile_selected:
            from difflet.pipeline.cache import load_qualified_cache_profile

            if getattr(backbone_config, "cfg_parallel_enabled", False):
                raise ValueError("qualified Flux cache profiles require cfg_parallel_enabled=False")
            assert cache_profile_file is not None
            assert cache_profile_qualification_file is not None
            self._qualified_cache_profile = load_qualified_cache_profile(
                cache_profile_file,
                cache_profile_qualification_file,
            )

        if plan_selected or mask_selected or probe_free_selected:
            from difflet.pipeline.cache import (
                QualityRecoveryConfig,
                load_cache_mask,
                load_cache_plan,
            )

            if (plan_selected or mask_selected) and getattr(
                backbone_config, "cfg_parallel_enabled", False
            ):
                raise ValueError(
                    "Flux cache plans/masks require cfg_parallel_enabled=False; "
                    "one shared predictor history cannot represent split CFG branches"
                )
            if plan_selected:
                self._cache_plan = load_cache_plan(cache_plan_file)
            elif mask_selected:
                assert cache_mask_file is not None
                assert cache_predictor is not None
                if cache_predictor not in ("legacy_residual", "taylorseer"):
                    raise ValueError("cache_predictor must be 'legacy_residual' or 'taylorseer'")
                coord = cache_predictor_coord or "index"
                if cache_predictor == "legacy_residual":
                    if cache_predictor_order is not None:
                        raise ValueError("cache_predictor_order applies only to taylorseer")
                    self._cache_predictor_spec = {
                        "type": "legacy_residual",
                        "coord": coord,
                    }
                else:
                    self._cache_predictor_spec = {
                        "type": "taylorseer",
                        "order": 1 if cache_predictor_order is None else cache_predictor_order,
                        "coord": coord,
                    }
                self._cache_mask = load_cache_mask(cache_mask_file)
                self._cache_recovery_config = QualityRecoveryConfig(
                    warmup_steps=(
                        0 if cache_recovery_warmup_steps is None else cache_recovery_warmup_steps
                    ),
                    cooldown_steps=(
                        0
                        if cache_recovery_cooldown_steps is None
                        else cache_recovery_cooldown_steps
                    ),
                    require_final_anchor=bool(cache_require_final_anchor),
                    max_consecutive_predictions=cache_recovery_max_consecutive,
                    recovery_steps=(1 if cache_recovery_steps is None else cache_recovery_steps),
                )
            if probe_free_selected:
                from difflet.pipeline.teacache import (
                    DEFAULT_TEACACHE_COOLDOWN_STEPS,
                    DEFAULT_TEACACHE_WARMUP_STEPS,
                )

                self._probe_free_recovery_config = QualityRecoveryConfig(
                    warmup_steps=(
                        DEFAULT_TEACACHE_WARMUP_STEPS
                        if cache_recovery_warmup_steps is None
                        else cache_recovery_warmup_steps
                    ),
                    cooldown_steps=(
                        DEFAULT_TEACACHE_COOLDOWN_STEPS
                        if cache_recovery_cooldown_steps is None
                        else cache_recovery_cooldown_steps
                    ),
                    require_final_anchor=bool(cache_require_final_anchor),
                    max_consecutive_predictions=cache_recovery_max_consecutive,
                    recovery_steps=(1 if cache_recovery_steps is None else cache_recovery_steps),
                )

        # Neuron applications replace these modules immediately below. Passing
        # None prevents diffusers from loading large CPU weights that are never
        # used, while still loading tokenizers, scheduler, and the VAE shell.
        self.pipe = pipeline_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            text_encoder=None,
            text_encoder_2=None,
            transformer=None,
        )

        self.text_encoder_config = text_encoder_config
        self.text_encoder2_config = text_encoder2_config
        self.backbone_config = backbone_config
        self.decoder_config = decoder_config

        if self._qualified_cache_profile is not None:
            self._validate_qualified_cache_runtime(
                num_steps=int(self._qualified_cache_profile.generation["num_steps"]),
                guidance_scale=float(
                    self._qualified_cache_profile.generation["guidance_scale"]
                ),
            )

        self.pipe.text_encoder = NeuronClipApplication(
            model_path=self.text_encoder_path, config=self.text_encoder_config
        )
        self.pipe.text_encoder_2 = NeuronT5Application(
            model_path=self.text_encoder_2_path, config=self.text_encoder2_config
        )
        self.pipe.transformer = NeuronFluxBackboneApplication(
            model_path=self.transformer_path,
            config=self.backbone_config,
        )
        self.pipe.vae.decoder = NeuronVAEDecoderApplication(
            model_path=self.vae_decoder_path, config=self.decoder_config
        )

        # TeaCache fused-A (cclog 85). Mount the probe NEFF when teacache is
        # requested; build the controller when a calibration is provided.
        self.teacache_probe = None
        self.pipe.teacache_probe = None
        self.pipe.teacache_controller = None
        self.pipe.teacache_speedup = None
        enable_teacache = (
            teacache_fused
            or teacache_speedup is not None
            or teacache_cadence is not None
            or teacache_online_delta_alpha is not None
        )
        if enable_teacache:
            # CFG-parallel scatters [neg,pos] across DP ranks (modeling_flux.py:
            # 338-399), which a single probe/skip decision cannot represent — must
            # be off when teacache is on (cclog 85). Flux's default is off.
            if getattr(self.backbone_config, "cfg_parallel_enabled", False):
                raise ValueError(
                    "Flux TeaCache requires cfg_parallel_enabled=False (the probe + "
                    "single skip decision is unsound across CFG-parallel DP ranks). "
                    "Disable CFG parallelism when enabling teacache."
                )
            if teacache_fused or teacache_speedup is not None:
                self.teacache_probe = NeuronFluxTeacacheProbeFusedApplication(
                    model_path=self.transformer_path,
                    config=self.backbone_config,
                )
                self.pipe.teacache_probe = self.teacache_probe
            if teacache_speedup is not None:
                from difflet.pipeline.teacache import (
                    TeaCacheController,
                    load_teacache_calibration_or_raise,
                )

                shape_label = f"{int(self.height)}x{int(self.width)}"
                if teacache_calibration is not None and teacache_calibration_path is not None:
                    raise ValueError(
                        "teacache_calibration and teacache_calibration_path are mutually exclusive"
                    )
                calibration = teacache_calibration
                if calibration is None:
                    calibration = load_teacache_calibration_or_raise(
                        teacache_calibration_path,
                        model="flux",
                        shape_label=shape_label,
                    )
                elif calibration.model != "flux" or calibration.shape_label != shape_label:
                    raise ValueError("TeaCache calibration does not match Flux model/profile")
                if (
                    calibration.target_speedup is not None
                    and float(teacache_speedup) > float(calibration.target_speedup) + 1e-6
                ):
                    raise ValueError(
                        "Flux TeaCache calibration target speedup is lower than requested: "
                        f"requested {teacache_speedup}, calibration has "
                        f"{calibration.target_speedup}."
                    )
                self.pipe.teacache_controller = TeaCacheController(calibration)
                self.pipe.teacache_speedup = float(teacache_speedup)

    def components(self) -> list[ComponentSpec]:
        # Compile order follows the original Flux application. Load order is
        # fixed explicitly because the first loaded Trainium component
        # establishes the process-wide communicator.
        specs = [
            ComponentSpec("text_encoder", self.pipe.text_encoder, load_priority=2),
            ComponentSpec("text_encoder_2", self.pipe.text_encoder_2, load_priority=0),
            ComponentSpec("transformer", self.pipe.transformer, load_priority=1),
            ComponentSpec("decoder", self.pipe.vae.decoder, load_priority=3),
        ]
        if self.teacache_probe is not None:
            # Same world_size as the backbone; load after text_encoder_2 (which
            # fixes the process communicator), alongside the transformer.
            specs.append(ComponentSpec("teacache_probe", self.teacache_probe, load_priority=1))
        return specs

    def __call__(self, *args, **kwargs):
        """Run one generation through the configured FLUX pipeline.

        Composable cache modes create one fresh session and pass it explicitly
        into this call. No request state is stored on the application or shared
        pipeline. The calibrated legacy TeaCache path retains its historical
        controller slot until that separate compatibility mode is removed.
        """

        external_session = kwargs.get("cache_session")
        configured_cache = (
            getattr(self, "_qualified_cache_profile", None) is not None
            or self._cache_plan is not None
            or self._cache_mask is not None
            or self._teacache_cadence is not None
            or self._teacache_online_delta_alpha is not None
        )
        if external_session is not None and configured_cache:
            raise ValueError(
                "an explicit cache_session cannot be combined with application cache configuration"
            )
        cache_session = None
        if external_session is None and (
            getattr(self, "_qualified_cache_profile", None) is not None
            or self._cache_plan is not None
            or self._cache_mask is not None
        ):
            cache_session = self._prepare_cache_session(*args, **kwargs)
        elif external_session is None and (
            self._teacache_cadence is not None
            or self._teacache_online_delta_alpha is not None
        ):
            cache_session = self._prepare_probe_free_teacache(*args, **kwargs)
        if cache_session is not None:
            kwargs["cache_session"] = cache_session
        return self.pipe(*args, **kwargs)

    def _request_identity(self, *args: Any, **kwargs: Any) -> tuple[int, int, int]:
        signature = inspect.signature(self.pipe.__call__)
        bound = signature.bind_partial(*args, **kwargs)
        call = bound.arguments
        sigmas = call.get("sigmas")
        if sigmas is not None:
            num_steps = len(sigmas)
        else:
            parameter = signature.parameters.get("num_inference_steps")
            if parameter is None or parameter.default is inspect.Parameter.empty:
                raise TypeError("FLUX pipeline must declare a default for num_inference_steps")
            requested_steps = call.get("num_inference_steps", parameter.default)
            if isinstance(requested_steps, bool) or not isinstance(requested_steps, int):
                raise TypeError("num_inference_steps must be an integer")
            num_steps = requested_steps
        if num_steps <= 0:
            raise ValueError("num_inference_steps or sigmas must define at least one step")
        height = int(call.get("height") or self.height)
        width = int(call.get("width") or self.width)
        return num_steps, height, width

    def _validate_qualified_cache_runtime(
        self,
        *,
        num_steps: int,
        guidance_scale: float,
    ) -> None:
        from difflet.pipeline.cache import scheduler_config_sha256

        profile = getattr(self, "_qualified_cache_profile", None)
        if profile is None:
            return
        neuron_config = getattr(self.backbone_config, "neuron_config", None)
        dtype = str(getattr(neuron_config, "torch_dtype", "")).removeprefix("torch.")
        tp_degree = getattr(neuron_config, "tp_degree", None)
        if not dtype or isinstance(tp_degree, bool) or not isinstance(tp_degree, int):
            raise ValueError("Flux backbone does not expose the qualified dtype and TP degree")
        profile.validate_runtime(
            model_id=str(self._cache_runtime_model_id),
            model_revision=str(self._cache_runtime_model_revision),
            height=int(self.height),
            width=int(self.width),
            num_steps=num_steps,
            scheduler_class=type(self.pipe.scheduler).__name__,
            scheduler_config_sha256=scheduler_config_sha256(self.pipe.scheduler),
            dtype=dtype,
            guidance_scale=guidance_scale,
            tp_degree=tp_degree,
        )

    def _validate_qualified_cache_request(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self, "_qualified_cache_profile", None) is None:
            return
        signature = inspect.signature(self.pipe.__call__)
        call = signature.bind_partial(*args, **kwargs).arguments
        if call.get("sigmas") is not None:
            raise ValueError("qualified cache profiles do not allow custom sigma schedules")
        parameter = signature.parameters.get("guidance_scale")
        default_guidance = (
            parameter.default
            if parameter is not None and parameter.default is not inspect.Parameter.empty
            else None
        )
        guidance_scale = call.get("guidance_scale", default_guidance)
        if guidance_scale is None:
            raise ValueError("qualified cache profile requires an explicit guidance scale")
        num_steps, _, _ = self._request_identity(*args, **kwargs)
        self._validate_qualified_cache_runtime(
            num_steps=num_steps,
            guidance_scale=float(guidance_scale),
        )

    def _prepare_probe_free_teacache(self, *args: Any, **kwargs: Any):
        from difflet.pipeline.cache import (
            CacheSession,
            CacheRunner,
            LegacyResidualPredictor,
            QualityRecoveryGuard,
            TeaCachePolicy,
        )
        from difflet.pipeline.teacache import TeaCacheCalibration

        num_steps, height, width = self._request_identity(*args, **kwargs)
        recovery_config = self._probe_free_recovery_config
        if recovery_config is None:
            raise RuntimeError("probe-free TeaCache recovery was not configured")
        calibration = TeaCacheCalibration(
            model="flux",
            shape_label=f"{height}x{width}",
            num_steps=num_steps,
            poly_coef=(0.0, 1.0),
            threshold=0.0,
            warmup_steps=recovery_config.warmup_steps,
            cooldown_steps=recovery_config.cooldown_steps,
            cadence=int(self._teacache_cadence or 0),
            online_delta_alpha=float(self._teacache_online_delta_alpha or 0.0),
        )
        runner = CacheRunner(
            TeaCachePolicy(calibration),
            LegacyResidualPredictor(),
            recovery=QualityRecoveryGuard(recovery_config),
        )
        source = (
            "teacache_cadence" if self._teacache_cadence is not None else "teacache_online_delta"
        )
        session = CacheSession(
            runner,
            num_steps=num_steps,
            configuration_source=source,
        )
        return session

    def _prepare_cache_session(self, *args: Any, **kwargs: Any):
        """Resolve request identity and return a fresh cache session."""

        from difflet.pipeline.cache import (
            ResolvedCacheSession,
            resolve_cache_config,
            resolve_cache_plan,
        )

        num_steps, height, width = self._request_identity(*args, **kwargs)
        if getattr(self, "_qualified_cache_profile", None) is not None:
            self._validate_qualified_cache_request(*args, **kwargs)
            session = self._qualified_cache_profile.build_session(num_steps)
            return session
        shape_label = f"{height}x{width}"
        scheduler_class = type(self.pipe.scheduler).__name__

        if self._cache_plan is not None:
            resolved = resolve_cache_plan(
                self._cache_plan,
                model="flux",
                shape_label=shape_label,
                num_steps=num_steps,
                scheduler_class=scheduler_class,
            )
        else:
            assert self._cache_mask is not None
            assert self._cache_predictor_spec is not None
            resolved = resolve_cache_config(
                num_steps=num_steps,
                mask=self._cache_mask,
                predictor=self._cache_predictor_spec,
                recovery=self._cache_recovery_config,
            )
        session = ResolvedCacheSession(resolved)
        return session
