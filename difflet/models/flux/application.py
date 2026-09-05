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
import logging
import os
from typing import Optional

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
from diffusers.models.autoencoders.vae import Decoder, DecoderTiny
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
    taef1: bool = False,
    taef1_path: str | None = None,
    compile_shapes=None,
):
    shape_extra = {"compile_shapes": compile_shapes} if compile_shapes else {}
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
        **shape_extra,
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
            **shape_extra,
        )
    elif taef1:
        decoder_config = VAEDecoderInferenceConfig(
            neuron_config=decoder_neuron_config,
            load_config=load_diffusers_config(taef1_path or vae_decoder_path),
            height=height,
            width=width,
            model_cls=DecoderTiny,
            **shape_extra,
        )
    else:
        decoder_config = VAEDecoderInferenceConfig(
            neuron_config=decoder_neuron_config,
            load_config=load_diffusers_config(vae_decoder_path),
            height=height,
            width=width,
            transformer_in_channels=backbone_config.in_channels,
            **shape_extra,
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
        taef1: bool = False,
        taef1_path: Optional[str] = None,
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
        # TAEF1: replace the entire VAE with the tiny autoencoder so the
        # pipeline's decode() path uses the lightweight decoder transparently.
        if taef1:
            from diffusers import AutoencoderTiny
            from difflet.pipeline.path_resolver import resolve_model_path
            taef1_model_path = taef1_path or vae_decoder_path
            # AutoencoderTiny.from_pretrained accepts a repo id, but the
            # compiled decoder application needs a LOCAL snapshot dir —
            # get_state_dict() only handles local paths (load_hf_model is
            # unimplemented in this fork). TAEF1 is ~9 MB, so pull everything.
            taef1_local_path = resolve_model_path(
                taef1_model_path,
                allow_patterns=["*.json", "*.safetensors", "*.md", "*.txt"],
            )
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taef1_model_path, torch_dtype=torch.bfloat16,
            )
            self.pipe.vae.decoder = NeuronVAEDecoderApplication(
                model_path=taef1_local_path, config=self.decoder_config,
                model_cls=DecoderTiny,
            )
        else:
            self.pipe.vae.decoder = NeuronVAEDecoderApplication(
                model_path=self.vae_decoder_path, config=self.decoder_config
            )

        # TeaCache fused-A (cclog 85). Mount the probe NEFF when teacache is
        # requested; build the controller when a calibration is provided.
        self.teacache_probe = None
        self.pipe.teacache_probe = None
        self.pipe.teacache_controller = None
        self.pipe.teacache_speedup = None
        enable_teacache = teacache_fused or teacache_speedup is not None
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

        # Probe-free TeaCache modes (fixed cadence / online-delta): purely
        # host-side skip decisions — no probe NEFF, no graph change, so these
        # are runtime-only kwargs excluded from the compile-cache key. The
        # pipeline syncs calibration.num_steps to the request at call time.
        if teacache_cadence is not None or teacache_online_delta_alpha is not None:
            if enable_teacache:
                raise ValueError(
                    "teacache_cadence/teacache_online_delta_alpha are mutually "
                    "exclusive with the adaptive probe modes "
                    "(teacache_fused/teacache_speedup)."
                )
            from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController

            self.pipe.teacache_controller = TeaCacheController(
                TeaCacheCalibration(
                    model="flux",
                    shape_label=f"{int(self.height)}x{int(self.width)}",
                    num_steps=0,  # synced to the request by the pipeline
                    poly_coef=(0.0,),
                    threshold=0.0,
                    cadence=int(teacache_cadence or 0),
                    online_delta_alpha=float(teacache_online_delta_alpha or 0.0),
                )
            )
            print(
                f"[teacache] probe-free controller enabled: cadence="
                f"{int(teacache_cadence or 0)} online_delta_alpha="
                f"{float(teacache_online_delta_alpha or 0.0)}"
            )

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
        return self.pipe(*args, **kwargs)
