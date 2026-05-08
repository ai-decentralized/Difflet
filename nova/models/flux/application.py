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
# <<< NxDI fork banner <<<
import logging
import os
import json
import time
from typing import Optional

import torch
import torch.nn as nn

from nova.models.flux.clip.modeling_clip import (
    CLIPInferenceConfig,
    NeuronClipApplication,
)
from nova.core.config import InferenceConfig, NeuronConfig
from nova.models.flux.modeling_flux import (
    FluxBackboneInferenceConfig,
    NeuronFluxBackboneApplication,
)
from nova.models.flux.pipeline import NeuronFluxPipeline
from nova.models.flux.vae.modeling_vae import (
    NeuronVAEDecoderApplication,
    VAEDecoderInferenceConfig,
)
from nova.models.flux.t5.modeling_t5 import (
    NeuronT5Application,
    T5InferenceConfig,
)
from nova.utils.diffusers_adapter import load_diffusers_config
from nova.utils.hf_adapter import load_pretrained_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_flux_parallelism_config(
    backbone_tp_degree: int,
    context_parallel_enabled: bool = False,
    cfg_parallel_enabled: bool = False
) -> int:
    """
    Get the world_size based on backbone_tp_degree and parallelism settings.

    Args:
        backbone_tp_degree: The tensor parallelism degree for the backbone model
        context_parallel_enabled: Whether context parallelism is enabled (default: False)
        cfg_parallel_enabled: Whether CFG parallelism is enabled (default: False)

    Returns:
        int: world_size (equals backbone_tp_degree, or 2x if context/CFG parallel enabled)

    Note:
        context_parallel_enabled and cfg_parallel_enabled are mutually exclusive.
        Both require world_size = 2 × backbone_tp_degree (dp_degree=2).
    """
    # Validate mutual exclusivity
    if context_parallel_enabled and cfg_parallel_enabled:
        raise ValueError(
            "context_parallel_enabled and cfg_parallel_enabled are mutually exclusive. "
            "Only one can be True at a time."
        )

    # Determine if we need 2x world_size (either for context parallel or CFG parallel)
    use_2x_world_size = context_parallel_enabled or cfg_parallel_enabled

    world_size = backbone_tp_degree * 2 if use_2x_world_size else backbone_tp_degree

    return world_size


def create_flux_config(model_path, world_size, backbone_tp_degree, dtype, height, width, inpaint=False,
                       cfg_parallel_enabled=False, context_parallel_enabled=False):
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


class NeuronFluxApplication(nn.Module):
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

        self.pipe = pipeline_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
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
        self.pipe.vae.decoder = NeuronVAEDecoderApplication(
            model_path=self.vae_decoder_path, config=self.decoder_config
        )

    def _compile_component(self, component, component_name, compiled_model_path, compiler_workdir, debug):
        component_path = os.path.join(compiled_model_path, f"{component_name}/")
        # Nova fork: the "is this component compiled?" marker is the
        # post-compile artifact ``model.pt``, not the containing directory.
        # ``neuron_config.json`` is dropped into the directory near the start
        # of compile so the upstream ``os.path.exists(component_path)`` check
        # races with concurrent ranks and short-circuits real compilation.
        compiled_marker = os.path.join(component_path, "model.pt")
        # All ranks line up before deciding so they reach the same conclusion.
        _spmd_barrier()
        if os.path.exists(compiled_marker) and _compiled_config_matches(component, component_path):
            logger.info(f"{component_name} already compiled at {component_path}, skipping compilation.")
        elif not _should_compile_component(component):
            logger.info(
                f"Waiting for compile owner to build replicated component "
                f"{component_name} at {component_path}."
            )
            _wait_for_compiled_component(component, component_name, component_path, compiled_marker)
        else:
            os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(compiler_workdir, component_name)
            component.compile(component_path, debug)
        # All ranks line up before moving on to the next component, so a fast
        # rank can't slip into the next component while peers are still
        # finishing the SPMD trace of this one.
        _spmd_barrier()
        if not os.path.exists(compiled_marker) or not _compiled_config_matches(component, component_path):
            raise RuntimeError(
                f"Flux component {component_name} was not compiled correctly at {component_path}"
            )

    def compile(self, compiled_model_path, debug=False):
        compiler_workdir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
        self._compile_component(self.pipe.text_encoder, "text_encoder", compiled_model_path, compiler_workdir, debug)
        self._compile_component(self.pipe.text_encoder_2, "text_encoder_2", compiled_model_path, compiler_workdir, debug)
        self._compile_component(self.pipe.transformer, "transformer", compiled_model_path, compiler_workdir, debug)
        self._compile_component(self.pipe.vae.decoder, "decoder", compiled_model_path, compiler_workdir, debug)
        os.environ["BASE_COMPILE_WORK_DIR"] = compiler_workdir

    def load(
        self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False
    ):
        # Load global TP components first. The first Neuron model loaded in a
        # process establishes the runtime collective communicator; starting
        # with a smaller component can leave TP=4 components with a
        # communicator of size 1.
        self._load_component(
            self.pipe.text_encoder_2,
            os.path.join(compiled_model_path, "text_encoder_2/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )
        self._load_component(
            self.pipe.transformer,
            os.path.join(compiled_model_path, "transformer/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )
        self._load_component(
            self.pipe.text_encoder,
            os.path.join(compiled_model_path, "text_encoder/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )
        self._load_component(
            self.pipe.vae.decoder,
            os.path.join(compiled_model_path, "decoder/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )

    def _load_component(
        self, component, component_path, start_rank_id, local_ranks_size, skip_warmup
    ):
        component_start_rank_id, component_local_ranks_size = _component_load_rank_range(
            component, start_rank_id, local_ranks_size
        )
        logger.info(f"Loading Flux component from {component_path}")
        _spmd_barrier()
        component.load(
            component_path,
            component_start_rank_id,
            component_local_ranks_size,
            skip_warmup,
        )
        _spmd_barrier()
        logger.info(f"Loaded Flux component from {component_path}")

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)


# Nova fork: small helper used by _compile_component above. Defined at module
# scope (not as a method) so it can be reused if more model classes need the
# same sync semantics. Kept at the bottom to minimize fork drift in the body.
def _spmd_barrier() -> None:
    """torch.distributed.barrier() if a process group is initialized; no-op otherwise.

    The Flux compile flow is invoked by both single-process tooling (where
    no PG exists) and torchrun-style multi-rank entries — both paths must
    work, so we tolerate the absence of a PG.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
    except Exception:  # pragma: no cover — defensive: never let a barrier crash compile
        pass


def _dist_rank_world():
    """Return process rank/world for torchrun-style launches.

    Prefer torch.distributed when initialized; fall back to env so pre-init
    compile decisions are still stable in torchrun.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass

    try:
        return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 0, 1


def _component_world_size(component) -> int:
    return int(getattr(component.config.neuron_config, "world_size", 1))


def _should_compile_component(component) -> bool:
    """Only one process compiles artifacts replicated on every rank."""
    rank, dist_world_size = _dist_rank_world()
    if dist_world_size > 1 and _component_world_size(component) == 1:
        return rank == 0
    return True


def _component_load_rank_range(component, start_rank_id, local_ranks_size):
    """Single-core replicated components always initialize local rank 0."""
    if _component_world_size(component) == 1:
        return 0, 1
    return start_rank_id, local_ranks_size


def _wait_for_compiled_component(component, component_name, component_path, compiled_marker):
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        if os.path.exists(compiled_marker) and _compiled_config_matches(component, component_path):
            return
        time.sleep(2)
    raise RuntimeError(
        f"Timed out waiting for Flux component {component_name} to compile at {component_path}"
    )


def _compiled_config_matches(component, component_path: str) -> bool:
    config_path = os.path.join(component_path, "neuron_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)["neuron_config"]
    except (OSError, KeyError, json.JSONDecodeError):
        return False

    current = component.config.neuron_config
    keys = ("tp_degree", "world_size", "start_rank_id", "local_ranks_size")
    for key in keys:
        if saved.get(key) != getattr(current, key):
            logger.info(
                f"Compiled component config mismatch at {component_path}: "
                f"{key} saved={saved.get(key)!r} current={getattr(current, key)!r}; recompiling."
            )
            return False
    return True
