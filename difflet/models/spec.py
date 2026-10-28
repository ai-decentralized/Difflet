"""What a model declares to run on the generic GPU flow (``difflet.backends.gpu``).

Each model that runs there has ``difflet/models/<model>/spec.py`` with one ``SPEC``. The spec is
data: the CLI, ``difflet serve`` and the application read it instead of each model carrying its
own copy of the same flow. Behaviour that is genuinely the model's own lives in the
application subclass ``application`` names, under ``difflet/backends/<backend>/<model>/``.

This module and the spec modules must not import torch: the serving parent process reads specs
without loading it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptBudget:
    """One text encoder's token limit, as the serving validator checks it."""

    tokenizer_subdir: str
    max_tokens: int
    #: The message for a prompt over the limit; ``{name}`` is ``prompt`` or ``negative_prompt``.
    too_long: str
    #: A chat template the pipeline wraps the prompt in, and the prefix tokens it drops.
    template: str | None = None
    drop_prefix_tokens: int = 0
    applies_to_negative_prompt: bool = False


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    #: The model's name in messages.
    display_name: str
    #: A short name for allocation ids ("qwen" gives "qwen-cuda").
    short_name: str
    #: Checkpoints ``difflet run`` routes to this model; the first is the default.
    cli_model_ids: tuple[str, ...]
    #: Checkpoints ``difflet serve`` accepts.
    served_checkpoint_ids: tuple[str, ...]
    output: str  # "image" | "video"
    default_height: int
    default_width: int
    default_num_frames: int | None
    default_steps: int
    #: "none": the model takes no guidance scale; "distilled": one pass with the scale as an
    #: input; "cfg": two passes per step.
    guidance: str
    default_guidance_scale: float | None
    max_guidance_scale: float = 20.0
    #: For a model that takes no guidance: the value its pipeline fills into the unused input,
    #: which is also the only value a served request may carry.
    serving_noop_guidance_scale: float | None = None
    #: Why a guidance scale has no effect, for the refusal messages.
    no_guidance_reason: str = ""
    default_fps: int | None = None
    accepts_cfg_parallel: bool = False
    accepts_negative_prompt: bool = False
    #: CLI options the model refuses: (argparse attribute, how the message names it).
    refused_cli_flags: tuple[tuple[str, str], ...] = ()
    #: CLI options forwarded to the application as keyword arguments when set.
    forwarded_cli_flags: tuple[str, ...] = ("torch_compile",)
    #: Serving profile options the model refuses: (profile attribute, how the message names it).
    refused_serving_options: tuple[tuple[str, str], ...] = ()
    #: Video request fields the model refuses: (field, why).
    refused_video_fields: tuple[tuple[str, str], ...] = ()
    #: Application keyword arguments the model refuses when they carry a value, by name or by
    #: prefix, and why.
    refused_application_kwargs: tuple[str, ...] = ()
    refused_application_kwarg_prefixes: tuple[str, ...] = ()
    refused_application_kwargs_reason: str = ""
    prompt_budgets: tuple[PromptBudget, ...] = ()
    smoke_steps: int = 4
    smoke_prompt: str = "a small red square"
    #: The application class, as ``module:attribute``; imported only where torch is loaded.
    application: str = ""


#: Models with a GPU spec, by registry entry name.
_GPU_SPEC_MODULES = {
    "qwen_image": "difflet.models.qwen_image.spec",
    "flux": "difflet.models.flux.spec",
    "wan": "difflet.models.wan.spec",
}


def gpu_model_spec(model_type: str) -> ModelSpec | None:
    """The spec of ``model_type``, or ``None`` when it does not run on the generic GPU flow."""
    module = _GPU_SPEC_MODULES.get(model_type)
    return None if module is None else importlib.import_module(module).SPEC


def gpu_model_specs() -> tuple[ModelSpec, ...]:
    return tuple(gpu_model_spec(name) for name in _GPU_SPEC_MODULES)


def gpu_spec_for_cli_model_id(model_id: str) -> ModelSpec | None:
    """The spec whose ``cli_model_ids`` list ``model_id``."""
    for spec in gpu_model_specs():
        if model_id in spec.cli_model_ids:
            return spec
    return None


__all__ = [
    "ModelSpec",
    "PromptBudget",
    "gpu_model_spec",
    "gpu_model_specs",
    "gpu_spec_for_cli_model_id",
]
