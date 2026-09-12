"""Benchmark matrix: the "best performing version" config for each supported model.

Each entry encodes the knobs that currently give the best end-to-end performance
on the target accelerator. Shapes are chosen to fit a single trn2.3xlarge (1 Neuron
device, 4 cores x 24 GB); ``tp=4`` is the max on that box (FLUX's registry default is
tp=8, overridden to 4 here). Adjust ``shape``/``tp``/``steps`` to retune.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _resolve_venv() -> str:
    """The Neuron inference venv all Trainium runs use.

    Resolution order: ``$DIFFLET_VENV`` -> ``<repo>/.venv`` (what
    ``scripts/setup_env.sh`` builds; recent Neuron DLAMIs no longer ship the
    prebuilt ``/opt`` venv) -> the historical ``/opt`` path.
    """
    env = os.environ.get("DIFFLET_VENV")
    if env:
        return env
    repo_venv = Path(__file__).resolve().parent.parent / ".venv"
    if (repo_venv / "bin" / "difflet").exists():
        return str(repo_venv)
    return "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"


NXD_VENV = _resolve_venv()

# Results are namespaced per hardware target so other backends reproduce
# side-by-side: benchmark/<device>/{<slug>.json,<slug>.md,RESULTS.md,logs/}.
# Trainium -> "trn2"; an H100/B300 reproduction sets DIFFLET_BENCH_DEVICE=h100/b300.
DEVICE = os.environ.get("DIFFLET_BENCH_DEVICE", "trn2")
_RESULTS_ROOT = "benchmark"


def results_dir(device: Optional[str] = None) -> str:
    return f"{_RESULTS_ROOT}/{device or DEVICE}"


def json_path(slug: str, device: Optional[str] = None) -> str:
    return f"{results_dir(device)}/{slug}.json"


def report_path(slug: str, device: Optional[str] = None) -> str:
    return f"{results_dir(device)}/{slug}.md"


def logs_dir(device: Optional[str] = None) -> str:
    return f"{results_dir(device)}/logs"


@dataclass
class BenchConfig:
    model_id: str
    model_type: str
    revision: Optional[str] = None           # pinned HF commit (exact weights for repro)
    tp: int = 4
    cp: int = 1
    # Parallel axes beyond tp/cp. First-class fields (not extra_generate_flags)
    # so the result JSON keys configs correctly: a tp4 --sp run must record
    # sp_enabled=True or it is indistinguishable from plain tp4 (the
    # planner-benchmark-collection doc's pre-collection must-fix).
    cp_mode: str = "gather_kv"               # gather_kv | ring | ulysses (cp>1 only)
    cfg_parallel: bool = False               # --cfg-parallel (true-CFG models only)
    sp: bool = False                         # --sp (Megatron sequence parallelism)
    dtype: str = "bf16"
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    steps: int = 20
    guidance_scale: Optional[float] = None
    seed: int = 42                           # difflet CLI default; pinned for repro
    prompt: str = "a cinematic shot of a red fox running through a snowy forest"
    output_kind: str = "video"               # video | image
    extra_generate_flags: list[str] = field(default_factory=list)
    config_label: str = ""                   # human description of the best-perf knobs
    notes: str = ""
    # e2e breakdown stage labels, in pipeline order, used when the generate log has
    # no [role] markers (Wan, LTX-2). None -> keep the parser's auto-labels.
    stage_names: Optional[list[str]] = None
    # note about stages that run on the host (not a Neuron load line), explaining
    # the compute residual for host-pipeline models.
    e2e_host_note: str = ""
    # Filled by resolve(): the MATRIX key and the parallel-config label. Together
    # they name the result files (see config_slug) so a tp2cp2 run never
    # overwrites the tp4 history of the same model.
    slug: str = ""
    config: str = "tp4"

    @property
    def config_slug(self) -> str:
        """Result-file stem: ``<slug>`` for tp4 (the historical files), else
        ``<slug>_<config>`` (precedent: scripts/flux_parallel_sweep.py's
        ``flux_<label>.json``)."""
        base = self.slug or "<slug>"
        return base if self.config == "tp4" else f"{base}_{self.config}"

    def shape_flags(self) -> list[str]:
        f: list[str] = []
        if self.height is not None:
            f += ["--height", str(self.height)]
        if self.width is not None:
            f += ["--width", str(self.width)]
        if self.num_frames is not None:
            f += ["--num-frames", str(self.num_frames)]
        return f

    def parallel_flags(self) -> list[str]:
        """The full parallel-configuration CLI tokens for this config."""
        f = ["--tp-degree", str(self.tp), "--cp-degree", str(self.cp)]
        if self.cp > 1 and self.cp_mode != "gather_kv":
            f += ["--cp-mode", self.cp_mode]
        if self.cfg_parallel:
            f.append("--cfg-parallel")
        if self.sp:
            f.append("--sp")
        return f

    def parallel_dict(self) -> dict:
        """Result-JSON parallel record (same schema as the phase-sweep JSONs)."""
        return {
            "tp_degree": self.tp,
            "cp_degree": self.cp,
            "cp_mode": self.cp_mode,
            "cfg_parallel_enabled": self.cfg_parallel,
            "sp_enabled": self.sp,
        }


# Parallel-topology labels, all sized to the 4 NeuronCores of a trn2.3xlarge
# (same labels as scripts/verify_cli.py PARALLEL_CONFIGS / the planner's
# config_label). Each maps to BenchConfig field overrides applied on top of the
# MATRIX entry by resolve(). CP runs use ulysses so HunyuanVideo (whose
# gather_kv CP hits a neuronx-cc internal error and whose ring CP has no mask
# path) is measurable with the same mode as the other models. tp2cfg doubles the
# world (2 tp x 2 CFG branches = 4 cores) and needs guidance > 1 to have a
# second branch at all: the 2026-09-12 campaign runs it at guidance 2.0.
CONFIGS: dict[str, dict[str, Any]] = {
    "tp4": {},
    "tp2cp2": {"tp": 2, "cp": 2, "cp_mode": "ulysses"},
    "tp4sp": {"tp": 4, "sp": True},
    "tp2cfg": {"tp": 2, "cfg_parallel": True, "guidance_scale": 2.0},
}

_CONFIG_DESC = {
    "tp4": "tp=4",
    "tp2cp2": "tp=2 x cp=2 (ulysses)",
    "tp4sp": "tp=4 + sequence parallel",
    "tp2cfg": "tp=2 x CFG-parallel (uncond/cond on separate core pairs)",
}


def resolve(slug: str, config: str = "tp4") -> "BenchConfig":
    """The MATRIX entry for ``slug`` with the ``config`` topology applied.

    Raises KeyError for an unknown slug or config label. The returned config
    carries ``slug``/``config`` so result paths derive from ``config_slug``.
    """
    from dataclasses import replace
    if slug not in MATRIX:
        raise KeyError(f"unknown model '{slug}'. known: {', '.join(MATRIX)}")
    if config not in CONFIGS:
        raise KeyError(f"unknown config '{config}'. known: {', '.join(CONFIGS)}")
    base = MATRIX[slug]
    overrides = dict(CONFIGS[config])
    if config != "tp4":
        overrides["config_label"] = f"{_CONFIG_DESC[config]}; {base.config_label}"
    return replace(base, slug=slug, config=config, **overrides)


def add_config_arg(parser) -> None:
    """``--config <label>`` for every harness entry point."""
    parser.add_argument("--config", default="tp4", choices=sorted(CONFIGS),
                        help="parallel topology label (default tp4); non-tp4 runs "
                             "write benchmark/<device>/<slug>_<config>.{json,md}")


# Keyed by a short slug used for the report filename (benchmark/<slug>.md).
MATRIX: dict[str, BenchConfig] = {
    "ltx_2": BenchConfig(
        model_id="Lightricks/LTX-2",
        revision="47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        model_type="ltx_2",
        tp=4, height=480, width=704, num_frames=49, steps=20, guidance_scale=1.0,
        output_kind="video",
        config_label="tp=4, bf16, TP-sharded transformer + attention_cte self-attn, "
                     "guidance=1.0 (batch-1 NEFF)",
        notes="Default registry shape 512x768x121 also compiles; 480x704x49 used here "
              "as the representative fast shape. CFG (guidance>1) needs a batch-2 NEFF.",
        stage_names=["transformer (denoise loop) [Neuron]"],
        e2e_host_note="text-encoder and VAE decode run on the host "
                      "(enable_host_pipeline/enable_decode_components), so only the "
                      "transformer is a Neuron load; the residual is host text-encode "
                      "+ denoise + host VAE decode.",
    ),
    "wan_2_1": BenchConfig(
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        revision="38ec498cb3208fb688890f8cc7e94ede2cbd7f68",
        model_type="wan",
        tp=4, height=480, width=832, num_frames=9, steps=20, guidance_scale=1.0,
        output_kind="video",
        config_label="tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage "
                     "(transformer + VAE) subprocess pipeline",
        stage_names=["text_encoder (UMT5)", "transformer (denoise loop)", "vae_decoder"],
    ),
    "wan_2_2": BenchConfig(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        revision="5be7df9619b54f4e2667b2755bc6a756675b5cd7",
        model_type="wan",
        tp=4, height=480, width=832, num_frames=9, steps=20, guidance_scale=1.0,
        output_kind="video",
        config_label="tp=4, bf16, A14B (high/low-noise experts), attention_cte",
        stage_names=["text_encoder (UMT5)", "transformer (denoise loop)", "vae_decoder"],
    ),
    "flux_1_dev": BenchConfig(
        model_id="black-forest-labs/FLUX.1-dev",
        revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        model_type="flux",
        tp=4, height=1024, width=1024, num_frames=None, steps=28, guidance_scale=3.5,
        output_kind="image",
        config_label="tp=4 (registry default tp=8 -> 4 on trn2.3xlarge), bf16, attention_cte",
        # measured load order (by size: T5 ~10GB, transformer ~24GB, then the two tiny ones)
        stage_names=["text_encoder_t5", "transformer (denoise loop)",
                     "text_encoder_clip", "vae_decoder"],
    ),
    "qwen_image": BenchConfig(
        model_id="Qwen/Qwen-Image",
        revision="75e0b4be04f60ec59a75f475837eced720f823b6",
        model_type="qwen_image",
        tp=4, height=1024, width=1024, num_frames=None, steps=20, guidance_scale=4.0,
        output_kind="image",
        config_label="tp=4, bf16, joint attention via attention_cte",
    ),
    "hunyuan_video": BenchConfig(
        model_id="hunyuanvideo-community/HunyuanVideo",
        revision="e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
        model_type="hunyuan_video",
        tp=4, height=320, width=512, num_frames=61, steps=20, guidance_scale=6.0,
        output_kind="video",
        config_label="tp=4, bf16, attention_cte",
        e2e_host_note="VAE decode runs on the host (no Neuron load line); the residual "
                      "is CLIP+Llama encode + denoise loop + host VAE decode.",
    ),
    "hunyuan_video_15": BenchConfig(
        model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        revision="286be7ce72277246578a3e3cc2487e95ddae5bcf",
        model_type="hunyuan_video_15",
        tp=4, height=480, width=848, num_frames=121, steps=20, guidance_scale=6.0,
        output_kind="video",
        config_label="tp=4, bf16, attention_cte + MX precision ops",
    ),
}
