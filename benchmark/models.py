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


# TeaCache warmup / cooldown are fixed in difflet/pipeline/teacache.py (5 + 5
# steps never skipped); the calibrations written for tp4tcad use the same.
TEACACHE_WARMUP = 5
TEACACHE_COOLDOWN = 5


def cadence2_skips(steps: int) -> int:
    """Steps fixed cadence 2 skips: every other step inside [warmup, steps - cooldown)."""
    return max((steps - TEACACHE_WARMUP - TEACACHE_COOLDOWN) // 2, 0)


def adaptive_target_speedup(steps: int) -> float:
    """tp4tcad's --teacache-speedup: the denoise-loop speedup of cadence 2's skip
    count (steps / full steps), so the calibrated controller is compared with
    tp4tc2 at the same skip budget -- 28 steps -> 1.474 (9 skips), 20 -> 1.333 (5)."""
    return round(steps / (steps - cadence2_skips(steps)), 3)


def teacache_calibration_path(slug: str, device: Optional[str] = None) -> str:
    """A model's tp4tcad calibration JSON (benchmark.teacache_calibrate writes it
    on the device). Absolute: the CLI cells run as subprocesses from the repo
    root, the real-loop and calibration harnesses load it in-process."""
    return str((Path(results_dir(device)) / "teacache_calib" / f"{slug}_tp4tcad.json").resolve())


def write_blocked_cell(slug: str, config: str, *, reason: str, evidence: str,
                       extra: Optional[dict] = None) -> str:
    """Record a (slug, config) cell as BLOCKED: a supported path that failed on
    device for a diagnosed reason (e.g. an HBM OOM), NOT an unsupported-by-design
    cell (those are UNSUPPORTED / status='skipped'). Writes the result JSON the
    report reads, so a blocked cell shows its diagnosis instead of a blank.

    ``reason`` is one line (shown in the matrix); ``evidence`` is the device
    proof (a log path + the key figures); ``extra`` merges in structured fields
    (peak HBM, the shape, the follow-up)."""
    import json
    cfg = resolve(slug, config)
    d = {
        "model_id": cfg.model_id, "model_slug": slug, "config": config,
        "config_slug": cfg.config_slug, "device_slug": DEVICE,
        "status": "blocked", "blocked_reason": reason, "blocked_evidence": evidence,
        "parallel": cfg.parallel_dict(), "shape": {"height": cfg.height, "width": cfg.width,
                                                   "num_frames": cfg.num_frames},
        "steps": cfg.steps, "teacache": cfg.teacache_dict(),
        "notes": [f"BLOCKED: {reason}", f"evidence: {evidence}"],
    }
    if extra:
        d.update(extra)
    p = Path(json_path(cfg.config_slug))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2))
    return str(p)


def cell_is_blocked(slug: str, config: str) -> bool:
    import json
    p = Path(json_path(resolve(slug, config).config_slug))
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "blocked"
    except (OSError, ValueError):
        return False


def _calibration_summary(path: Optional[str]) -> dict:
    """The fit / threshold fields of a calibration JSON, for the result record
    (so the report can show the signal quality next to the speedup)."""
    import json
    if not path or not Path(path).exists():
        return {}
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    keys = ("fit_r2", "signal_pearson", "threshold", "accumulate", "n_samples",
            "calibration_prompts", "poly_degree", "target_skips", "hardware_measured")
    out = {k: doc[k] for k in keys if k in doc}
    if "poly_coef" in doc and "poly_degree" not in out:
        out["poly_degree"] = len(doc["poly_coef"]) - 1
    return out


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
    # --attention-impl: "megakernel" (attention_cte routing, the default and the
    # identity of every pre-existing artifact) or "sdpa" (PyTorch SDPA through
    # XLA; its own compile-cache identity). Recorded in the result JSON's
    # parallel block only when not the default, so older files still match.
    attention_impl: str = "megakernel"
    # TeaCache (runtime-only knobs -- not in the compile-cache key, so both run
    # on the warm tp4 artifact): fixed cadence skips every N-th DiT step
    # (--teacache-cadence N; warmup/cooldown 5 steps each are fixed in
    # difflet/pipeline/teacache.py), online-delta is the calibration-free
    # adaptive controller (--teacache-online-delta ALPHA: skip when the last
    # full step's relative-L1 delta < alpha x the latched baseline delta).
    teacache_cadence: Optional[int] = None
    teacache_online_delta: Optional[float] = None
    # Calibrated adaptive (--teacache-speedup X --teacache-calibration PATH): the
    # TeaCache paper's controller -- a per-model polynomial maps the block-0
    # modulated-input rel-L1 signal to the expected output change, accumulated
    # until a threshold. resolve() fills both for tp4tcad: the target is cadence
    # 2's skip budget (adaptive_target_speedup) and the calibration is
    # teacache_calibration_path(slug). The probe models (flux: its own probe
    # artifact identity; qwen_image / hunyuan_video: an additive probe component
    # of the DiT stage) need the pair on compile too (compile_teacache_flags);
    # Wan / LTX-2 compute the signal on the host and stay on the tp4 artifact.
    teacache_speedup: Optional[float] = None
    teacache_calibration: Optional[str] = None
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
        if self.attention_impl != "megakernel":
            f += ["--attention-impl", self.attention_impl]
        return f

    def teacache_flags(self) -> list[str]:
        """``difflet generate`` TeaCache tokens. The probe-free modes are
        generate-only runtime knobs; the calibrated-adaptive pair also goes to
        compile (compile_teacache_flags), which is what builds the probe NEFF."""
        f: list[str] = []
        if self.teacache_cadence is not None:
            f += ["--teacache-cadence", str(self.teacache_cadence)]
        if self.teacache_online_delta is not None:
            f += ["--teacache-online-delta", str(self.teacache_online_delta)]
        return f + self.compile_teacache_flags()

    def compile_teacache_flags(self) -> list[str]:
        """``difflet compile`` TeaCache tokens: only calibrated adaptive changes
        the artifact (flux probe identity; qwen_image / hunyuan_video probe
        component), so only it is passed to compile."""
        if self.teacache_speedup is None:
            return []
        return ["--teacache-speedup", str(self.teacache_speedup),
                "--teacache-calibration", str(self.teacache_calibration)]

    def teacache_dict(self) -> Optional[dict]:
        """Result-JSON TeaCache record, or None when TeaCache is off."""
        if (self.teacache_cadence is None and self.teacache_online_delta is None
                and self.teacache_speedup is None):
            return None
        mode = ("fixed_cadence" if self.teacache_cadence is not None
                else "online_delta" if self.teacache_online_delta is not None
                else "adaptive")
        d = {"mode": mode, "cadence": self.teacache_cadence,
             "online_delta_alpha": self.teacache_online_delta,
             "warmup_steps": TEACACHE_WARMUP, "cooldown_steps": TEACACHE_COOLDOWN}
        if mode == "adaptive":
            d["target_speedup"] = self.teacache_speedup
            d["calibration"] = self.teacache_calibration
            d.update(_calibration_summary(self.teacache_calibration))
        return d

    def parallel_dict(self) -> dict:
        """Result-JSON parallel record (same schema as the phase-sweep JSONs)."""
        d = {
            "tp_degree": self.tp,
            "cp_degree": self.cp,
            "cp_mode": self.cp_mode,
            "cfg_parallel_enabled": self.cfg_parallel,
            "sp_enabled": self.sp,
        }
        if self.attention_impl != "megakernel":
            d["attention_impl"] = self.attention_impl
        return d


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
    # Same topology as tp4, DiT attention through PyTorch SDPA instead of the
    # attention_cte megakernel routing (--attention-impl sdpa).
    "tp4sdpa": {"attention_impl": "sdpa"},
    # tp4 at guidance 2.0: the fair (same-work) baseline for tp2cfg -- both
    # CFG branches run on tp4, sequentially, so tp2cfg's parallel two-branch
    # step is compared against a measured two-branch tp4 step, not 2x a
    # single-branch one. True-CFG models only.
    "tp4cfg2": {"guidance_scale": 2.0},
    # TeaCache on the tp4 artifact (no recompile). Fixed cadence 2: skips every
    # other DiT step between the fixed 5-step warmup and cooldown -> 9 of 28
    # steps (FLUX) or 5 of 20 (the others). Online-delta alpha 0.6 (the repo's
    # DEFAULT_ALPHA): the calibration-free adaptive controller.
    "tp4tc2": {"teacache_cadence": 2},
    "tp4tcod": {"teacache_online_delta": 0.6},
    # Calibrated adaptive (--teacache-speedup + --teacache-calibration): the
    # per-model polynomial controller. resolve() sets the target to cadence 2's
    # skip budget and the calibration to teacache_calibration_path(slug), which
    # benchmark.teacache_calibrate writes from on-device (signal, delta) pairs.
    # flux runs on its own probe artifact; qwen_image / hunyuan_video add a
    # probe component to the tp4 DiT stage; Wan / LTX-2 compute the signal on
    # the host (no NEFF change).
    "tp4tcad": {"teacache_adaptive": True},
}

# Online-delta alpha sweep (2026-09-18): the same runtime-only overlay as
# tp4tcod at other alphas, one label per value so every point is its own cell
# (benchmark/<device>/<slug>_tp4tcodNN.json; NN = alpha x 10). 0.6 is repeated
# so the whole curve is measured on one host against one tp4 reference output
# (the committed tp4tcod rows are the 2026-09-13 host and are never rerun).
# Skips are capped at cadence 2's count by the controller's no-two-skips-in-a-
# row latch, so values above ~0.6 can only confirm saturation.
ONLINE_DELTA_SWEEP: dict[str, float] = {
    f"tp4tcod{int(round(a * 10)):02d}": a for a in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
}
CONFIGS.update({label: {"teacache_online_delta": a} for label, a in ONLINE_DELTA_SWEEP.items()})


def is_sweep_label(label: str) -> bool:
    """True for the online-delta alpha-sweep cells (reported in their own
    table, not as campaign features)."""
    return label in ONLINE_DELTA_SWEEP


_CONFIG_DESC = {
    "tp4": "tp=4",
    "tp2cp2": "tp=2 x cp=2 (ulysses)",
    "tp4sp": "tp=4 + sequence parallel",
    "tp2cfg": "tp=2 x CFG-parallel (uncond/cond on separate core pairs)",
    "tp4sdpa": "tp=4, --attention-impl sdpa (PyTorch SDPA via XLA instead of attention_cte)",
    "tp4cfg2": "tp=4 at guidance 2.0 (two sequential CFG branches; baseline for tp2cfg)",
    "tp4tc2": "tp=4 + TeaCache fixed cadence 2 (--teacache-cadence 2)",
    "tp4tcod": "tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.6)",
    "tp4tcad": "tp=4 + TeaCache calibrated adaptive (--teacache-speedup at cadence 2's "
               "skip budget, --teacache-calibration per model)",
}
_CONFIG_DESC.update({
    label: f"tp=4 + TeaCache online-delta adaptive (--teacache-online-delta {a}; alpha sweep)"
    for label, a in ONLINE_DELTA_SWEEP.items()
})


# (slug, config) cells that are unsupported BY DESIGN on this codebase, with the
# reason the report shows. The gates live in difflet (registry capabilities +
# difflet/cli/main.py validators); tests/unit/benchmark cross-checks this table
# against the registry so it cannot drift silently. Cells that are supported but
# fail on device are NOT listed here -- they get a status="failed"/"blocked"
# result with the diagnostic, never a pre-declared skip.
_DISTILLED = ("guidance-distilled model: a single forward pass with the guidance "
              "scale baked into the timestep embedding, so there is no second "
              "(unconditional) CFG branch to run on a separate core pair; "
              "`difflet` rejects --cfg-parallel for it (registry is_distilled=True, "
              "cli/main.py _validate_cfg_parallel)")
_DISTILLED_CFG2 = ("guidance-distilled model: guidance is a conditioning input of its single "
                   "forward pass, so 'guidance 2.0 on tp4' is not a two-branch CFG baseline "
                   "-- the tp2cfg cell it would baseline is N/A for this model too")
UNSUPPORTED: dict[tuple[str, str], str] = {
    ("flux_1_dev", "tp2cfg"): _DISTILLED,
    ("qwen_image", "tp2cfg"): _DISTILLED,
    ("hunyuan_video", "tp2cfg"): _DISTILLED,
    ("hunyuan_video_15", "tp2cfg"): _DISTILLED,
    ("flux_1_dev", "tp4cfg2"): _DISTILLED_CFG2,
    ("qwen_image", "tp4cfg2"): _DISTILLED_CFG2,
    ("hunyuan_video", "tp4cfg2"): _DISTILLED_CFG2,
    ("hunyuan_video_15", "tp4cfg2"): _DISTILLED_CFG2,
    ("ltx_2", "tp2cp2"): ("LTX-2 has no context-parallel path (registry supports_cp=False; "
                          "difflet/models/ltx_2/entry.py raises NotImplementedError: the "
                          "tri-stream video+audio+text transformer has no CP foundation yet)"),
    ("ltx_2", "tp4sp"): ("LTX-2 has no sequence-parallel path (registry supports_sp=False; "
                         "`difflet` rejects --sp for it, cli/main.py _validate_sp)"),
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
    if overrides.pop("teacache_adaptive", False):
        # per-model: the target follows the step count, the calibration the slug
        overrides["teacache_speedup"] = adaptive_target_speedup(base.steps)
        overrides["teacache_calibration"] = teacache_calibration_path(slug)
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
