"""Everything a tp4tcad cell needs before ``drive_parallel.sh tp4tcad``, per model.

    python -m benchmark.tcad_prep --model flux_1_dev [--prompts 3]

Steps (each skipped when its output already exists, so a killed run resumes):
  1. the tp4 artifact -- the adapter's exact ``difflet compile`` (a hit when it
     exists); its wall is recorded in the marker, never in a result JSON.
  2. the tp4 reference output at the campaign prompt / seed 42 ->
     benchmark/<device>/logs/<spec_slug>_out.<png|mp4>, the file
     campaign_summary's PSNR column compares every TeaCache cell against.
  3. the placeholder calibration (benchmark.teacache_calibrate placeholder).
  4. probe models (flux / qwen_image / hunyuan_video): the adaptive artifact --
     ``difflet compile --teacache-speedup 1.0 --teacache-calibration <placeholder>``
     builds flux's probe identity or the qwen / hunyuan probe component. Its
     wall is the real cost of the probe NEFF (the cell's compile-only step is
     then a hit, like tp4tc2's). Wan / LTX-2 stay on the tp4 artifact.
  5. record-only collection, one process per calibration prompt.
  6. the fit -> benchmark/<device>/teacache_calib/<slug>_tp4tcad.json.
Logs under benchmark/<device>/logs/tp4tcad_prep/; the marker
<slug>.prep.json there holds each step's wall time and timestamp.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from benchmark.adapters.trainium import TrainiumAdapter, spec_slug
from benchmark.models import (NXD_VENV, cell_is_blocked, logs_dir, resolve,
                              write_blocked_cell)

PROBE_MODELS = ("flux", "qwen_image", "hunyuan_video")

# HBM out-of-memory signatures the Neuron runtime prints when a collect/generate
# cannot fit the DiT + probe (+ VAE) resident on a core-pair.
_OOM_SIGNATURES = (
    "memory allocation failed (ret=-12)",
    "Failed to allocate DEVICE memory",
    "NRT_RESOURCE in nrt_tensor_allocate",
    "Failed to allocate nrt tensor",
)


class CollectBlocked(RuntimeError):
    """A collect failed with a diagnosed device limit (HBM OOM), not a bug."""

    def __init__(self, reason: str, evidence: str, extra: dict):
        super().__init__(reason)
        self.reason, self.evidence, self.extra = reason, evidence, extra


def _diagnose_oom(log: Path) -> "CollectBlocked | None":
    """If a collect log carries an HBM-OOM signature, return the blocked
    diagnosis (peak HBM tensors, the failed allocation, the shape); else None."""
    text = log.read_text(errors="ignore")
    if not any(sig in text for sig in _OOM_SIGNATURES):
        return None
    import re
    tensors = re.findall(r"\\_NC \d+\s*\|\s*[\d.]+[GM]B\s*\|[^|]*\|[^|]*\|\s*([\d.]+GB)", text)
    failed = re.search(r"Failed to allocate ([\d.]+MB)[^\n]*usage: tensors", text)
    peak = max(tensors, default=None, key=lambda s: float(s[:-2]))
    return CollectBlocked(
        reason=("HBM exhausted: the DiT already fills the core-pair HBM at this shape, so the "
                "added TeaCache probe NEFF (calibrated-adaptive only) cannot be resident with it"),
        evidence=f"{log} — Neuron runtime OOM (nrt_tensor_allocate ret=-12)"
                 + (f"; DiT tensors {peak}/~24GB per core-pair" if peak else "")
                 + (f", failed to allocate {failed.group(1)} more" if failed else ""),
        extra={"peak_hbm_tensors_gb": float(peak[:-2]) if peak else None,
               "blocked_class": "hbm_oom"})


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Prep:
    def __init__(self, slug: str, prompts: int):
        self.slug = slug
        self.prompts = prompts
        self.tp4 = resolve(slug, "tp4")
        self.tcad = resolve(slug, "tp4tcad")
        self.logs = Path(logs_dir()) / "tp4tcad_prep"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.adapter = TrainiumAdapter(log_dir=str(self.logs))
        self.marker = self.logs / f"{slug}.prep.json"
        self.state = json.loads(self.marker.read_text()) if self.marker.exists() else {}

    def _done(self, step: str, **kv) -> None:
        self.state[step] = {"at": _now(), **kv}
        self.marker.write_text(json.dumps(self.state, indent=2))

    def _say(self, msg: str) -> None:
        print(f"[prep] {_now()} {self.slug}: {msg}", flush=True)

    def _python(self, argv: list[str], log: Path) -> None:
        env = dict(os.environ)
        env["PATH"] = f"{NXD_VENV}/bin:" + env.get("PATH", "")
        env["PYTHONPATH"] = str(Path.cwd()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with open(log, "w") as fh:
            proc = subprocess.run([f"{NXD_VENV}/bin/python", *argv], stdout=fh,
                                  stderr=subprocess.STDOUT, env=env)
        if proc.returncode != 0:
            tail = "\n".join(log.read_text(errors="ignore").splitlines()[-25:])
            raise RuntimeError(f"{' '.join(argv)} failed ({proc.returncode}); {log}\n{tail}")

    # -- steps ------------------------------------------------------------- #
    def compile_tp4(self) -> None:
        if "compile_tp4" in self.state:
            return self._say("tp4 artifact: compiled earlier (marker), skipping")
        self._say("tp4 artifact: difflet compile")
        wall, breakdown = self.adapter.compile(self.tp4)
        self._done("compile_tp4", wall_s=round(wall, 1), breakdown=breakdown)
        self._say(f"tp4 artifact: {wall:.0f} s")

    def reference_output(self) -> None:
        ext = ".png" if self.tp4.output_kind == "image" else ".mp4"
        out = self.logs.parent / f"{spec_slug(self.tp4)}_out{ext}"
        if out.exists():
            return self._say(f"tp4 reference output exists: {out}")
        self._say(f"tp4 reference output: difflet generate (seed {self.tp4.seed}) -> {out}")
        g = self.adapter.run_generate(self.tp4)
        if not out.exists():
            raise RuntimeError(f"tp4 generate produced no {out}; see {g.get('log')}")
        self._done("reference_output", wall_s=round(g["wall_seconds"], 1),
                   load_s=g.get("load_seconds"), output=str(out))
        self._say(f"tp4 reference output: {g['wall_seconds']:.0f} s")

    def placeholder(self) -> Path:
        from benchmark.teacache_calibrate import write_placeholder
        p = write_placeholder(self.tcad)
        self._say(f"placeholder calibration: {p}")
        return p

    def compile_adaptive(self, placeholder: Path) -> None:
        if self.tcad.model_type not in PROBE_MODELS:
            return self._say("no probe NEFF for this model (host signal): the tp4 artifact is the "
                             "adaptive artifact")
        if "compile_adaptive" in self.state:
            return self._say("adaptive artifact: compiled earlier (marker), skipping")
        cfg = replace(self.tcad, teacache_speedup=1.0, teacache_calibration=str(placeholder))
        self._say("adaptive artifact: difflet compile --teacache-speedup 1.0 (placeholder)")
        wall, breakdown = self.adapter.compile(cfg)
        self._done("compile_adaptive", wall_s=round(wall, 1), breakdown=breakdown,
                   log=str(self.logs / f"{spec_slug(cfg)}_compile.log"))
        self._say(f"adaptive artifact: {wall:.0f} s")

    def collect(self) -> None:
        from benchmark.teacache_calibrate import pairs_path
        pp = pairs_path(self.tcad)
        have = set()
        if pp.exists():
            have = {t["prompt_index"] for t in json.loads(pp.read_text())["trajectories"]}
        for i in range(self.prompts):
            if i in have:
                self._say(f"collect prompt {i}: recorded earlier, skipping")
                continue
            log = self.logs / f"{self.slug}_collect{i}.log"
            self._say(f"collect prompt {i}: record-only generate -> {log}")
            t0 = time.perf_counter()
            try:
                self._python(["-m", "benchmark.teacache_calibrate", "collect", "--model", self.slug,
                              "--prompt-index", str(i)], log)
            except RuntimeError:
                blocked = _diagnose_oom(log)
                if blocked is not None:
                    raise blocked
                raise
            self._done(f"collect_{i}", wall_s=round(time.perf_counter() - t0, 1))
            self._say(f"collect prompt {i}: {time.perf_counter() - t0:.0f} s")

    def fit(self, refit: bool) -> None:
        calib = Path(self.tcad.teacache_calibration)
        if calib.exists() and not refit:
            return self._say(f"calibration exists: {calib} (pass --refit to redo)")
        log = self.logs / f"{self.slug}_fit.log"
        self._python(["-m", "benchmark.teacache_calibrate", "fit", "--model", self.slug], log)
        doc = json.loads(calib.read_text())
        self._done("fit", threshold=doc["threshold"], fit_r2=doc["fit_r2"],
                   signal_pearson=doc["signal_pearson"], n_samples=doc["n_samples"],
                   simulated_skips=doc["simulated_skips_per_prompt"])
        self._say(f"fit: Pearson {doc['signal_pearson']:.3f} R^2 {doc['fit_r2']:.3f} "
                  f"threshold {doc['threshold']:.5f} simulated skips {doc['simulated_skips_per_prompt']} "
                  f"(target {doc['target_skips']}/{doc['num_steps']})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", type=int, default=3, help="calibration prompts to record")
    p.add_argument("--refit", action="store_true")
    a = p.parse_args()
    if cell_is_blocked(a.model, "tp4tcad"):
        print(f"[prep] {a.model}: tp4tcad already recorded BLOCKED; skipping", flush=True)
        print(f"[prep] {a.model}: BLOCKED_SKIP", flush=True)
        return 0
    prep = Prep(a.model, a.prompts)
    prep.compile_tp4()
    prep.reference_output()
    placeholder = prep.placeholder()
    prep.compile_adaptive(placeholder)
    try:
        prep.collect()
    except CollectBlocked as blocked:
        path = write_blocked_cell(a.model, "tp4tcad", reason=blocked.reason,
                                  evidence=blocked.evidence, extra=blocked.extra)
        prep._done("blocked", reason=blocked.reason, evidence=blocked.evidence, **blocked.extra)
        prep._say(f"BLOCKED (device limit) — recorded {path}: {blocked.reason}")
        prep._say(f"{a.model}: BLOCKED_SKIP")
        return 0
    prep.fit(a.refit)
    prep._say("PREP_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
