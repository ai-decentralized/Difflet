"""Trainium backend adapter — drives the difflet CLI in the Neuron inference venv.

Implements the harness ``BackendAdapter`` contract by shelling out to ``difflet
compile`` / ``difflet generate`` (always with the nxd_inference venv python) and
parsing difflet's own emitted timing lines plus subprocess wall-clock. This keeps
the measurement faithful to how difflet is actually run in production.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from benchmark.harness import BackendAdapter, OutputInfo
from benchmark.models import NXD_VENV

_PY = f"{NXD_VENV}/bin/python"
_DIFFLET = f"{NXD_VENV}/bin/difflet"

# difflet log signals we parse for sub-phase timings.
_RE_BUILD = re.compile(r"Finished building model in ([\d.]+) seconds")
_RE_LOAD = re.compile(r"Finished weights loading in ([\d.]+) seconds")
_RE_FWD = re.compile(r"trainium forward elapsed = ([\d.]+)s")
_RE_SHARD = re.compile(r"Done Sharding weights in ([\d.]+)")


def _filter(text: str) -> str:
    drop = ("UserWarning", "warnings.warn", "Warning:", "OperatorEntry", "CCOM WARN",
            "nccl_net", "net_plugin", "OFI", "blockwise", "operator:")
    return "\n".join(l for l in text.splitlines() if not any(d in l for d in drop))


class TrainiumAdapter(BackendAdapter):
    name = "trainium"

    def __init__(self, *, cache_dir: str | None = None, log_dir: str | None = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/difflet")
        self.log_dir = Path(log_dir or "benchmark/results/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # -- info ------------------------------------------------------------- #
    def device_info(self) -> str:
        try:
            out = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=30).stdout
            inst = re.search(r"instance-type:\s*(\S+)", out)
            cores = re.findall(r"\|\s*\d+\s*\|\s*(\d+)\s*\|", out)
            mem = re.search(r"(\d+)\s*GB", out)
            parts = []
            if inst:
                parts.append(inst.group(1))
            if cores:
                parts.append(f"{cores[0]} NeuronCores")
            if mem:
                parts.append(f"{mem.group(1)} GB/device")
            return " / ".join(parts) or "Trainium (neuron-ls parse failed)"
        except Exception:
            return "Trainium (neuron-ls unavailable)"

    def toolchain(self) -> dict[str, str]:
        try:
            out = subprocess.run(
                [_PY, "-c",
                 "import importlib.metadata as m\n"
                 "for p in ['torch','torch-neuronx','neuronx-cc','neuronx-distributed','diffusers']:\n"
                 "    try: print(p+'='+m.version(p))\n"
                 "    except Exception: pass"],
                capture_output=True, text=True, timeout=60).stdout
            return dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
        except Exception:
            return {}

    # -- phases ----------------------------------------------------------- #
    def prepare(self, spec) -> None:
        cfg = spec
        log = self.log_dir / f"{spec_slug(cfg)}_download.log"
        self._run([_DIFFLET, "download", "--model-id", cfg.model_id], log, timeout=7200)

    def compile(self, spec) -> tuple[float, dict[str, float]]:
        cfg = spec
        log = self.log_dir / f"{spec_slug(cfg)}_compile.log"
        cmd = [_DIFFLET, "compile", "--model-id", cfg.model_id,
               "--tp-degree", str(cfg.tp), "--cp-degree", str(cfg.cp),
               "--cache-dir", self.cache_dir] + cfg.shape_flags()
        t0 = time.perf_counter()
        text = self._run(cmd, log, timeout=14400)
        wall = time.perf_counter() - t0
        builds = [float(x) for x in _RE_BUILD.findall(text)]
        breakdown = {f"component_{i}": b for i, b in enumerate(builds)}
        breakdown["wall_total"] = wall
        return wall, breakdown

    def run_generate(self, spec) -> dict:
        cfg = spec
        out_path = self.log_dir.parent / f"{spec_slug(cfg)}_out"
        log = self.log_dir / f"{spec_slug(cfg)}_generate.log"
        cmd = [_DIFFLET, "generate", "--model-id", cfg.model_id,
               "--tp-degree", str(cfg.tp), "--cp-degree", str(cfg.cp),
               "--cache-dir", self.cache_dir,
               "--prompt", cfg.prompt, "--steps", str(cfg.steps),
               "--output", str(out_path) + ".mp4"] + cfg.shape_flags()
        if cfg.guidance_scale is not None:
            cmd += ["--guidance-scale", str(cfg.guidance_scale)]
        cmd += cfg.extra_generate_flags
        t0 = time.perf_counter()
        text = self._run(cmd, log, timeout=14400)
        wall = time.perf_counter() - t0
        res: dict = {"wall_seconds": wall, "log": str(log)}
        load = _RE_LOAD.findall(text)
        if load:
            res["load_seconds"] = float(load[-1])
        shard = _RE_SHARD.findall(text)
        if shard:
            res["shard_seconds"] = float(shard[-1])
        fwd = [float(x) for x in _RE_FWD.findall(text)]
        if fwd:
            res["step_seconds"] = fwd
        res["output"] = self._inspect_output(out_path)
        return res

    # -- helpers ---------------------------------------------------------- #
    def _inspect_output(self, out_path: Path) -> dict:
        pt = out_path.with_suffix(".pt")
        cands = [pt, out_path.with_suffix(".mp4")]
        target = next((p for p in cands if p.exists()), None)
        if target is None:
            return OutputInfo(note="no output file produced").__dict__
        if target.suffix != ".pt":
            return OutputInfo(note=f"saved {target.name}").__dict__
        code = (
            "import torch,sys,json\n"
            f"x=torch.load('{pt}',map_location='cpu')\n"
            "x=x[0] if isinstance(x,(list,tuple)) else x\n"
            "x=x.float()\n"
            "print(json.dumps(dict(shape=list(x.shape),dtype=str(x.dtype),"
            "finite=bool(torch.isfinite(x).all()),min=float(x.min()),max=float(x.max()),"
            "mean=float(x.mean()),std=float(x.std()))))"
        )
        try:
            out = subprocess.run([_PY, "-c", code], capture_output=True, text=True, timeout=300)
            import json
            d = json.loads(out.stdout.strip().splitlines()[-1])
            return OutputInfo(shape=d["shape"], dtype=d["dtype"], finite=d["finite"],
                              min=d["min"], max=d["max"], mean=d["mean"], std=d["std"],
                              note=f"saved {pt.name}").__dict__
        except Exception as e:
            return OutputInfo(note=f"output inspect failed: {e}").__dict__

    def _run(self, cmd: list[str], log: Path, timeout: int) -> str:
        env = dict(os.environ)
        env["PATH"] = f"{NXD_VENV}/bin:" + env.get("PATH", "")
        with open(log, "w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  env=env, timeout=timeout)
        text = _filter(log.read_text(errors="ignore"))
        if proc.returncode != 0:
            tail = "\n".join(text.splitlines()[-25:])
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
        return text


def spec_slug(cfg) -> str:
    base = cfg.model_id.split("/")[-1].replace(".", "_").replace("-", "_")
    return base.lower()
