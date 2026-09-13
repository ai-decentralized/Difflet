"""Closed-loop load generator for a running ``difflet serve`` (serving-layer metrics).

    python -m benchmark.serve_bench --model flux_1_dev --port 8091 --levels 1,2,4 \
        --requests 8 --warmup 2 --out benchmark/trn2/serving/flux_1_dev_tp4.json \
        [--phase-file P] [--sampler-jsonl S]

For each concurrency level c it keeps c in-flight requests until N requests
have completed (closed loop, no think time), records every request's wall
latency and HTTP status, and reports p50/p90/p99 latency of the successes,
throughput (successes / level wall), and the error-code histogram. Image
models go to ``POST /v1/chat/completions`` (JSON, base64 data URL back); video
models to ``POST /v1/videos/sync`` (multipart form, mp4 bytes back) -- the
request shapes the serving smoke script uses. ``--phase-file`` is written with
the level name so ``scripts/sample_serving_resources.py`` can label its
samples; ``--sampler-jsonl`` is then summarised per level (mean NeuronCore
utilisation, peak process-tree RSS).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from benchmark.models import MATRIX, resolve


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _image_request(base: str, cfg, timeout: float) -> tuple[int, int, str]:
    body = json.dumps({
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": cfg.prompt}],
        "extra_body": {"height": cfg.height, "width": cfg.width, "steps": cfg.steps,
                       "seed": cfg.seed},
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, len(data), ""
    except urllib.error.HTTPError as e:
        return e.code, 0, e.read()[:200].decode(errors="ignore")
    except Exception as e:  # timeout / connection
        return 0, 0, f"{type(e).__name__}: {e}"[:200]


def _video_request(base: str, cfg, timeout: float) -> tuple[int, int, str]:
    boundary = "----difflet" + uuid.uuid4().hex
    fields = {
        "model": cfg.model_id, "prompt": cfg.prompt, "height": str(cfg.height),
        "width": str(cfg.width), "num_frames": str(cfg.num_frames),
        "num_inference_steps": str(cfg.steps), "seed": str(cfg.seed),
    }
    if cfg.guidance_scale is not None:
        fields["guidance_scale"] = str(cfg.guidance_scale)
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
    body = ("".join(parts) + f"--{boundary}--\r\n").encode()
    req = urllib.request.Request(
        f"{base}/v1/videos/sync", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, len(data), ""
    except urllib.error.HTTPError as e:
        return e.code, 0, e.read()[:200].decode(errors="ignore")
    except Exception as e:
        return 0, 0, f"{type(e).__name__}: {e}"[:200]


class _NeuronMonitor:
    """A resident ``neuron-monitor`` (1 s period) read on a thread; samples are
    (wall time, mean NeuronCore utilisation %, max core %, device bytes) so a
    load level can be summarised over its own window. The repo's
    sample_serving_resources.py takes one ~6 ms neuron-monitor sample per tick,
    which cannot see load; this one integrates over the level."""

    def __init__(self, binary: str = "neuron-monitor", period: str = "1s"):
        import subprocess
        import tempfile
        cfg = {"period": period,
               "neuron_runtimes": [{"tag_filter": ".*",
                                    "metrics": [{"type": "neuroncore_counters"},
                                                {"type": "memory_used"}]}],
               "system_metrics": []}
        self._cfg = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(cfg, self._cfg)
        self._cfg.close()
        self.samples: list[tuple[float, float, float, float]] = []
        self._proc = None
        self._binary = binary
        self._subprocess = subprocess

    def start(self):
        try:
            self._proc = self._subprocess.Popen(
                [self._binary, "-c", self._cfg.name], stdout=self._subprocess.PIPE,
                stderr=self._subprocess.DEVNULL, text=True)
        except Exception as exc:
            print(f"[serve_bench] neuron-monitor unavailable: {exc}", flush=True)
            self._proc = None
            return
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self._proc.stdout:
            try:
                d = json.loads(line)
            except Exception:
                continue
            utils, dev = [], 0.0
            for rt in d.get("neuron_runtime_data") or []:
                rep = rt.get("report") or {}
                for core in (rep.get("neuroncore_counters") or {}).get("neuroncores_in_use", {}).values():
                    u = core.get("neuroncore_utilization")
                    if isinstance(u, (int, float)):
                        utils.append(float(u))
                for v in _find(rep.get("memory_used") or {}, "neuron_device"):
                    if isinstance(v, (int, float)):
                        dev += float(v)
            if utils:
                self.samples.append((time.time(), statistics.mean(utils), max(utils), dev))

    def stop(self):
        if self._proc:
            self._proc.terminate()

    def window(self, t0: float, t1: float) -> dict | None:
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return None
        return {"samples": len(s),
                "neuroncore_util_mean_pct": round(statistics.mean(x[1] for x in s), 1),
                "neuroncore_util_max_core_pct": round(max(x[2] for x in s), 1),
                "device_mem_used_gb_max": round(max(x[3] for x in s) / 1e9, 2)}


def run_level(send, concurrency: int, n_requests: int) -> dict:
    lock = threading.Lock()
    issued = 0
    records: list[dict] = []

    def worker():
        nonlocal issued
        while True:
            with lock:
                if issued >= n_requests:
                    return
                issued += 1
            t0 = time.perf_counter()
            code, nbytes, err = send()
            t1 = time.perf_counter()
            with lock:
                records.append({"start": t0, "end": t1, "latency_s": round(t1 - t0, 3),
                                "http": code, "bytes": nbytes, "error": err})

    t_start = time.perf_counter()
    wall0 = time.time()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start
    ok = [r["latency_s"] for r in records if r["http"] == 200]
    codes: dict[str, int] = {}
    for r in records:
        codes[str(r["http"])] = codes.get(str(r["http"]), 0) + 1
    return {
        "concurrency": concurrency, "requests": n_requests, "successes": len(ok),
        "wall_s": round(wall, 3),
        "throughput_per_s": round(len(ok) / wall, 4) if wall else None,
        "throughput_per_hour": round(len(ok) / wall * 3600, 1) if wall else None,
        "latency_s": {"p50": round(_pct(ok, 50), 3), "p90": round(_pct(ok, 90), 3),
                      "p99": round(_pct(ok, 99), 3), "mean": round(statistics.mean(ok), 3),
                      "min": round(min(ok), 3), "max": round(max(ok), 3)} if ok else None,
        "http_codes": codes,
        "errors": [r["error"] for r in records if r["http"] != 200][:5],
        "t_start": t_start, "t_end": t_start + wall,
        "wall_t0": wall0, "wall_t1": wall0 + wall,
        "records": records,
    }


def _find(obj, key):
    """Yield every value under ``key`` anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            else:
                yield from _find(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find(v, key)


def summarize_sampler(path: Path, phases: list[str]) -> dict:
    """Per-phase mean NeuronCore utilisation and peak process-tree RSS from the
    sample_serving_resources.py JSONL (tolerant of its exact layout)."""
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        phase = str(d.get("phase") or next(iter(_find(d, "phase")), "?"))
        util = [float(u) for u in _find(d, "neuroncore_utilization") if isinstance(u, (int, float))]
        rss = [int(v) for v in _find(d, "rss_kib") if isinstance(v, (int, float))]
        hbm = [float(v) for v in _find(d, "device_mem_total_bytes") if isinstance(v, (int, float))]
        e = out.setdefault(phase, {"samples": 0, "util_sum": 0.0, "util_n": 0, "rss_peak_kib": 0})
        e["samples"] += 1
        if util:
            e["util_sum"] += statistics.mean(util)
            e["util_n"] += 1
        if rss:
            e["rss_peak_kib"] = max(e["rss_peak_kib"], max(rss))
    for phase, e in out.items():
        e["neuroncore_util_mean_pct"] = round(e["util_sum"] / e["util_n"], 1) if e["util_n"] else None
        e["rss_peak_gb"] = round(e["rss_peak_kib"] / 1048576, 2) if e["rss_peak_kib"] else None
        for k in ("util_sum", "util_n", "rss_peak_kib"):
            e.pop(k, None)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MATRIX))
    p.add_argument("--config", default="tp4")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--levels", default="1,2,4")
    p.add_argument("--requests", type=int, default=8, help="requests per level")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--out", required=True)
    p.add_argument("--phase-file", default=None)
    p.add_argument("--sampler-jsonl", default=None)
    p.add_argument("--ready-seconds", type=float, default=None,
                   help="server start -> /ready, measured by the driver")
    a = p.parse_args()
    cfg = resolve(a.model, a.config)
    base = f"http://127.0.0.1:{a.port}"
    send = (lambda: _image_request(base, cfg, a.timeout)) if cfg.output_kind == "image" \
        else (lambda: _video_request(base, cfg, a.timeout))

    def phase(name: str):
        if a.phase_file:
            Path(a.phase_file).write_text(name)

    mon = _NeuronMonitor()
    mon.start()
    phase("warmup")
    warm = []
    for i in range(a.warmup):
        t0 = time.perf_counter()
        code, nbytes, err = send()
        warm.append({"latency_s": round(time.perf_counter() - t0, 3), "http": code, "bytes": nbytes,
                     "error": err})
        print(f"[serve_bench] warmup {i+1}/{a.warmup}: HTTP {code} {warm[-1]['latency_s']}s "
              f"{nbytes} B {err}", flush=True)
    levels = []
    for c in [int(x) for x in a.levels.split(",") if x]:
        phase(f"c{c}")
        lv = run_level(send, c, a.requests)
        levels.append(lv)
        lat = lv["latency_s"] or {}
        print(f"[serve_bench] c={c}: {lv['successes']}/{lv['requests']} ok, wall {lv['wall_s']}s, "
              f"{lv['throughput_per_hour']}/h, p50 {lat.get('p50')} p90 {lat.get('p90')} "
              f"p99 {lat.get('p99')} s, codes {lv['http_codes']}", flush=True)
    phase("done")
    time.sleep(2)
    mon.stop()
    sampler = summarize_sampler(Path(a.sampler_jsonl), []) if a.sampler_jsonl else {}
    for lv in levels:
        lv["resources"] = sampler.get(f"c{lv['concurrency']}")
        lv["neuron"] = mon.window(lv["wall_t0"], lv["wall_t1"])
        if lv["neuron"]:
            print(f"[serve_bench] c={lv['concurrency']} neuron: {lv['neuron']}", flush=True)
        for r in lv["records"]:
            r.pop("start", None); r.pop("end", None)
        for k in ("t_start", "t_end", "wall_t0", "wall_t1"):
            lv.pop(k, None)
    out = {
        "model_slug": a.model, "config": a.config, "model_id": cfg.model_id,
        "kind": cfg.output_kind,
        "shape": {"height": cfg.height, "width": cfg.width, "num_frames": cfg.num_frames},
        "steps": cfg.steps, "seed": cfg.seed, "guidance_scale": cfg.guidance_scale,
        "parallel": cfg.parallel_dict(), "endpoint": "/v1/chat/completions" if cfg.output_kind == "image"
        else "/v1/videos/sync",
        "ready_seconds": a.ready_seconds, "warmup": warm, "levels": levels,
        "resources_by_phase": sampler,
        "method": ("closed loop: c in-flight requests until N complete, no think time; latency = "
                   "client wall per request incl. queueing; throughput = successes / level wall; "
                   "one resident worker (max_running_requests=1), so c>1 measures queueing, "
                   "not parallel execution"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[serve_bench] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
