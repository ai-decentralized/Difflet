"""Closed-loop load generator for a running ``difflet serve`` (serving-layer metrics).

    python -m benchmark.serve_bench --model flux_1_dev --port 8091 --levels 1,2,4 \
        --requests 8 --warmup 2 --out benchmark/trn2/serving/flux_1_dev_tp4.json \
        [--phase-file P] [--sampler-jsonl S] [--api-mode sync|async] [--burst N]

For each concurrency level c it keeps c in-flight requests until N requests
have completed (closed loop, no think time), records every request's wall
latency and HTTP status, and reports p50/p90/p99 latency of the successes,
throughput (successes / level wall), and the error-code histogram. Image
models go to ``POST /v1/chat/completions`` (JSON, base64 data URL back); video
models to ``POST /v1/videos/sync`` (multipart form, mp4 bytes back) -- the
request shapes the serving smoke script uses.

``--api-mode async`` (video models only) drives the asynchronous Videos job API
instead: ``POST /v1/videos`` (returns the queued job at once), poll
``GET /v1/videos/{id}`` every ``--poll-interval`` s until the job is terminal,
``GET /v1/videos/{id}/content`` for the bytes, then ``DELETE`` so retained
artifacts / storage reservations do not pile up across levels. Latency =
create start -> content downloaded (the same user-visible completion the sync
path measures; DELETE is not counted). Per request it also records the create
acknowledgement time, the first time the job was seen in_progress (queue
wait), the number of polls, and the download time. ``--burst N`` adds one
open-loop burst: N creates back to back, then poll all -- how the async API
absorbs a burst without holding N connections open (the server admits
1 + --max-queued-requests jobs; the rest get 429).

Both API paths share the server's single FIFO and its one resident worker, so
the device time per request is the same by construction; this measures the
client-visible difference. ``--phase-file`` is written with the level name so
``scripts/sample_serving_resources.py`` can label its samples;
``--sampler-jsonl`` is then summarised per level (mean NeuronCore
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


def _http(method: str, url: str, timeout: float, data: bytes | None = None,
          headers: dict | None = None) -> tuple[int, bytes, str]:
    """(status, body, error). status 0 = no HTTP response (timeout / connection)."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), ""
    except urllib.error.HTTPError as e:
        return e.code, b"", e.read()[:200].decode(errors="ignore")
    except Exception as e:  # timeout / connection
        return 0, b"", f"{type(e).__name__}: {e}"[:200]


def _image_request(base: str, cfg, timeout: float) -> tuple[int, int, str, dict]:
    body = json.dumps({
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": cfg.prompt}],
        "extra_body": {"height": cfg.height, "width": cfg.width, "steps": cfg.steps,
                       "seed": cfg.seed},
    }).encode()
    code, data, err = _http("POST", f"{base}/v1/chat/completions", timeout, body,
                            {"Content-Type": "application/json"})
    return code, len(data), err, {}


def _video_multipart(cfg) -> tuple[bytes, dict]:
    """The multipart create request both video endpoints accept."""
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
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def _video_request(base: str, cfg, timeout: float) -> tuple[int, int, str, dict]:
    body, headers = _video_multipart(cfg)
    code, data, err = _http("POST", f"{base}/v1/videos/sync", timeout, body, headers)
    return code, len(data), err, {}


def _poll_until_terminal(base: str, video_id: str, t0: float, deadline: float,
                         poll_s: float) -> tuple[dict | None, dict]:
    """Poll one job until completed/failed (or deadline). Returns (job or None,
    {queued_s, terminal_s, polls, poll_errors}) with times relative to t0."""
    polls = poll_errors = 0
    queued_s = None
    job = None
    while time.perf_counter() < deadline:
        time.sleep(poll_s)
        st, d, _ = _http("GET", f"{base}/v1/videos/{video_id}", 30.0)
        polls += 1
        if st != 200:
            poll_errors += 1
            if poll_errors > 20:
                break
            continue
        job = json.loads(d)
        status = job.get("status")
        if status == "in_progress" and queued_s is None:
            queued_s = round(time.perf_counter() - t0, 3)
        if status in ("completed", "failed"):
            return job, {"queued_s": queued_s, "terminal_s": round(time.perf_counter() - t0, 3),
                         "polls": polls, "poll_errors": poll_errors}
    return None, {"queued_s": queued_s, "terminal_s": None, "polls": polls,
                  "poll_errors": poll_errors}


def _video_async_request(base: str, cfg, timeout: float, poll_s: float
                         ) -> tuple[int, int, str, dict]:
    """One async job end to end: create -> poll -> content -> DELETE.
    Returns (http, bytes, error, extra); http 200 only when the job completed
    and its content downloaded; 599 = the job itself failed."""
    body, headers = _video_multipart(cfg)
    t0 = time.perf_counter()
    code, data, err = _http("POST", f"{base}/v1/videos", timeout, body, headers)
    create_s = round(time.perf_counter() - t0, 4)
    extra: dict = {"create_s": create_s, "create_http": code}
    if code != 200:
        return code, 0, err, extra
    try:
        job = json.loads(data)
        video_id = job["id"]
    except Exception as exc:
        return 0, 0, f"bad create response: {exc}", extra
    extra["job_id"] = video_id
    job, poll = _poll_until_terminal(base, video_id, t0, t0 + timeout, poll_s)
    extra.update(poll)
    if job is None:
        _http("DELETE", f"{base}/v1/videos/{video_id}", 30.0)   # cancel / clean up
        return 0, 0, "poll timeout: job not terminal", extra
    extra["job_status"] = job.get("status")
    extra["created_at"] = job.get("created_at")
    extra["completed_at"] = job.get("completed_at")
    if job.get("status") != "completed":
        _http("DELETE", f"{base}/v1/videos/{video_id}", 30.0)
        return 599, 0, f"job failed: {json.dumps(job.get('error'))[:160]}", extra
    t_dl = time.perf_counter()
    st, content, cerr = _http("GET", f"{base}/v1/videos/{video_id}/content", timeout)
    extra["download_s"] = round(time.perf_counter() - t_dl, 4)
    extra["latency_s"] = round(time.perf_counter() - t0, 3)   # create -> bytes in hand
    t_del = time.perf_counter()
    dst, _, derr = _http("DELETE", f"{base}/v1/videos/{video_id}", 30.0)
    extra["delete_s"] = round(time.perf_counter() - t_del, 4)
    extra["delete_http"] = dst
    if st != 200:
        return st, 0, f"content: {cerr}", extra
    return 200, len(content), "", extra


def run_burst(base: str, cfg, n: int, timeout: float, poll_s: float) -> dict:
    """Open-loop burst: n creates back to back (no waiting), then poll every job
    until terminal. Measures the create acknowledgements, how many were admitted
    (1 + max_queued_requests fit; the rest 429), and the completion profile."""
    t0 = time.perf_counter()
    creates: list[dict] = []
    body, headers = _video_multipart(cfg)
    for _ in range(n):
        tc = time.perf_counter()
        code, data, err = _http("POST", f"{base}/v1/videos", timeout, body, headers)
        rec = {"create_s": round(time.perf_counter() - tc, 4), "http": code, "error": err,
               "submitted_at_s": round(tc - t0, 4)}
        if code == 200:
            try:
                rec["job_id"] = json.loads(data)["id"]
            except Exception as exc:
                rec["error"] = f"bad create response: {exc}"
        creates.append(rec)
    all_acked_s = round(time.perf_counter() - t0, 4)
    active = {r["job_id"]: r for r in creates if r.get("job_id")}
    deadline = t0 + timeout
    while active and time.perf_counter() < deadline:
        time.sleep(poll_s)
        for vid, rec in list(active.items()):
            st, d, _ = _http("GET", f"{base}/v1/videos/{vid}", 30.0)
            if st != 200:
                continue
            job = json.loads(d)
            if job.get("status") == "in_progress" and "queued_s" not in rec:
                rec["queued_s"] = round(time.perf_counter() - t0, 3)
            if job.get("status") in ("completed", "failed"):
                rec["job_status"] = job.get("status")
                rec["completed_s"] = round(time.perf_counter() - t0, 3)
                rec["created_at"] = job.get("created_at")
                rec["completed_at"] = job.get("completed_at")
                del active[vid]
    wall = round(time.perf_counter() - t0, 3)
    for rec in creates:                      # drain: bytes + DELETE, not timed into the wall
        vid = rec.get("job_id")
        if not vid:
            continue
        if rec.get("job_status") == "completed":
            st, content, _ = _http("GET", f"{base}/v1/videos/{vid}/content", timeout)
            rec["bytes"] = len(content) if st == 200 else 0
        _http("DELETE", f"{base}/v1/videos/{vid}", 30.0)
    done = sorted(r["completed_s"] for r in creates if r.get("job_status") == "completed")
    codes: dict[str, int] = {}
    for r in creates:
        codes[str(r["http"])] = codes.get(str(r["http"]), 0) + 1
    cs = [r["create_s"] for r in creates]
    return {
        "jobs": n, "admitted": len([r for r in creates if r.get("job_id")]),
        "completed": len(done), "http_codes": codes, "all_acked_s": all_acked_s,
        "create_s": {"p50": round(_pct(cs, 50), 4), "max": round(max(cs), 4)},
        "wall_s": wall,
        "throughput_per_hour": round(len(done) / wall * 3600, 1) if done and wall else None,
        "completion_s": done,
        "inter_completion_s": ([round(b - a, 3) for a, b in zip(done, done[1:])] if len(done) > 1
                               else []),
        "records": creates,
    }


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
            code, nbytes, err, extra = send()
            t1 = time.perf_counter()
            rec = {"start": t0, "end": t1, "latency_s": round(t1 - t0, 3),
                   "http": code, "bytes": nbytes, "error": err}
            rec.update(extra)          # async: latency_s = create -> bytes (DELETE excluded)
            with lock:
                records.append(rec)

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
    out = {
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
    okr = [r for r in records if r["http"] == 200 and "create_s" in r]
    if okr:
        def _p(key, p=50, nd=4):
            xs = [r[key] for r in okr if r.get(key) is not None]
            return round(_pct(xs, p), nd) if xs else None
        out["async"] = {
            "create_s": {"p50": _p("create_s"), "p90": _p("create_s", 90),
                         "max": round(max(r["create_s"] for r in okr), 4)},
            "queued_s": {"p50": _p("queued_s", nd=3), "max": _p("queued_s", 100, 3)},
            "terminal_s": {"p50": _p("terminal_s", nd=3)},
            "download_s": {"p50": _p("download_s")},
            "delete_s": {"p50": _p("delete_s")},
            "polls_mean": round(statistics.mean(r["polls"] for r in okr), 1),
            # what the async round trip adds on top of the job's own completion:
            # poll quantisation + the content download
            "overhead_s": {"p50": round(_pct([r["latency_s"] - r["terminal_s"] for r in okr
                                              if r.get("terminal_s") is not None], 50), 3)},
        }
    return out


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


_METHOD = {
    "sync": ("closed loop: c in-flight requests until N complete, no think time; latency = "
             "client wall per request incl. queueing; throughput = successes / level wall; "
             "one resident worker (max_running_requests=1), so c>1 measures queueing, "
             "not parallel execution"),
    "async": ("closed loop over the async Videos job API: each request = POST /v1/videos "
              "(create) -> poll GET /v1/videos/{id} every poll_interval_s until terminal -> "
              "GET .../content -> DELETE; latency = create start -> content downloaded "
              "(DELETE excluded), so it is quantised by the poll interval; throughput = "
              "successes / level wall; the same single FIFO + one resident worker as the "
              "sync path, so c>1 measures queueing (jobs wait server-side without an open "
              "connection), not parallel execution; burst = N creates back to back then poll"),
}


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
    p.add_argument("--api-mode", choices=("sync", "async"), default="sync",
                   help="video models: /v1/videos/sync (blocking) or the /v1/videos job API")
    p.add_argument("--poll-interval", type=float, default=0.25,
                   help="async: seconds between job status polls")
    p.add_argument("--burst", type=int, default=0,
                   help="async: after the levels, submit N creates back to back and poll all")
    a = p.parse_args()
    cfg = resolve(a.model, a.config)
    base = f"http://127.0.0.1:{a.port}"
    if cfg.output_kind == "image":
        if a.api_mode == "async":
            print("[serve_bench] image models have no async endpoint (/v1/videos is video-only)",
                  file=sys.stderr)
            return 2
        send = lambda: _image_request(base, cfg, a.timeout)          # noqa: E731
        endpoint = "/v1/chat/completions"
    elif a.api_mode == "async":
        send = lambda: _video_async_request(base, cfg, a.timeout, a.poll_interval)  # noqa: E731
        endpoint = "/v1/videos (async job API: create, poll, content, delete)"
    else:
        send = lambda: _video_request(base, cfg, a.timeout)          # noqa: E731
        endpoint = "/v1/videos/sync"

    def phase(name: str):
        if a.phase_file:
            Path(a.phase_file).write_text(name)

    mon = _NeuronMonitor()
    mon.start()
    phase("warmup")
    warm = []
    for i in range(a.warmup):
        t0 = time.perf_counter()
        code, nbytes, err, extra = send()
        rec = {"latency_s": round(time.perf_counter() - t0, 3), "http": code, "bytes": nbytes,
               "error": err}
        rec.update(extra)
        warm.append(rec)
        print(f"[serve_bench] warmup {i+1}/{a.warmup}: HTTP {code} {rec['latency_s']}s "
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
        if lv.get("async"):
            print(f"[serve_bench] c={c} async: {lv['async']}", flush=True)
    burst = None
    if a.burst > 0 and a.api_mode == "async":
        phase(f"burst{a.burst}")
        burst = run_burst(base, cfg, a.burst, a.timeout, a.poll_interval)
        print(f"[serve_bench] burst {a.burst}: admitted {burst['admitted']}, completed "
              f"{burst['completed']}, all acked in {burst['all_acked_s']}s, wall {burst['wall_s']}s, "
              f"{burst['throughput_per_hour']}/h, codes {burst['http_codes']}", flush=True)
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
        "parallel": cfg.parallel_dict(), "endpoint": endpoint, "api_mode": a.api_mode,
        "poll_interval_s": a.poll_interval if a.api_mode == "async" else None,
        "ready_seconds": a.ready_seconds, "warmup": warm, "levels": levels, "burst": burst,
        "resources_by_phase": sampler,
        "method": _METHOD[a.api_mode],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[serve_bench] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
