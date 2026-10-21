"""`difflet serve` on CUDA, end to end: start the server, wait for /ready, fetch one image.

Starts ``difflet serve --model-id Qwen/Qwen-Image`` with ``DIFFLET_BACKEND=cuda`` as a
subprocess, polls ``GET /ready`` (the worker loads the model and runs its 4-step startup smoke
first), POSTs one ``/v1/chat/completions`` request, saves the PNG, and writes
``serve_timings.json`` next to it (``ready_s``, ``request_s``). With ``--requests 2`` the second
request proves the model stays resident (its ``request_s`` carries no load). **Two GPUs:**
``--dp N`` starts the server with ``--dp-degree N`` (one resident worker per GPU) and
``--concurrent K`` fires K requests at once from threads — on ``--dp 2`` two requests finish in
one request-time, a third queues (raise ``--queue-timeout`` above a request); ``--reference PNG``
compares every image's bytes with a serial run's (same seed → identical), recorded per request
as ``identical_to_reference``; ``wall_s`` is the concurrent batch's wall time. The snapshot must
already be in the HF cache (``hf download Qwen/Qwen-Image``): ``ready_s`` is load + the 4-step
startup smoke, and the 900 s startup timeout is not sized for a 54 GB download. Honours
``DIFFLET_API_KEY`` and an S3 artifact store configured through the repo-root ``.env``.

    DIFFLET_BACKEND=cuda python scripts/cuda_serve_smoke.py --output /tmp/qwen-serve.png --steps 50
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL_ID = "Qwen/Qwen-Image"
DEFAULT_PROMPT = "a photorealistic cat sitting in a sunlit garden"


def build_chat_body(prompt: str, *, steps: int, seed: int, height: int, width: int) -> dict:
    """One chat request; no guidance_scale (the CUDA server refuses any but the no-op value)."""
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "extra_body": {
            "num_inference_steps": int(steps),
            "seed": int(seed),
            "height": int(height),
            "width": int(width),
        },
    }


def decode_image_data_url(url: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not url.startswith(prefix):
        raise ValueError(f"expected a PNG data URL, got {url[:40]!r}")
    return base64.b64decode(url[len(prefix) :])


def fetch_image_bytes(url: str, *, opener=urllib.request.urlopen, timeout: float = 60.0) -> bytes:
    """The PNG behind a chat response: inline data URL, or the http(s) URL an S3 artifact
    store returns when the server's ``.env`` configures one."""
    if url.startswith("data:"):
        return decode_image_data_url(url)
    if url.startswith(("http://", "https://")):
        with opener(url, timeout=timeout) as resp:
            return resp.read()
    raise ValueError(f"unsupported image URL {url[:40]!r}")


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def image_url_from_response(body: dict) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected chat response: {body!r}") from exc
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            return str(part["image_url"]["url"])
    raise ValueError(f"chat response carries no image_url part: {content!r}")


def serve_command(
    *,
    port: int,
    cache_dir: str | None,
    python: str = sys.executable,
    dp: int = 1,
    queue_timeout: float | None = None,
) -> tuple[list[str], dict[str, str]]:
    cmd = [python, "-m", "difflet.cli.main", "serve", "--model-id", MODEL_ID, "--port", str(port)]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]
    if int(dp) > 1:
        cmd += ["--dp-degree", str(int(dp))]
    if queue_timeout is not None:
        cmd += ["--queue-timeout", f"{queue_timeout:g}"]
    return cmd, {**os.environ, "DIFFLET_BACKEND": "cuda"}


def fire_concurrently(poster, count: int) -> tuple[list, float]:
    """``poster(index)`` for ``count`` indices at once, one thread each; the results in index
    order and the batch's wall time. The first failure is raised after every thread returns."""
    from concurrent.futures import ThreadPoolExecutor

    count = max(1, int(count))
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(poster, index) for index in range(count)]
        results = [f.result() for f in futures]
    return results, time.monotonic() - start


def identical_to_reference(png: bytes, reference: "pathlib.Path | str | None") -> bool | None:
    """Byte equality with a reference image; None when no reference was given."""
    if reference is None:
        return None
    return pathlib.Path(reference).read_bytes() == png


def wait_for_ready(base_url: str, *, timeout: float, process: subprocess.Popen) -> float:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"difflet serve exited with code {process.returncode} before ready")
        try:
            with urllib.request.urlopen(f"{base_url}/ready", timeout=5) as resp:
                if resp.status == 200:
                    return time.monotonic() - start
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f"difflet serve not ready after {timeout:.0f} s")


def post_chat(
    base_url: str, body: dict, *, timeout: float, api_key: str | None = None
) -> tuple[dict, float]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers=request_headers(api_key),
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload, time.monotonic() - start


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--output", default="/tmp/qwen-serve.png")
    ap.add_argument("--requests", type=int, default=2, help="requests to send (2 proves residency)")
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--dp", type=int, default=1, help="--dp-degree for the server")
    ap.add_argument(
        "--concurrent",
        type=int,
        default=0,
        help="fire this many requests at once instead of --requests serially",
    )
    ap.add_argument("--queue-timeout", type=float, default=None, help="passed to the server")
    ap.add_argument("--reference", default=None, help="PNG to compare every image's bytes with")
    args = ap.parse_args(argv)

    cmd, env = serve_command(
        port=args.port, cache_dir=args.cache_dir, dp=args.dp, queue_timeout=args.queue_timeout
    )
    api_key = os.environ.get("DIFFLET_API_KEY") or None  # the server reads the same variable
    base_url = f"http://127.0.0.1:{args.port}"
    print(f"[serve-smoke] starting: {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(cmd, env=env, start_new_session=True)
    timings: dict = {
        "model": MODEL_ID,
        "steps": args.steps,
        "seed": args.seed,
        "dp": int(args.dp),
        "concurrent": int(args.concurrent),
        "reference": args.reference,
        "requests": [],
    }
    try:
        timings["ready_s"] = round(
            wait_for_ready(base_url, timeout=args.startup_timeout, process=process), 1
        )
        print(f"[serve-smoke] ready in {timings['ready_s']} s", flush=True)
        body = build_chat_body(
            args.prompt, steps=args.steps, seed=args.seed, height=args.height, width=args.width
        )
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)

        def record(index: int, payload: dict, request_s: float) -> None:
            png = fetch_image_bytes(image_url_from_response(payload))
            path = out if index == 0 else out.with_name(f"{out.stem}-{index + 1}{out.suffix}")
            path.write_bytes(png)
            same = identical_to_reference(png, args.reference)
            timings["requests"].append(
                {
                    "request_s": round(request_s, 1),
                    "image": str(path),
                    "identical_to_reference": same,
                }
            )
            print(
                f"[serve-smoke] request {index + 1}: {request_s:.1f} s -> {path}"
                + ("" if same is None else f" (identical to reference: {same})"),
                flush=True,
            )

        if args.concurrent > 0:

            def poster(index: int):
                return post_chat(base_url, body, timeout=args.request_timeout, api_key=api_key)

            results, wall = fire_concurrently(poster, args.concurrent)
            for index, (payload, request_s) in enumerate(results):
                record(index, payload, request_s)
            timings["wall_s"] = round(wall, 1)
            print(f"[serve-smoke] {args.concurrent} concurrent requests in {wall:.1f} s wall")
        else:
            for index in range(max(1, int(args.requests))):
                payload, request_s = post_chat(
                    base_url, body, timeout=args.request_timeout, api_key=api_key
                )
                record(index, payload, request_s)
        timings["request_s"] = timings["requests"][0]["request_s"]
        out.with_name("serve_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        print(json.dumps(timings, indent=2))
        return 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())
