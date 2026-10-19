"""`difflet serve` on CUDA, end to end: start the server, wait for /ready, fetch an output.

Starts ``difflet serve --model-id <model>`` with ``DIFFLET_BACKEND=cuda`` as a
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

**FLUX.1-dev and Wan 2.2.** ``--model-id black-forest-labs/FLUX.1-dev`` goes through the
same chat path (PNG; ``--guidance-scale`` is sent when given). ``--model-id
Wan-AI/Wan2.2-T2V-A14B-Diffusers`` POSTs the multipart form of ``/v1/videos/sync`` (``size``,
``num_frames``, steps, seed, and ``guidance_scale`` / ``guidance_scale_2`` when given) and saves
the MP4. Flags left out take the model's CUDA CLI defaults (``MODEL_DEFAULTS``); Wan's startup
timeout is 1800 s, since its 126 GB snapshot reads cold from the volume at about 0.23 GB/s.
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
#: Per model: the CUDA CLI's defaults (steps, shape), what it returns, how long a cold load may
#: take, and the default output file.
MODEL_DEFAULTS: dict[str, dict] = {
    MODEL_ID: dict(
        kind="image",
        steps=50,
        height=1024,
        width=1024,
        num_frames=None,
        startup_timeout=900.0,
        output="/tmp/qwen-serve.png",
    ),
    "black-forest-labs/FLUX.1-dev": dict(
        kind="image",
        steps=28,
        height=1024,
        width=1024,
        num_frames=None,
        startup_timeout=900.0,
        output="/tmp/flux-serve.png",
    ),
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": dict(
        kind="video",
        steps=40,
        height=480,
        width=832,
        num_frames=9,
        startup_timeout=1800.0,
        output="/tmp/wan-serve.mp4",
    ),
}


def build_chat_body(
    prompt: str,
    *,
    steps: int,
    seed: int,
    height: int,
    width: int,
    model_id: str = MODEL_ID,
    guidance_scale: float | None = None,
) -> dict:
    """One chat request. ``guidance_scale`` only when given: the Qwen-Image server refuses any but
    its no-op value, the Flux server defaults it to 3.5."""
    extra = {
        "num_inference_steps": int(steps),
        "seed": int(seed),
        "height": int(height),
        "width": int(width),
    }
    if guidance_scale is not None:
        extra["guidance_scale"] = float(guidance_scale)
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "extra_body": extra,
    }


def build_video_form(
    prompt: str,
    *,
    steps: int,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    model_id: str,
    guidance_scale: float | None = None,
    guidance_scale_2: float | None = None,
) -> list[tuple[str, str]]:
    """The ``/v1/videos/sync`` multipart fields (fps is the server profile's, so not sent)."""
    fields = [
        ("model", model_id),
        ("prompt", prompt),
        ("size", f"{int(width)}x{int(height)}"),
        ("num_frames", str(int(num_frames))),
        ("num_inference_steps", str(int(steps))),
        ("seed", str(int(seed))),
    ]
    if guidance_scale is not None:
        fields.append(("guidance_scale", f"{float(guidance_scale):g}"))
    if guidance_scale_2 is not None:
        fields.append(("guidance_scale_2", f"{float(guidance_scale_2):g}"))
    return fields


def encode_multipart(fields: list[tuple[str, str]]) -> tuple[bytes, str]:
    """``multipart/form-data`` for plain text fields: the body and its Content-Type."""
    import uuid

    boundary = f"difflet-{uuid.uuid4().hex}"
    lines: list[bytes] = []
    for name, value in fields:
        lines += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            str(value).encode("utf-8"),
        ]
    lines += [f"--{boundary}--".encode(), b""]
    return b"\r\n".join(lines), f"multipart/form-data; boundary={boundary}"


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
    model_id: str = MODEL_ID,
) -> tuple[list[str], dict[str, str]]:
    cmd = [python, "-m", "difflet.cli.main", "serve", "--model-id", model_id, "--port", str(port)]
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


def post_video_sync(
    base_url: str, fields: list[tuple[str, str]], *, timeout: float, api_key: str | None = None
) -> tuple[bytes, float]:
    """One ``/v1/videos/sync`` request: the MP4 bytes and the request time."""
    data, content_type = encode_multipart(fields)
    headers = request_headers(api_key)
    headers["Content-Type"] = content_type
    req = urllib.request.Request(
        f"{base_url}/v1/videos/sync", data=data, headers=headers, method="POST"
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        mp4 = resp.read()
    return mp4, time.monotonic() - start


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The flags; any left out take ``--model-id``'s defaults (``MODEL_DEFAULTS``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=MODEL_ID, choices=sorted(MODEL_DEFAULTS))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--num-frames", type=int, default=None, help="video models only")
    ap.add_argument("--guidance-scale", type=float, default=None, help="sent only when given")
    ap.add_argument(
        "--guidance-scale-2", type=float, default=None, help="Wan 2.2's low-noise expert"
    )
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--requests", type=int, default=2, help="requests to send (2 proves residency)")
    ap.add_argument("--startup-timeout", type=float, default=None)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--dp", type=int, default=1, help="--dp-degree for the server")
    ap.add_argument(
        "--concurrent",
        type=int,
        default=0,
        help="fire this many requests at once instead of --requests serially",
    )
    ap.add_argument("--queue-timeout", type=float, default=None, help="passed to the server")
    ap.add_argument(
        "--reference", default=None, help="an output to compare every output's bytes with"
    )
    args = ap.parse_args(argv)
    defaults = MODEL_DEFAULTS[args.model_id]
    for name in ("steps", "height", "width", "num_frames", "startup_timeout", "output"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    args.kind = defaults["kind"]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cmd, env = serve_command(
        port=args.port,
        cache_dir=args.cache_dir,
        dp=args.dp,
        queue_timeout=args.queue_timeout,
        model_id=args.model_id,
    )
    api_key = os.environ.get("DIFFLET_API_KEY") or None  # the server reads the same variable
    base_url = f"http://127.0.0.1:{args.port}"
    print(f"[serve-smoke] starting: {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(cmd, env=env, start_new_session=True)
    timings: dict = {
        "model": args.model_id,
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
        shape = dict(steps=args.steps, seed=args.seed, height=args.height, width=args.width)
        if args.kind == "video":
            fields = build_video_form(
                args.prompt,
                num_frames=args.num_frames,
                model_id=args.model_id,
                guidance_scale=args.guidance_scale,
                guidance_scale_2=args.guidance_scale_2,
                **shape,
            )

            def send() -> tuple[bytes, float]:
                return post_video_sync(
                    base_url, fields, timeout=args.request_timeout, api_key=api_key
                )

        else:
            body = build_chat_body(
                args.prompt, model_id=args.model_id, guidance_scale=args.guidance_scale, **shape
            )

            def send() -> tuple[bytes, float]:
                payload, request_s = post_chat(
                    base_url, body, timeout=args.request_timeout, api_key=api_key
                )
                return fetch_image_bytes(image_url_from_response(payload)), request_s

        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)

        def record(index: int, data: bytes, request_s: float) -> None:
            path = out if index == 0 else out.with_name(f"{out.stem}-{index + 1}{out.suffix}")
            path.write_bytes(data)
            same = identical_to_reference(data, args.reference)
            timings["requests"].append(
                {
                    "request_s": round(request_s, 1),
                    # "image" as F3b / F2b recorded it; "video" for a clip.
                    args.kind: str(path),
                    "bytes": len(data),
                    "identical_to_reference": same,
                }
            )
            print(
                f"[serve-smoke] request {index + 1}: {request_s:.1f} s -> {path}"
                + ("" if same is None else f" (identical to reference: {same})"),
                flush=True,
            )

        if args.concurrent > 0:
            results, wall = fire_concurrently(lambda index: send(), args.concurrent)
            for index, (data, request_s) in enumerate(results):
                record(index, data, request_s)
            timings["wall_s"] = round(wall, 1)
            print(f"[serve-smoke] {args.concurrent} concurrent requests in {wall:.1f} s wall")
        else:
            for index in range(max(1, int(args.requests))):
                data, request_s = send()
                record(index, data, request_s)
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
