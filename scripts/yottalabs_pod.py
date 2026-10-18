#!/usr/bin/env python3
"""Yottalabs pod lifecycle for the CUDA verify sessions (plan F8; skill difflet-cuda-verify).

Stdlib only. The API key is read from ~/.config/yottalabs/api_key (override with
YOTTALABS_API_KEY_FILE) at send time and goes into the X-Api-Key header and nowhere else: it is
never printed, logged, put in an exception or shown by --dry-run.

  scripts/yottalabs_pod.py gpus [--type NVIDIA_RTX_PRO_6000_96G]
  scripts/yottalabs_pod.py create --name difflet-f9 --gpus 1 [--regions us-east-3,eu-3,eu-2]
        [--disk-gb 150] [--volume-id 469225541779435520 --mount /data]
        [--wait-for-capacity --interval 300 --max-wait 21600] [--dry-run]     -> prints the pod id
  scripts/yottalabs_pod.py wait-running ID [--timeout 900]                     -> prints "host port"
  scripts/yottalabs_pod.py ssh-config ID --alias difflet-gpu   >> ~/.ssh/config
  scripts/yottalabs_pod.py status ID | list | pause ID | resume ID | delete ID

Deploying (create, resume) needs the owner's yes. At the end of a session, pausing a pod that has
the network volume, or deleting one that cannot be paused (API code 12005, no volume), is
pre-authorised; deleting (terminating) a volume pod needs the owner's word "terminate"
(.claude/skills/difflet-cuda-backend/SKILL.md).

Exit codes: 0 ok, 1 API error (or a FAILED pod), 2 no capacity or timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

API_BASE = "https://api.yottalabs.ai"
DEFAULT_KEY_FILE = "~/.config/yottalabs/api_key"
DEFAULT_IMAGE = "yottalabsai/pytorch:2.9.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_GPU_TYPE = "NVIDIA_RTX_PRO_6000_96G"
DEFAULT_REGIONS = "us-east-3,eu-3,eu-2"
# The difflet-cache network volume lives in us-east-3; no other region can mount it.
VOLUME_REGION = "us-east-3"
DIFFLET_CACHE_VOLUME_ID = 469225541779435520
SSH_USER = "user"
SSH_KEY = "~/.ssh/yottalabs_difflet_cuda_qwen.pem"
HTTP_TIMEOUT_S = 30

CODE_OK = 10000
CODE_PARAMETER_ERROR = 10001
CODE_CANNOT_PAUSE = 12005
CODE_NO_CAPACITY = 12014

# A resume the API never takes leaves the pod PAUSED; past this many seconds that is no capacity.
RESUME_GRACE_S = 60

Transport = Callable[[str, str, Any], dict]


class PodToolError(Exception):
    """A failure reported to the user as one line; exit code ``exit_code``."""

    exit_code = 1


class ApiError(PodToolError):
    def __init__(self, code: Any, message: str):
        super().__init__(f"API error {code}: {message}")
        self.code = code
        self.message = message


class NoCapacity(ApiError):
    exit_code = 2


class PodTimeout(PodToolError):
    exit_code = 2


class PodFailed(PodToolError):
    pass


# --- transport ------------------------------------------------------------------------------


def read_api_key(path: str | os.PathLike) -> str:
    p = Path(path).expanduser()
    try:
        key = p.read_text().strip()
    except OSError as exc:
        raise PodToolError(f"cannot read the API key file {p}: {exc.strerror}") from None
    if not key:
        raise PodToolError(f"the API key file {p} is empty")
    return key


class UrllibTransport:
    """``transport(method, path, body) -> envelope dict`` over urllib. The key is read from the
    file on every call and scrubbed from anything the server sends back before it is parsed."""

    def __init__(
        self,
        key_file: str | os.PathLike | None = None,
        base: str = API_BASE,
        opener: Callable[..., Any] = urllib.request.urlopen,
        timeout: float = HTTP_TIMEOUT_S,
    ):
        self.key_file = key_file or os.environ.get("YOTTALABS_API_KEY_FILE", DEFAULT_KEY_FILE)
        self.base = base
        self.opener = opener
        self.timeout = timeout

    def __call__(self, method: str, path: str, body: Any = None) -> dict:
        key = read_api_key(self.key_file)

        def scrub(text: str) -> str:
            return text.replace(key, "<redacted>")

        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("X-Api-Key", key)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "difflet-yottalabs-pod/1")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with self.opener(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            # The API answers logical errors in a JSON envelope, sometimes with a 4xx status.
            try:
                text = scrub(exc.read().decode("utf-8", "replace"))
            except Exception:
                text = ""
            try:
                envelope = json.loads(text)
            except ValueError:
                envelope = None
            if isinstance(envelope, dict) and "code" in envelope:
                return envelope
            raise ApiError(f"HTTP {exc.code}", f"{method} {path}: {scrub(text[:200])}") from None
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ApiError("network", scrub(f"{method} {self.base}{path}: {reason}")) from None
        text = scrub(raw.decode("utf-8", "replace"))
        try:
            envelope = json.loads(text)
        except ValueError:
            raise ApiError("bad-response", f"{method} {path}: not JSON: {text[:200]}") from None
        if not isinstance(envelope, dict):
            raise ApiError("bad-response", f"{method} {path}: not an envelope")
        return envelope


def api(transport: Transport, method: str, path: str, body: Any = None) -> Any:
    """One call; returns the envelope's ``data`` or raises (12014 as :class:`NoCapacity`)."""
    envelope = transport(method, path, body)
    code = envelope.get("code")
    if code == CODE_OK:
        return envelope.get("data")
    message = str(envelope.get("message", ""))
    if code == CODE_NO_CAPACITY:
        raise NoCapacity(code, message)
    raise ApiError(code, message)


# --- pure helpers ---------------------------------------------------------------------------


def build_create_body(
    name: str,
    gpus: int,
    region: str,
    disk_gb: int,
    volume_id: int | None,
    mount: str,
    image: str,
    gpu_type: str = DEFAULT_GPU_TYPE,
) -> dict:
    """The ``POST /v2/pods`` body for one region: the network volume only in the region it
    lives in (and then no container disk size); everywhere else a container disk."""
    body: dict[str, Any] = {
        "name": name,
        "regions": [region],
        "image": image,
        "gpuType": gpu_type,
        "gpuCount": gpus,
        "expose": [{"port": 22, "protocol": "SSH"}],
    }
    if volume_id is not None and region == VOLUME_REGION:
        body["persistentVolumes"] = [
            {"volumeId": volume_id, "mountPath": mount, "needBackup": False}
        ]
    else:
        body["containerVolumeInGb"] = disk_gb
    return body


def _pool_sells(pool: dict, gpu_count: int | None) -> bool:
    if gpu_count is None:
        return True
    options = pool.get("gpuCountOptions")
    return not options or gpu_count in options


def pick_region(
    pools: Iterable[dict], preferred: Iterable[str], gpu_count: int | None = None
) -> str | None:
    """The first preferred region with an AVAILABLE pool (that sells ``gpu_count`` cards)."""
    available = {
        p.get("region")
        for p in pools
        if p.get("status") == "AVAILABLE" and _pool_sells(p, gpu_count)
    }
    for region in preferred:
        if region in available:
            return region
    return None


def ssh_endpoint(pod_json: dict | None) -> tuple[str, int] | None:
    """``(host, proxyPort)`` of the pod's SSH expose once both are set, else ``None``."""
    if not pod_json:
        return None
    for item in pod_json.get("expose") or []:
        if str(item.get("protocol", "")).upper() != "SSH" and item.get("port") != 22:
            continue
        host, port = item.get("host"), item.get("proxyPort")
        if host and port:
            return str(host), int(port)
    return None


def pod_status(pod_json: dict | None) -> str:
    """The record's status; ``GONE`` once the API returns ``data: null`` (after a delete)."""
    if not pod_json:
        return "GONE"
    return str(pod_json.get("status") or "UNKNOWN")


def pod_items(data: Any) -> list[dict]:
    """``GET /v2/pods``'s ``data``: ``{"items": [...]}`` or a bare list."""
    if isinstance(data, dict):
        return list(data.get("items") or [])
    if isinstance(data, list):
        return data
    return []


def next_poll_decision(status: str, target: str, previous: str | None = None) -> str:
    """``done`` / ``wait`` / ``no_capacity`` / ``failed`` for one observation while waiting for
    ``target`` (RUNNING, PAUSED or GONE). A resume that went INITIALIZING → PAUSED found no
    capacity; PAUSED before any INITIALIZING is the resume not yet taken."""
    if status == target:
        return "done"
    if status == "FAILED":
        return "failed"
    if target == "GONE":
        return "wait"
    if status in ("TERMINATING", "GONE"):
        return "failed"
    if target == "RUNNING" and status == "PAUSED" and previous == "INITIALIZING":
        return "no_capacity"
    return "wait"


# --- commands -------------------------------------------------------------------------------


class Ctx:
    def __init__(self, transport, sleep, clock, stdout, stderr, key_file):
        self.transport = transport
        self.sleep = sleep
        self.clock = clock
        self.stdout = stdout
        self.stderr = stderr
        self.key_file = key_file

    def out(self, line: str = "") -> None:
        print(line, file=self.stdout, flush=True)

    def log(self, line: str) -> None:
        print(f"[yottalabs {time.strftime('%H:%M:%S')}] {line}", file=self.stderr, flush=True)


def get_pod(ctx: Ctx, pod_id: str) -> dict | None:
    return api(ctx.transport, "GET", f"/v2/pods/{pod_id}")


def poll(
    ctx: Ctx,
    pod_id: str,
    target: str,
    timeout: float,
    interval: float,
    ready: Callable[[dict | None], bool] | None = None,
    resume_grace: float | None = None,
) -> dict | None:
    start = ctx.clock()
    previous = None
    while True:
        record = get_pod(ctx, pod_id)
        status = pod_status(record)
        if status != previous:
            ctx.log(f"{pod_id}: {status}")
        decision = next_poll_decision(status, target, previous)
        elapsed = ctx.clock() - start
        if decision == "done" and (ready is None or ready(record)):
            return record
        if decision == "no_capacity":
            raise NoCapacity(CODE_NO_CAPACITY, f"{pod_id}: INITIALIZING → PAUSED: no capacity")
        if decision == "failed":
            raise PodFailed(f"{pod_id} is {status} while waiting for {target}")
        if resume_grace is not None and status == "PAUSED" and elapsed >= resume_grace:
            raise NoCapacity(
                CODE_NO_CAPACITY,
                f"{pod_id}: still PAUSED {elapsed:.0f} s after resume: no capacity",
            )
        if elapsed >= timeout:
            raise PodTimeout(f"timeout: {pod_id} is {status} after {elapsed:.0f} s (want {target})")
        ctx.sleep(interval)
        previous = status


def _row(cells: Iterable[Any], widths: Iterable[int]) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()


def cmd_gpus(args, ctx: Ctx) -> int:
    pools = [p for p in api(ctx.transport, "GET", "/v2/gpus") or [] if isinstance(p, dict)]
    if args.type:
        pools = [p for p in pools if p.get("gpuType") == args.type]
    widths = (26, 14, 12, 10, 6, 8, 8)
    ctx.out(_row(("gpuType", "region", "status", "$/GPU/h", "max", "counts", "diskGB"), widths))
    for p in pools:
        counts = ",".join(str(c) for c in p.get("gpuCountOptions") or [])
        ctx.out(
            _row(
                (
                    p.get("gpuType", "?"),
                    p.get("region", "?"),
                    p.get("status", "?"),
                    p.get("pricePerGpuPerHour", "?"),
                    p.get("maxGpuCount", "?"),
                    counts or "-",
                    p.get("containerDiskLimitInGb", "?"),
                ),
                widths,
            )
        )
    return 0


def cmd_create(args, ctx: Ctx) -> int:
    preferred = [r.strip() for r in args.regions.split(",") if r.strip()]
    if not preferred:
        raise PodToolError("--regions is empty")

    def body_for(region: str) -> dict:
        return build_create_body(
            args.name,
            args.gpus,
            region,
            args.disk_gb,
            args.volume_id,
            args.mount,
            args.image,
            args.gpu_type,
        )

    if args.dry_run:
        ctx.out(
            f"# dry run: nothing sent. The first preferred region whose {args.gpu_type} pool is "
            f"AVAILABLE (GET {API_BASE}/v2/gpus) gets its request; X-Api-Key is read from "
            f"{ctx.key_file} at send time and not shown."
        )
        for region in preferred:
            ctx.out(f"POST {API_BASE}/v2/pods  # region {region}")
            ctx.out(json.dumps(body_for(region), indent=2))
        return 0

    start = ctx.clock()
    while True:
        try:
            listing = api(ctx.transport, "GET", "/v2/gpus") or []
        except ApiError as exc:
            # The listing is read-only, so the watcher rides out a failed read; the POST below
            # is never retried after an error that is not 12014 (it may have created a pod).
            if not args.wait_for_capacity:
                raise
            ctx.log(f"pool listing failed, retrying: {exc}")
            listing = None
        pools = [
            p for p in listing or [] if isinstance(p, dict) and p.get("gpuType") == args.gpu_type
        ]
        region = pick_region(pools, preferred, gpu_count=args.gpus)
        if listing is not None and region is None:
            ctx.log(
                f"no AVAILABLE {args.gpu_type} x{args.gpus} pool in {','.join(preferred)} "
                f"(no capacity)"
            )
        elif region is not None:
            if args.volume_id is not None and region != VOLUME_REGION:
                ctx.log(
                    f"the volume lives in {VOLUME_REGION}; {region} gets a "
                    f"{args.disk_gb} GB container disk instead"
                )
            limit = next(
                (p.get("containerDiskLimitInGb") for p in pools if p.get("region") == region),
                None,
            )
            body = body_for(region)
            if "containerVolumeInGb" in body and isinstance(limit, int) and args.disk_gb > limit:
                ctx.log(f"warning: --disk-gb {args.disk_gb} exceeds {region}'s limit {limit} GB")
            try:
                record = api(ctx.transport, "POST", "/v2/pods", body)
            except NoCapacity as exc:
                ctx.log(f"{region}: {exc.message} (code {exc.code}; the pool flag can lag)")
            else:
                pod_id = (record or {}).get("id")
                if pod_id is None:
                    raise ApiError("bad-response", "POST /v2/pods returned no pod id")
                ctx.log(
                    f"created {pod_id} in {region}: {args.gpu_type} x{args.gpus}, "
                    f"status {pod_status(record)}"
                )
                ctx.out(str(pod_id))
                return 0
        if not args.wait_for_capacity:
            return 2
        elapsed = ctx.clock() - start
        if elapsed + args.interval > args.max_wait:
            ctx.log(f"no capacity after {elapsed:.0f} s (--max-wait {args.max_wait}); giving up")
            return 2
        ctx.log(f"next check in {args.interval} s ({elapsed:.0f} s waited)")
        ctx.sleep(args.interval)


def _endpoint_line(record: dict | None) -> str:
    ep = ssh_endpoint(record)
    return f"{ep[0]} {ep[1]}" if ep else "-"


def cmd_status(args, ctx: Ctx) -> int:
    record = get_pod(ctx, args.id)
    if record is None:
        ctx.out(f"id      {args.id}")
        ctx.out("status  GONE")
        return 0
    ctx.out(f"id      {record.get('id', args.id)}")
    ctx.out(f"status  {pod_status(record)}")
    ctx.out(f"region  {record.get('region', '?')}")
    ctx.out(f"gpus    {record.get('gpuCount', '?')}")
    ctx.out(f"price   {record.get('singleCardPrice', '?')} per GPU-hour")
    ctx.out(f"disk    {record.get('containerVolumeInGb', '?')} GB")
    ctx.out(f"volumes {json.dumps(record.get('volumes') or [])}")
    ctx.out(f"ssh     {_endpoint_line(record)}")
    return 0


def cmd_list(args, ctx: Ctx) -> int:
    widths = (20, 24, 13, 14, 4)
    ctx.out(_row(("id", "name", "status", "region", "gpus"), widths))
    for p in pod_items(api(ctx.transport, "GET", "/v2/pods")):
        ctx.out(
            _row(
                (
                    p.get("id", "?"),
                    p.get("name", "?"),
                    p.get("status", "?"),
                    p.get("region", "?"),
                    p.get("gpuCount", "?"),
                ),
                widths,
            )
        )
    return 0


def cmd_wait_running(args, ctx: Ctx) -> int:
    record = poll(
        ctx,
        args.id,
        "RUNNING",
        args.timeout,
        args.interval,
        ready=lambda r: ssh_endpoint(r) is not None,
    )
    ctx.out(_endpoint_line(record))
    return 0


def cmd_pause(args, ctx: Ctx) -> int:
    try:
        api(ctx.transport, "POST", f"/v2/pods/{args.id}/pause")
    except ApiError as exc:
        if exc.code == CODE_CANNOT_PAUSE:
            raise ApiError(
                exc.code,
                f"{exc.message} (a pod without a network volume cannot be paused: "
                f"delete it instead)",
            ) from None
        raise
    poll(ctx, args.id, "PAUSED", args.timeout, args.interval)
    ctx.out(f"{args.id} PAUSED")
    return 0


def cmd_resume(args, ctx: Ctx) -> int:
    api(ctx.transport, "POST", f"/v2/pods/{args.id}/resume")
    record = poll(
        ctx,
        args.id,
        "RUNNING",
        args.timeout,
        args.interval,
        ready=lambda r: ssh_endpoint(r) is not None,
        resume_grace=RESUME_GRACE_S,
    )
    ctx.out(_endpoint_line(record))
    return 0


def cmd_delete(args, ctx: Ctx) -> int:
    api(ctx.transport, "DELETE", f"/v2/pods/{args.id}")
    poll(ctx, args.id, "GONE", args.timeout, args.interval)
    ctx.out(f"{args.id} gone")
    return 0


def cmd_ssh_config(args, ctx: Ctx) -> int:
    record = get_pod(ctx, args.id)
    ep = ssh_endpoint(record)
    if ep is None:
        ctx.log(f"{args.id} has no SSH endpoint yet (status {pod_status(record)})")
        return 2
    ctx.out(f"Host {args.alias}")
    ctx.out(f"  HostName {ep[0]}")
    ctx.out(f"  Port {ep[1]}")
    ctx.out(f"  User {SSH_USER}")
    ctx.out(f"  IdentityFile {SSH_KEY}")
    ctx.out("  StrictHostKeyChecking accept-new")
    ctx.out("  IdentitiesOnly yes")
    ctx.out("  ServerAliveInterval 30")
    return 0


# --- CLI ------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="yottalabs_pod.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("gpus", help="the GPU pools: region, status, price, max count, disk limit")
    p.add_argument("--type", default=None, help=f"only this gpuType (e.g. {DEFAULT_GPU_TYPE})")
    p.set_defaults(func=cmd_gpus)

    p = sub.add_parser("create", help="create a pod in the first AVAILABLE preferred region")
    p.add_argument("--name", required=True)
    p.add_argument("--gpus", type=int, choices=(1, 2), required=True)
    p.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    p.add_argument(
        "--regions",
        default=DEFAULT_REGIONS,
        help="preference order (default %(default)s); the eu PRO 6000 pools sell 2-GPU nodes only",
    )
    p.add_argument("--disk-gb", type=int, default=150, help="container disk when no volume")
    p.add_argument(
        "--volume-id",
        type=int,
        default=None,
        help=f"network volume, mounted only in {VOLUME_REGION} where it lives "
        f"(difflet-cache: {DIFFLET_CACHE_VOLUME_ID})",
    )
    p.add_argument("--mount", default="/data")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument(
        "--wait-for-capacity",
        action="store_true",
        help="poll the pools every --interval s until one is AVAILABLE, then create once",
    )
    p.add_argument("--interval", type=float, default=300)
    p.add_argument("--max-wait", type=float, default=21600)
    p.add_argument("--dry-run", action="store_true", help="print the request(s), send nothing")
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("status", help="one pod's status, region, price and SSH endpoint")
    p.add_argument("id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("list", help="the account's pods")
    p.set_defaults(func=cmd_list)

    def lifecycle(name, func, helptext, timeout, interval):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("id")
        q.add_argument("--timeout", type=float, default=timeout)
        q.add_argument("--interval", type=float, default=interval)
        q.set_defaults(func=func)
        return q

    lifecycle(
        "wait-running",
        cmd_wait_running,
        "wait for RUNNING with an SSH endpoint; prints 'host port'",
        900,
        10,
    )
    lifecycle("pause", cmd_pause, "pause (volume pods only) and wait for PAUSED", 300, 10)
    lifecycle(
        "resume",
        cmd_resume,
        "resume and wait for RUNNING; INITIALIZING → PAUSED is no capacity",
        900,
        5,
    )
    lifecycle(
        "delete",
        cmd_delete,
        "terminate and wait until the pod is gone (a volume pod: only on the owner's 'terminate')",
        300,
        5,
    )

    p = sub.add_parser("ssh-config", help="the ~/.ssh/config Host block for a RUNNING pod")
    p.add_argument("id")
    p.add_argument("--alias", required=True)
    p.set_defaults(func=cmd_ssh_config)
    return ap


def main(
    argv: list[str] | None = None,
    *,
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    stdout=None,
    stderr=None,
) -> int:
    args = build_parser().parse_args(argv)
    key_file = os.environ.get("YOTTALABS_API_KEY_FILE", DEFAULT_KEY_FILE)
    ctx = Ctx(
        transport if transport is not None else UrllibTransport(key_file),
        sleep,
        clock,
        stdout or sys.stdout,
        stderr or sys.stderr,
        key_file,
    )
    try:
        return args.func(args, ctx)
    except PodToolError as exc:
        ctx.log(str(exc))
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
