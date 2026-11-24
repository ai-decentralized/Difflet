"""`difflet gateway` — one endpoint in front of several pinned `difflet serve` backends.

``up``/``down`` start and stop the backends named in the fleet file, ``status`` prints each one's
status, ``serve`` runs the gateway, ``page`` writes the page as a static file to host elsewhere.
docs/serving/gateway.md.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import NoReturn

import httpx

from difflet.bench.host import HostError
from difflet.serving.gateway.fleet import FleetConfig, FleetConfigError, load_fleet
from difflet.serving.gateway.launcher import LauncherError, down, up
from difflet.serving.gateway.status import fleet_status


def run(args: argparse.Namespace) -> None:
    if args.gateway_action == "page":
        _write_page(args.api_base, args.out)
        return
    if not args.config:
        _fail(f"gateway {args.gateway_action} needs --config FLEET_FILE")
    try:
        config = load_fleet(args.config)
        if args.gateway_action == "up":
            up(config)
        elif args.gateway_action == "down":
            down(config)
        elif args.gateway_action == "status":
            _print_status(config)
        else:
            _serve(config, args.api_key or os.environ.get("DIFFLET_API_KEY") or None)
    except (FleetConfigError, LauncherError, HostError) as exc:
        _fail(str(exc))


def _fail(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _write_page(api_base: str | None, out: str | None) -> None:
    """The page as one static file that calls the gateway at ``api_base`` (it must allow the
    page's origin in ``gateway.cors_origins``)."""
    if not (api_base and re.fullmatch(r"https?://\S+", api_base)) or not out:
        _fail("gateway page needs --api-base http(s)://GATEWAY and --out PATH")
    from difflet.serving.gateway.page import render_page

    try:
        Path(out).write_text(render_page(None, {}, api_base=api_base))
    except OSError as exc:
        _fail(f"cannot write {out}: {exc}")
    print(f"wrote {out} (calls {api_base.rstrip('/')})")


def _serve(config: FleetConfig, api_key: str | None) -> None:
    import uvicorn

    from difflet.serving.gateway.app import create_gateway_app

    uvicorn.run(create_gateway_app(config, api_key=api_key), host=config.host, port=config.port,
                workers=1)


def _print_status(config: FleetConfig) -> None:
    async def probe():
        async with httpx.AsyncClient() as client:
            return await fleet_status(config, client)

    found = asyncio.run(probe())
    rows = [("model", "device", "port", "status")] + [
        (e.model_id, str(e.device), str(e.port), found[e.model_id]) for e in config.models]
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
