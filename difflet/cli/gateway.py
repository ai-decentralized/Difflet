"""`difflet gateway` — one endpoint in front of several pinned `difflet serve` backends.

``up``/``down`` start and stop the backends named in the fleet file, ``status`` prints each one's
status, ``serve`` runs the gateway. docs/serving/gateway.md.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

from difflet.bench.host import HostError
from difflet.serving.gateway.fleet import FleetConfig, FleetConfigError, load_fleet
from difflet.serving.gateway.launcher import LauncherError, down, up
from difflet.serving.gateway.status import fleet_status


def run(args: argparse.Namespace) -> None:
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
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


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
