"""Each backend's status from its pid file and its ``/ready`` (design §6.5).

A loading backend refuses connections (its port opens only after load and the smoke pass), so
HTTP alone cannot tell loading from down; the launcher's pid file can.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import httpx

from difflet.serving.gateway.fleet import FleetConfig, FleetEntry
from difflet.serving.gateway.launcher import read_pid

Status = Literal["ready", "loading", "recovering", "down"]


async def backend_status(entry: FleetEntry, pid_file: Path, client: httpx.AsyncClient) -> Status:
    if read_pid(pid_file) is None:
        return "down"
    try:
        resp = await client.get(f"{entry.base_url}/ready", timeout=2.0)
    except httpx.ConnectError:
        return "loading"
    except httpx.HTTPError:
        return "down"
    if resp.status_code == 503:
        return "recovering"
    if resp.status_code != 200:
        return "down"
    try:
        body = resp.json()
    except ValueError:
        return "down"
    # Another model on this port means a wrong config entry: never route to it as ready.
    return "ready" if isinstance(body, dict) and body.get("model") == entry.model_id else "down"


async def fleet_status(config: FleetConfig, client: httpx.AsyncClient) -> dict[str, Status]:
    found = await asyncio.gather(*(backend_status(e, config.pid_file(e), client)
                                   for e in config.models))
    return {e.model_id: s for e, s in zip(config.models, found)}
