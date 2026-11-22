from __future__ import annotations

import re
from pathlib import Path

from difflet.serving.gateway.fleet import FleetConfig, FleetEntry
from difflet.serving.gateway.page import render_page

FLUX = "black-forest-labs/FLUX.1-dev"
HV = "hunyuanvideo-community/HunyuanVideo"
BASE = "http://gw:8000"


def _config(*entries):
    return FleetConfig(host="0.0.0.0", port=8000, runs_dir=Path("/runs"), models=entries)


def _flux():
    return FleetEntry(model_id=FLUX, device=0, port=8100, cache_dir=Path("/c"),
                      serve_args=("--tp-degree", "4", "--height", "1024", "--width", "1024"))


def _hv(shapes="320x512x61,320x512x33"):
    return FleetEntry(model_id=HV, device=3, port=8103, cache_dir=Path("/c"),
                      serve_args=("--tp-degree", "4", "--shapes", shapes))


def test_page_lists_exactly_config_shapes():
    page = render_page(_config(_hv()), {HV: "ready"}, base_url=BASE)
    assert re.findall(r"\d+×\d+(?:×\d+)?", page) == ["320×512×61", "320×512×33"]


def test_page_video_example_uses_width_by_height():
    page = render_page(_config(_hv()), {HV: "ready"}, base_url=BASE)
    assert "size=512x320" in page and "num_frames=61" in page
    assert f"{BASE}/v1/videos" in page and "/content" in page


def test_page_image_example_is_chat_completions():
    page = render_page(_config(_flux()), {FLUX: "ready"}, base_url=BASE)
    assert f"{BASE}/v1/chat/completions" in page
    assert "&quot;height&quot;: 1024, &quot;width&quot;: 1024" in page
    assert "1024×1024" in page


def test_page_escapes_html():
    page = render_page(_config(_flux()), {FLUX: "ready"}, base_url='http://x"><script>alert(1)')
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_page_shows_status_and_queue_line():
    page = render_page(_config(_flux(), _hv()), {FLUX: "loading"}, base_url=BASE)
    assert '<span class="badge loading"' in page
    assert '<span class="badge down"' in page  # a model missing from the statuses is down
    assert ("Each model runs one request at a time; others wait in its queue (HTTP 429 when the "
            "queue is full or times out).") in page
    assert "<title>Difflet models</title>" in page
