from __future__ import annotations

import json
import re
from pathlib import Path

from difflet.serving.gateway.fleet import FleetConfig, FleetEntry
from difflet.serving.gateway.page import _json, render_page

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
    assert "<script>alert(1)" not in page
    assert "&lt;script&gt;alert(1)" in page


def test_page_shows_status_and_queue_line():
    page = render_page(_config(_flux(), _hv()), {FLUX: "loading"}, base_url=BASE)
    assert '<span class="badge loading"' in page
    assert '<span class="badge down"' in page  # a model missing from the statuses is down
    assert ("Each model runs one request at a time; others wait in its queue (HTTP 429 when the "
            "queue is full or times out).") in page
    assert "<title>Difflet models</title>" in page


QWEN = "Qwen/Qwen-Image"
H3 = "MiniMaxAI/MiniMax-H3"


def _models(page):
    found = re.search(r'<script type="application/json" id="models">(.*?)</script>', page, re.S)
    return {m["id"]: m for m in json.loads(found.group(1))}


def test_page_is_a_generation_form_that_never_reloads():
    page = render_page(_config(_flux(), _hv()), {FLUX: "ready", HV: "loading"}, base_url=BASE,
                       total_runs=7)
    assert 'http-equiv="refresh"' not in page  # a reload would drop a running generation
    for marker in ('id="model"', 'id="prompt"', 'id="run"', 'data-kind="image"',
                   'data-kind="video"', "fetch('/ready')", "'/v1/videos'", 'id="runs">7<'):
        assert marker in page
    models = _models(page)
    assert models[FLUX]["kind"] == "image" and models[FLUX]["status"] == "ready"
    assert models[FLUX]["shapes"] == [[1024, 1024, None]]
    assert models[HV]["status"] == "loading"
    assert models[HV]["shapes"][0] == [320, 512, 61]


def test_page_models_carry_the_backend_defaults():
    qwen = FleetEntry(model_id=QWEN, device=1, port=8101, cache_dir=Path("/c"),
                      serve_args=("--height", "1024", "--width", "1024"))
    h3 = FleetEntry(model_id=H3, device=5, port=8105, cache_dir=Path("/c"),
                    serve_args=("--height", "256", "--width", "448", "--num-frames", "124"))
    models = _models(render_page(_config(_flux(), qwen, h3), {}, base_url=BASE))

    assert (models[FLUX]["steps"], models[FLUX]["guidance"]) == (28, 3.5)
    assert (models[QWEN]["steps"], models[QWEN]["guidance"]) == (50, None)  # takes no guidance
    assert (models[H3]["steps"], models[H3]["guidance"]) == (30, None)  # its serving metadata's


def test_page_shows_the_measured_estimate():
    flux = FleetEntry(model_id=FLUX, device=0, port=8100, cache_dir=Path("/c"),
                      serve_args=("--height", "1024", "--width", "1024"), estimate=(1.5, 0.25))
    page = render_page(_config(flux, _hv()), {FLUX: "ready"}, base_url=BASE)

    assert _models(page)[FLUX]["estimate"] == {"base": 1.5, "per_step": 0.25}
    assert _models(page)[HV]["estimate"] is None
    assert "<td>~8 s (28 steps)</td>" in page  # 1.5 + 0.25 * 28 = 8.5, FLUX's default steps
    assert 'id="progress"' in page


def test_every_id_the_script_uses_exists_once():
    page = render_page(_config(_flux(), _hv()), {FLUX: "ready"}, base_url=BASE)
    ids = re.findall(r'\bid="([^"]+)"', page)
    assert len(ids) == len(set(ids)), "duplicate element ids"
    used = set(re.findall(r"\$\('([^']+)'\)", page))
    used |= set(re.findall(r"getElementById\('([^']+)'\)", page))
    assert used <= set(ids), used - set(ids)


def test_embedded_json_cannot_end_its_script_element():
    assert "</" not in _json({"x": "</script><script>alert(1)"})
