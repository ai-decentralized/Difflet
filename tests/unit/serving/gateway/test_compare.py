from __future__ import annotations

import json
import re

from difflet.serving.gateway.compare import LEVEL, QUALITY, SPEED, render_compare


def test_the_page_claims_follow_from_its_data():
    # The page computes these in the browser; a changed figure that flips one must be deliberate.
    flux = SPEED[0]
    assert round(flux["rtx"]["compiled"][0] / flux["trn3"], 2) == 1.13
    with_rtx = [m for m in SPEED if m["rtx"]]
    assert [m["model"] for m in SPEED if not m["rtx"]] == ["Wan 2.1 14B", "LTX-2"]
    for mode, wins in (("compiled", 3), ("eager", 4)):
        assert sum(m["rtx"][mode][0] / m["trn3"] >= 1 + LEVEL["speed"] for m in with_rtx) == wins
    rows = [q for q in QUALITY if q["summary"]]
    assert len(rows) == 7
    for i, metric in ((0, "psnr"), (1, "ssim")):
        assert sum(q["trn3"][i] - q["rtx"][i] > -LEVEL[metric] for q in rows) == 4


def test_every_rtx_figure_says_how_it_was_obtained():
    for m in SPEED:
        if m["rtx"]:
            assert {mark for _, mark in m["rtx"].values()} <= {"", "≈", "≤"}, m["model"]
            assert m["source"]


def test_every_id_the_script_uses_exists_once():
    page = render_compare()
    ids = re.findall(r'\bid="([^"]+)"', page)
    assert len(ids) == len(set(ids)), "duplicate element ids"
    used = set(re.findall(r"\$\('([^']+)'\)", page))
    used |= set(re.findall(r"getElementById\('([^']+)'\)", page))
    assert used <= set(ids), used - set(ids)


def test_embedded_data_round_trips():
    page = render_compare()
    blob = re.search(r'<script type="application/json" id="compare-data">(.*?)</script>', page, re.S)
    data = json.loads(blob.group(1))
    assert data["speed"] == json.loads(json.dumps(SPEED))
    assert data["quality"] == json.loads(json.dumps(QUALITY))
