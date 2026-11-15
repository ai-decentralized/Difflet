"""scripts/mx_attention/task6.py: which files a key's quality reads from run_arms.sh's layout (the
first request's latents in batch mode, a non-finite latent caught, determinism over the arm's two
requests and its second process), with the comparisons themselves stubbed."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "mx_attention"))
import task6  # noqa: E402


def _arm(root: Path, arm: str, md5s: tuple[str, str, str], latents: torch.Tensor) -> None:
    d = root / "wan21_official" / arm
    for name, body in zip(("timed_1", "timed_2", "gen2"), md5s):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.mp4").write_text(body)
    (d / "work_timed" / "w0").mkdir(parents=True)
    torch.save(latents, d / "work_timed" / "w0" / "latents_req0000.pt")
    torch.save(latents + 1, d / "work_timed" / "w0" / "latents_req0001.pt")
    (d / "work_gen2").mkdir()
    torch.save(latents, d / "work_gen2" / "latents.pt")


def test_quality_reads_the_first_requests_latents_and_checks_determinism(tmp_path, monkeypatch):
    monkeypatch.setattr(task6, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(task6.cdr, "compare_runs", lambda *a: calls.append(a) or {})
    monkeypatch.setattr(task6, "_frames", lambda paths, out: None)
    _arm(tmp_path, "bf16", ("a", "a", "a"), torch.zeros(2, 3))
    bad = torch.zeros(2, 3)
    bad[0, 0] = float("nan")
    _arm(tmp_path, "both", ("b", "b", "c"), bad)
    q = task6.quality("wan21_official", "both")
    [(lat_b, lat_a, out_b, out_a)] = calls  # no fp32 reference at Wan's 81 frames
    assert (lat_b.name, lat_a.name, out_a.name) == ("latents_req0000.pt",) * 2 + ("timed_1.mp4",)
    assert "/bf16/" in str(lat_b) and "/both/" in str(lat_a)
    assert q["deterministic"] == {"bf16": True, "both": False}  # gen2 differs
    assert q["arm_latents_finite"] == {"timed": False, "gen2": False}
