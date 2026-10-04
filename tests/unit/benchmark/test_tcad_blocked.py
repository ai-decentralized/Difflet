"""A tp4tcad cell that OOMs on device is recorded as BLOCKED (a diagnosed device
limit), not a crash or a by-design N/A: tcad_prep detects the HBM signature,
models.write_blocked_cell persists the diagnosis, and campaign_summary renders
it in both the feature table and the TeaCache table."""
from __future__ import annotations

import json

import pytest

from benchmark import campaign_summary as cs
from benchmark import tcad_prep
from benchmark.models import cell_is_blocked, resolve, write_blocked_cell

# A trimmed HunyuanVideo collect log: the Neuron runtime's HBM-OOM dump.
OOM_LOG = """\
INFO:Neuron:Presharded file read: 0.34s for 4 shard(s) (44381.9 MB total)
2026-Sep-17 18:39:50 ERROR  TDRV:dmem_alloc_internal  Failed to allocate DEVICE memory (14155776 bytes): memory allocation failed (ret=-12)
2026-Sep-17 18:39:50 ERROR  TDRV:log_dev_mem  Failed to allocate 13.500MB (alignment: none, usage: tensors) on ND 0:NC 0
              |  TOTAL   |   Code   |Constants | Tensors  |Scratchpad|
ND 0 HBM 0    | 23.957GB |109.148MB | 2.955MB  | 22.037GB | 1.125GB  |
  \\_NC 0      | 23.568GB | 62.020MB | 1.493MB  | 22.037GB | 1.125GB  |
2026-Sep-17 18:39:52 ERROR   NRT:nrt_tensor_allocate  Failed to allocate nrt tensor UNNAMED
terminate called after throwing an instance of 'c10::Error'
"""

CLEAN_LOG = "[calibrate] prompt 0: 19 pairs, finite=True\n[teacache] stats: {'skipped_steps': 5}\n"


def test_diagnose_oom_recognises_the_hbm_signature(tmp_path):
    log = tmp_path / "collect.log"
    log.write_text(OOM_LOG)
    blocked = tcad_prep._diagnose_oom(log)
    assert blocked is not None
    assert "HBM exhausted" in blocked.reason
    assert "22.037GB" in blocked.evidence and "13.500MB" in blocked.evidence
    assert blocked.extra["blocked_class"] == "hbm_oom"
    assert blocked.extra["peak_hbm_tensors_gb"] == 22.037
    log.write_text(CLEAN_LOG)
    assert tcad_prep._diagnose_oom(log) is None


def test_write_and_detect_blocked_cell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cell_is_blocked("hunyuan_video", "tp4tcad") is False
    p = write_blocked_cell("hunyuan_video", "tp4tcad",
                           reason="HBM exhausted: DiT + probe do not fit",
                           evidence="logs/hunyuan_video_collect0.log — ret=-12",
                           extra={"peak_hbm_tensors_gb": 22.037, "blocked_class": "hbm_oom"})
    d = json.loads(open(p).read())
    assert d["status"] == "blocked" and d["config"] == "tp4tcad"
    assert d["blocked_class"] == "hbm_oom" and d["peak_hbm_tensors_gb"] == 22.037
    # the topology record is still the tp4 one, and the teacache record is adaptive
    assert d["parallel"] == resolve("hunyuan_video", "tp4").parallel_dict()
    assert d["teacache"]["mode"] == "adaptive"
    assert cell_is_blocked("hunyuan_video", "tp4tcad") is True
    # a corrupt file is not "blocked"
    open(p, "w").write("{ not json")
    assert cell_is_blocked("hunyuan_video", "tp4tcad") is False


def test_campaign_summary_renders_a_blocked_cell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_blocked_cell("hunyuan_video", "tp4tcad",
                       reason="HBM exhausted: DiT already fills the core-pair",
                       evidence="logs/... ret=-12",
                       extra={"blocked_class": "hbm_oom"})
    feat = "\n".join(cs.feature_table("tp4tcad", None))
    assert "BLOCKED" in feat and "HBM exhausted" in feat
    assert "hunyuan_video_tp4tcad.md" not in feat  # no link to a report that isn't there
    tea = "\n".join(cs.teacache_table(["tp4tcad"]))
    assert "BLOCKED" in tea and "HunyuanVideo" in tea
