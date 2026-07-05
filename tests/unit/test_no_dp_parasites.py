"""No CFG/CP/TP collective may ride the dp axis.

Static audit: outside the NxDI verbatim-fork utils (which merely *define*
the legacy helper) nothing may reference NxD's ``get_data_parallel_group``
or the merged-axis ``get_dp_rank_spmd``. Together with the manager's rules —
a dp group is only built when dp > 1, and nothing outside the manager calls
``get_dp_group`` — this enforces that the dp axis carries no per-layer /
per-step collective during the denoise loop.
"""

from pathlib import Path

import difflet
import difflet.ops as ops

REPO = Path(difflet.__file__).resolve().parent

FORBIDDEN = ("get_data_parallel_group", "get_dp_rank_spmd")

# NxDI verbatim-fork file that defines (but must not spread) the legacy helper.
ALLOWED = {
    REPO / "backends/trainium/utils/distributed.py",
}


def test_ops_surface_has_no_dp_exports():
    assert "get_data_parallel_group" not in ops.__all__
    assert "get_dp_rank_spmd" not in ops.__all__
    for name in (
        "init_parallel_mesh",
        "get_cfg_group",
        "get_cp_group",
        "get_cfg_rank_spmd",
        "get_cp_rank_spmd",
    ):
        assert name in ops.__all__


def test_no_source_file_references_dp_group():
    offenders = []
    for path in sorted(REPO.rglob("*.py")):
        if path in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{path.relative_to(REPO)}: {token}")
    assert offenders == [], "\n".join(offenders)


def test_dp_axis_group_has_no_forward_consumer():
    # get_dp_group exists on the manager (reserved for the DP feature) but no
    # model / pipeline / ops code may call it in this increment.
    manager = REPO / "backends/trainium/core/parallel_mesh.py"
    offenders = []
    for path in sorted(REPO.rglob("*.py")):
        if path == manager:
            continue
        if "get_dp_group" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], "\n".join(offenders)
