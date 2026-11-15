"""Task 6: Plan A's bf16 block captures (/mnt/nvme/difflet-prof/run/<key>/bf16/blk<i>) analysed
again with this checkout's analyze.py, only for the attention phases (exp now split out of
mask_softmax); written under task6/planA_bf16/<key>/blk<i>.json, never into Plan A's run. The
check: engines and the attention component equal Plan A's committed breakdown, and the new
mask_softmax + exp equal its mask_softmax.
usage: python reanalyze_bf16.py <worktree> <key>..."""
import json
import sys
from pathlib import Path

wt = Path(sys.argv[1])
sys.path.insert(0, str(wt / "scripts/profile_dit"))
import analyze  # noqa: E402

for key in sys.argv[2:]:
    for blk in sorted(Path(f"/mnt/nvme/difflet-prof/run/{key}/bf16").glob("blk*/")):
        out = analyze.analyze(blk, key)
        out.pop("timeline")
        committed = json.loads((wt / f"docs/verification/assets/2026-11-13-dit-block-profile/{key}/bf16/{blk.name}.json").read_text())
        check = {"engines_equal": all(abs(out["engines_ms"][k] - v) < 1e-6 for k, v in committed["engines_ms"].items()),
                 "attention_equal": abs(sum(out["components"]["attention"].values())
                                        - sum(committed["components"]["attention"].values())) < 1e-6}
        for mod, phases in committed["attention_detail"].items():
            old = sum(phases.get("mask_softmax", {}).values())
            new = sum(out["attention_detail"].get(mod, {}).get("mask_softmax", {}).values()) + sum(
                out["attention_detail"].get(mod, {}).get("exp", {}).values())
            check[f"{mod}_mask_softmax_plus_exp_minus_old_ms"] = new - old
        out["check_against_committed"] = check
        dst = Path(f"/mnt/nvme/difflet-mxattn/task6/planA_bf16/{key}/{blk.name}.json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1) + "\n")
        print(key, blk.name, json.dumps(check), flush=True)
