"""V3 summary.json from the per-run res-*.json, compare.json, traj_compare.json and embeds_parity.json."""
import json
import re
from pathlib import Path

W = Path("/home/ubuntu/qwen-v3-work")


def j(p):
    return json.loads(Path(p).read_text())


def timeline(name, log):
    text = (W / log).read_text(errors="replace")
    start = float(re.search(r"start_epoch=([\d.]+)", text).group(1))
    end = float(re.search(r"end_epoch=([\d.]+)", text).group(1))
    dec = json.JSONDecoder()
    ev = {}
    for chunk in text.split("[v3probe] ")[1:]:
        try:
            rec = dec.raw_decode(chunk)[0]
        except ValueError:
            continue
        ev.setdefault(rec["event"], {})[rec["rank"]] = rec
    out = {}
    loads = ev.get("app_load", {})
    if loads:
        out["start_to_ranks_loaded_s"] = round(max(r["t_epoch"] for r in loads.values()) - start, 2)
        out["rank_app_load_s_max"] = max(r["app_load_s"] for r in loads.values())
        out["rank_spawn_imports_prepare_s"] = round(
            out["start_to_ranks_loaded_s"] - out["rank_app_load_s_max"], 2)
    gen = ev.get("generate", {}).get("0")
    if gen:
        out["generate_end_s"] = round(gen["t_epoch"] - start, 2)
    hd = ev.get("host_decode", {}).get("driver")
    if hd and gen:
        out["generate_end_to_decode_start_s"] = round(hd["t_epoch"] - hd["decode_s"] - gen["t_epoch"], 2)
        out["host_decode_s"] = hd["decode_s"]
        out["decode_end_to_exit_s"] = round(end - hd["t_epoch"], 2)
    out["process_wall_s"] = round(end - start, 2)
    return out


runs = ["tp4-compile", "tp4-s50-gen1", "tp4-s50-gen2", "tp4-s50-gen3-noprobe", "tp4-s30",
        "tp2cp2-ulysses-s50", "tp4-eager-s50", "cold-compile", "cold-gen"]
summary = {"runs": {}, "timelines": {}}
for r in runs:
    res = j(W / f"res-{r}.json")
    summary["runs"][r] = res
    if r != "tp4-s50-gen3-noprobe":
        summary["timelines"][r] = timeline(r, f"{r}.log")
summary["compare_images"] = j(W / "compare.json")
summary["teacher_forced_and_free_run"] = j(W / "diag/traj_compare.json")
summary["embeds_parity"] = j(W / "embeds_parity.json")
g1, g2 = summary["runs"]["tp4-s50-gen1"], summary["runs"]["tp4-s50-gen2"]
cc, cg = summary["timelines"]["cold-compile"], summary["runs"]["cold-compile"]
warm_wu = max(v["phase_seconds"]["warmup forward"] for v in summary["runs"]["tp4-compile"]["app_load"].values())
cold_wu = max(v["phase_seconds"]["warmup forward"] for v in cg["app_load"].values())
summary["derived"] = {
    "cli_cold_compile_wall_s": round(cold_wu - warm_wu, 2),
    "cli_cold_compile_rule": "rank-max warmup forward of `difflet compile` on an empty cache root "
                             f"({cold_wu:.2f}) minus the same on the warm main cache ({warm_wu:.2f})",
    "cold_compile_new_neff": cg["new_neff"],
    "png_sha256_identical_tp4_s50": len({summary["runs"][r]["png_sha256"] for r in (
        "tp4-s50-gen1", "tp4-s50-gen2", "tp4-s50-gen3-noprobe", "cold-gen")}) == 1,
    "first_encode_page_cache_note": "gen1 rank-0 encode 98.4 s vs 0.57 s in gen2; inference: the "
        "16 GB bf16 text_encoder shards are mmap'd lazily (text encoder load 0.7 s) and were not "
        "in the page cache, so gen1's encode read them from disk; free -g buff/cache 55 -> 68 GiB",
}
for k in ("tp4-s50-gen1", "tp4-s50-gen2"):
    pass
(W / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary["derived"], indent=1))
print(json.dumps(summary["timelines"], indent=1))
