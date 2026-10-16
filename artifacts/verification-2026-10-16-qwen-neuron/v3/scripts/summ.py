"""Summarise V3 CLI launch logs (cli.sh + the sitecustomize probe) into one JSON per run."""
import hashlib
import json
import re
import sys
from pathlib import Path

RANK_LINE = re.compile(
    r"\[qwen_image\] neuron rank (\d+)/(\d+): exec_mode=(\w+) compiled_blocks=(\d+) "
    r"unwarmed_shapes=(\[.*?\]) fallbacks: load=(\[.*?\]) generate=(\[.*?\]) "
    r"load\+warmup ([\d.]+)s, stage ([\d.]+)s")
EXIT = re.compile(r"EXIT rc=(\d+) process_wall_s=([\d.]+) start_epoch=([\d.]+) end_epoch=([\d.]+) "
                  r"new_neff=(-?\d+) new_hlo_files=(-?\d+)")


def file_sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def summarise(log, run_dir):
    text = Path(log).read_text(errors="replace")
    dec = json.JSONDecoder()
    probes = []
    for chunk in text.split("[v3probe] ")[1:]:
        try:
            probes.append(dec.raw_decode(chunk)[0])
        except ValueError:
            pass
    ranks = {}
    for m in RANK_LINE.finditer(text):
        ranks[m.group(1)] = {"exec_mode": m.group(3), "compiled_blocks": int(m.group(4)),
                             "unwarmed_shapes": m.group(5), "fallbacks_load": m.group(6),
                             "fallbacks_generate": m.group(7), "load_warmup_s": float(m.group(8)),
                             "stage_s": float(m.group(9))}
    ex = EXIT.search(text)
    out = {"log": str(log), "cmd": next((l[5:] for l in text.splitlines() if l.startswith("cmd: ")), None),
           "launch": next((l for l in text.splitlines() if l.startswith("launch ")), None),
           "rank_lines": ranks}
    if ex:
        out.update(rc=int(ex.group(1)), process_wall_s=float(ex.group(2)), new_neff=int(ex.group(5)),
                   new_hlo_files=int(ex.group(6)))
    by = {}
    for p in probes:
        by.setdefault(p["event"], {})[p["rank"]] = p
    if "app_load" in by:
        out["app_load"] = {r: {k: p.get(k) for k in ("app_load_s", "phase_seconds", "counters",
                                                     "max_memory_allocated_gib", "VmHWM_gib")}
                           for r, p in sorted(by["app_load"].items())}
    if "encode_prompt" in by:
        e0 = by["encode_prompt"].get("0", {})
        out["text_encode_rank0_s"] = e0.get("encode_s")
        out["text_valid_rows"] = e0.get("valid_rows")
        out["embeds_sha256_distinct"] = sorted({p["embeds_sha256"] for p in by["encode_prompt"].values()})
    if "generate" in by:
        g = by["generate"]
        g0 = g.get("0", {})
        out["generate_rank0"] = {k: g0.get(k) for k in ("generate_s", "denoise_s", "pre_denoise_s",
                                                       "steps", "step0_ms", "step_ms", "forward_ms",
                                                       "latents_shape", "latents_dtype",
                                                       "latents_finite", "latents_std")}
        out["rank0_host_VmHWM_gib"] = g0.get("VmHWM_gib")
        out["host_VmHWM_gib_by_rank"] = {r: p.get("VmHWM_gib") for r, p in sorted(g.items())}
        out["max_memory_allocated_gib_by_rank"] = {r: p.get("max_memory_allocated_gib")
                                                   for r, p in sorted(g.items())}
        out["counters_end_by_rank"] = {r: p.get("counters") for r, p in sorted(g.items())}
        out["step_median_ms_by_rank"] = {r: p["step_ms"]["median"] for r, p in sorted(g.items())}
        out["latents_sha256_by_rank"] = {r: p.get("latents_sha256") for r, p in sorted(g.items())}
        out["latents_identical_across_ranks"] = len(set(out["latents_sha256_by_rank"].values())) == 1
    if "host_decode" in by:
        d = by["host_decode"]["driver"]
        out["host_decode_s"] = d.get("decode_s")
        out["host_decode_threads"] = d.get("torch_threads")
        out["driver_VmHWM_gib"] = d.get("VmHWM_gib")
        out["latents_pt_sha256"] = d.get("latents_pt_sha256")
    run = Path(run_dir)
    out["png_sha256"] = file_sha(run / "out.png")
    lat = run / "work" / "latents.pt"
    if lat.is_file():
        import torch

        t = torch.load(lat, map_location="cpu")
        raw = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
        out["latents_pt_tensor_sha256"] = hashlib.sha256(raw.contiguous().numpy().tobytes()).hexdigest()
    return out


if __name__ == "__main__":
    res = summarise(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
