"""Back-fill detailed compile + e2e breakdowns into the result JSONs from logs.

Re-parses each model's saved compile/generate log (benchmark/<device>/logs/) with
parse_compile / parse_generate and writes the detailed, *named* breakdowns back
into benchmark/<device>/<slug>.json, then re-renders the md. Idempotent.

    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python -m benchmark.backfill_breakdowns
"""
from __future__ import annotations

import json
import os

from benchmark import parse_compile, parse_generate, report

# slug -> {compile log stem, generate log stem, stage-name template, host note}
# stage_names overrides parser labels when the log has no [role] markers
# (Wan, LTX-2). host_note records stages that run on the host CPU (not a Neuron
# load line), so the compute residual is explained.
SPEC = {
    "ltx_2": dict(
        clog="ltx_2", glog="ltx_2",
        stage_names=["transformer (denoise loop) [Neuron]"],
        host_note="text-encoder and VAE decode run on the host (enable_host_pipeline"
                  "/enable_decode_components), so only the transformer is a Neuron "
                  "load; the residual is host text-encode + denoise + host VAE decode."),
    "qwen_image": dict(clog="qwen_image", glog="qwen_image"),
    "hunyuan_video": dict(
        clog="hunyuanvideo", glog="hunyuanvideo",
        host_note="VAE decode runs on the host (no Neuron load line); the residual "
                  "is CLIP+Llama encode + denoise loop + host VAE decode."),
    "wan_2_1": dict(
        clog=None, glog="wan2_1_t2v_14b_diffusers",
        stage_names=["text_encoder (UMT5)", "transformer (denoise loop)", "vae_decoder"]),
    "wan_2_2": dict(
        clog="wan2_2_t2v_a14b_diffusers", glog="wan2_2_t2v_a14b_diffusers",
        stage_names=["text_encoder (UMT5)", "transformer (denoise loop)", "vae_decoder"]),
}
from benchmark.models import json_path, report_path, logs_dir
LD = logs_dir()


def main() -> int:
    for slug, s in SPEC.items():
        jp = json_path(slug)
        if not os.path.exists(jp):
            continue
        d = json.load(open(jp))

        clog = s.get("clog")
        if clog and os.path.exists(f"{LD}/{clog}_compile.log"):
            cb = parse_compile.parse_file(f"{LD}/{clog}_compile.log")
            if cb:  # skip cache-hit logs (no builds)
                d["compile_breakdown"] = cb
                if d.get("compile_seconds") in (None, 0) and "wall_total_s" in cb:
                    d["compile_seconds"] = cb["wall_total_s"]

        glog = s.get("glog")
        if glog and os.path.exists(f"{LD}/{glog}_generate.log"):
            eb = parse_generate.parse_file(f"{LD}/{glog}_generate.log",
                                           d.get("e2e_cold_seconds"))
            names = s.get("stage_names")
            if names and len(names) == len(eb.get("stages", [])):
                for st, nm in zip(eb["stages"], names):
                    st["stage"] = nm
            if s.get("host_note"):
                eb["note"] = s["host_note"]
            if eb.get("stages"):
                d["e2e_breakdown"] = eb
                d["load_seconds"] = eb["weights_load_total_s"]

        json.dump(d, open(jp, "w"), indent=2)
        open(report_path(slug), "w").write(report.render(d))
        eb = d.get("e2e_breakdown", {})
        print(f"{slug:14} compile={d.get('compile_seconds')}  "
              f"load_total={d.get('load_seconds')}  "
              f"compute_resid={eb.get('compute_and_overhead_s')}  "
              f"stages={[x['stage'] for x in eb.get('stages', [])]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
