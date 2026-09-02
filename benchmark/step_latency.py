"""In-process warm DiT-forward timer (the stable per-step metric).

Why this and not the marginal-from-generate method: each `difflet generate` is a
fresh process whose text-encoder load (5-11 GB) swings with OS page-cache warmth,
so e2e timings are load-dominated and noisy (LTX-2 @20 steps measured 293 s and
628 s on different runs) — subtracting two such runs gives nonsense. Loading the
compiled transformer ONCE and timing N warm forwards isolates the pure Neuron
DiT-step latency, which is what "optimal performance" should report.

    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python -m benchmark.step_latency --model wan_2_1 [--iters 20]

Patches benchmark/<device>/<slug>.json (step_latency, throughput) and re-renders
benchmark/<slug>.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmark import report
from benchmark.harness import Stats
from benchmark.models import MATRIX


def _load_transformer(slug, cfg):
    """Return (callable_forward, input_tuple) for the compiled DiT, loaded on device."""
    import torch
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from difflet.pipeline.path_resolver import resolve_model_path

    cache = Path("~/.cache/difflet").expanduser()
    model_dir = resolve_model_path(cfg.model_id, local_files_only=True)
    parallel = DiffletParallelConfig(tp_degree=cfg.tp, cp_degree=cfg.cp, cp_mode=cfg.cp_mode, cfg_parallel_enabled=cfg.cfg_parallel, sp_enabled=cfg.sp)
    shape = {"height": cfg.height, "width": cfg.width, "num_frames": cfg.num_frames}

    if cfg.model_type == "ltx_2":
        from difflet.pipeline.difflet_pipeline import DiffletPipeline
        pipe = DiffletPipeline.from_pretrained(
            cfg.model_id, model_type="ltx_2", parallel=parallel, dtype=torch.bfloat16,
            height=cfg.height, width=cfg.width, num_frames=cfg.num_frames,
            compile_cache_dir=str(cache), skip_compile=True, skip_warmup=True)
        sub = pipe.app.transformer
        return sub, sub.models[0].input_generator()[0]

    # NOTE: flux is intentionally NOT handled here. Its compiled-graph
    # input_generator() order differs from the wrapper forward() signature
    # (which derives image_rotary_emb from img_ids/txt_ids), so positional
    # calling mismaps the args. flux's per-step is taken from the warm
    # denoise-loop rate in its generate log instead (see flux_1_dev.json note).

    if cfg.model_type == "wan":
        from difflet.models.wan.application import NeuronWanApplication
        h, w, f = cfg.height, cfg.width, cfg.num_frames
        app = NeuronWanApplication(
            model_path=model_dir, parallel=parallel, dtype=torch.bfloat16,
            shape=shape, text_seq_len=512, batch_size=1,
            enable_text_encoder=False, enable_transformer=True,
            enable_transformer_2=False, enable_vae_decoder=False,
        )
        app.load(str(cache / f"wan_transformer_tp{cfg.tp}cp{cfg.cp}_h{h}w{w}f{f}"),
                 start_rank_id=0, local_ranks_size=parallel.world_size, skip_warmup=True)
        sub = app.transformer
        return sub, sub.models[0].input_generator()[0]

    if cfg.model_type == "qwen_image":
        from difflet.models.qwen_image.application import NeuronQwenImageApplication
        h, w = cfg.height, cfg.width
        app = NeuronQwenImageApplication(
            model_path=model_dir, parallel=parallel, dtype=torch.bfloat16, shape=shape)
        app.load(str(cache / f"qwen_image_dit_tp{cfg.tp}cp{cfg.cp}_h{h}w{w}"), skip_warmup=True)
        sub = app.transformer
        return sub, sub.models[0].input_generator()[0]

    if cfg.model_type == "hunyuan_video":
        from difflet.models.hunyuan_video.application import NeuronHunyuanVideoApplication
        h, w, f = cfg.height, cfg.width, cfg.num_frames
        app = NeuronHunyuanVideoApplication(
            model_path=model_dir, parallel=parallel, dtype=torch.bfloat16, shape=shape,
            text_seq_len=256, enable_vae_decoder=True)
        app.teacache_probe = None  # not compiled into the generate dir (orchestrator parity)
        app.load(str(cache / f"hunyuan_video_dit_tp{cfg.tp}cp{cfg.cp}_h{h}w{w}f{f}"),
                 skip_warmup=True)
        sub = app.transformer
        return sub, sub.models[0].input_generator()[0]

    raise SystemExit(f"no step-latency loader for model_type {cfg.model_type}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()
    cfg = MATRIX[args.model]

    call, inputs = _load_transformer(args.model, cfg)
    for _ in range(args.warmup):
        call(*inputs)
    samples = []
    for _ in range(args.iters):
        t = time.perf_counter()
        call(*inputs)
        samples.append(time.perf_counter() - t)
    st = Stats.from_samples(samples)
    print(f"[step] {args.model}: {st.mean*1000:.1f} ms/forward "
          f"(median {st.median*1000:.1f}, p90 {st.p90*1000:.1f}, n={st.n})", flush=True)

    from benchmark.models import json_path, report_path
    jp = Path(json_path(args.model))
    if jp.exists():
        d = json.loads(jp.read_text())
        d["step_latency"] = st.__dict__
        if st.mean > 0:
            d["throughput"] = {"DiT steps/s": round(1.0 / st.mean, 3)}
        d["config_slug"] = args.model
        d.setdefault("notes", []).append(
            f"per-step = {st.mean*1000:.1f} ms/DiT-forward (warm, in-process, "
            f"n={st.n}) via benchmark.step_latency — the stable Neuron-compute metric "
            f"(e2e generate is load-dominated/noisy across processes).")
        jp.write_text(json.dumps(d, indent=2))
        Path(report_path(args.model)).write_text(report.render(d))
        print(f"[step] patched {report_path(args.model)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
