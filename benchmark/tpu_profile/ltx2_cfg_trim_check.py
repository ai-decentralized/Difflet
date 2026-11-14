"""LTX-2 with CFG (guidance > 1): text trim on vs off, one process per chip.

    python -m benchmark.tpu_profile.ltx2_cfg_trim_check [--steps 4 --guidance 4.0]

main's bench runs LTX-2 at guidance 1.0, so the batch-2 conditioning path
(negative + positive prompt through Gemma and the connectors, then a CFG
denoise) is not covered there. Each rank loads the application once with
``DIFFLET_TPU_LTX2_TRIM_TEXT=1``, runs a short CFG request to latents, then
reinstalls the padded path on the same host pipeline and repeats with the same
seed. Rank 0 prints the conditioning time of each and the relative L2 / cosine
of the video and audio latents (trim vs padded).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

MODEL = os.environ.get("DIFFLET_LTX2_PATH", "/mnt/models/hf/hub/models--Lightricks--LTX-2/snapshots")
PROMPT = "a cinematic shot of a red fox running through a snowy forest"
NEGATIVE = "blurry, low quality, distorted"


def _model_path() -> str:
    path = Path(MODEL)
    if (path / "transformer").is_dir():
        return str(path)
    return str(next(p for p in sorted(path.iterdir()) if (p / "transformer").is_dir()))


def _worker(rank, world, args, reply_q):
    try:
        os.environ["DIFFLET_TPU_LTX2_TRIM_TEXT"] = "1"
        from difflet.backends.tpu.ops_impl import device_runtime

        device_runtime.initialize_replica(rank, world)
        import torch

        # As main's bench worker does (benchmark/adapters/tpu.py): share the host.
        torch.set_num_threads(max(1, (os.cpu_count() or world) // world))

        from difflet.models.ltx_2 import tpu_application as mod
        from difflet.models.ltx_2.entry import create_ltx_2_application
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        app = create_ltx_2_application(
            model_path=_model_path(), parallel=DiffletParallelConfig(tp_degree=world),
            dtype=torch.bfloat16, shape={"height": 480, "width": 704, "num_frames": 49},
            backend="tpu", enable_host_pipeline=True,
        )
        app.load_eager()
        orch, pipe = app.pipeline, app.host_pipeline

        def request():
            mark = time.monotonic()
            cond = orch.prepare_conditioning(prompt=PROMPT, negative_prompt=NEGATIVE,
                                             guidance_scale=args.guidance,
                                             max_sequence_length=orch.text_seq_len)
            encode = time.monotonic() - mark
            states, audio_states, mask, audio_mask = cond
            out = orch(encoder_hidden_states=states, audio_encoder_hidden_states=audio_states,
                       encoder_attention_mask=mask, audio_encoder_attention_mask=audio_mask,
                       num_inference_steps=args.steps, guidance_scale=args.guidance,
                       generator=torch.Generator().manual_seed(42), output_type="latent")
            return encode, out.latents.float(), out.audio_latents.float(), tuple(states.shape)

        trim_first = request()
        trim = request()
        # Back to the padded path on the same pipeline.
        pipe._get_gemma_prompt_embeds = type(pipe)._get_gemma_prompt_embeds.__get__(pipe)
        del pipe.connectors.forward
        mod._install_broadcast_prompt_encoder(
            pipe, is_encoder=rank == 0, dtype=torch.float32,
            packed_width=mod.packed_text_dim(_model_path()))
        padded_first = request()
        padded = request()

        def err(a, b):
            return {"rel_l2": float((a - b).norm() / b.norm()),
                    "cos": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))}

        reply_q.put({"rank": rank, "cond_shape": trim[3], "threads": torch.get_num_threads(),
                     "encode_s_trim": [round(trim_first[0], 2), round(trim[0], 2)],
                     "encode_s_padded": [round(padded_first[0], 2), round(padded[0], 2)], "video": err(trim[1], padded[1]),
                     "audio": err(trim[2], padded[2])})
    except BaseException as exc:  # noqa: BLE001
        import traceback

        reply_q.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}",
                     "tb": traceback.format_exc()[-2500:]})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--world", type=int, default=4)
    args = parser.parse_args()
    from difflet.backends.tpu.ops_impl import device_runtime

    device_runtime.prepare_launch(args.world)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, args.world, args, reply_q)) for r in range(args.world)]
    for p in procs:
        p.start()
    results = sorted((reply_q.get(timeout=1800) for _ in procs), key=lambda r: r["rank"])
    for p in procs:
        p.join(timeout=60)
    ok = all("error" not in r for r in results)
    for r in results:
        if "error" in r:
            print(f"rank {r['rank']} FAILED: {r['error']}\n{r['tb']}", flush=True)
    if ok:
        print(results[0], flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
