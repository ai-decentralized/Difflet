"""The gloo-rank worker of the LTX-2 neuron application's TeaCache test (CPU only). It lives in
an importable module because the spawned rank unpickles its target by module name."""

import contextlib
import io

import torch

from tests.unit.backends._neuron_ltx2_toy import (
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    tiny_conditioning,
)


def teacache_worker(rank: int, world_size: int, model_dir: str, kwargs: dict, steps: int) -> list:
    """Two CFG ``generate`` calls (two DiT forwards per full step) of one application with a
    probe-free TeaCache mode: per call the latents, the DiT forwards, the iterations, the
    controller's counters and what the loop printed."""
    from difflet.models.ltx_2.neuron_application import TorchNeuronLtx2Application
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = TorchNeuronLtx2Application(
        model_path=model_dir, parallel=DiffletParallelConfig(), dtype=torch.float32,
        shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN, enable_text_encoder=False,
        exec_mode="eager", device="cpu", **kwargs)
    app.load()
    steps_seen: list = []
    app.step_hook = lambda: steps_seen.append(1)
    calls: list = []
    app.transformer_adapter.forward_hook = calls.append
    results = []
    for _ in range(2):
        steps_seen.clear()
        calls.clear()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            out = app.generate(**tiny_conditioning(2), num_inference_steps=steps,
                               guidance_scale=3.0, seed=0, output_type="latent")
        stats = app.pipeline._teacache_controller.stats()
        results.append({
            "video": out.latents.tolist(), "audio": out.audio_latents.tolist(),
            "steps": len(steps_seen), "dit_calls": len(calls),
            "full": stats["full_steps"], "skipped": stats["skipped_steps"],
            "stats_line": f"[teacache] stats: {stats}" in printed.getvalue(),
        })
    return results
