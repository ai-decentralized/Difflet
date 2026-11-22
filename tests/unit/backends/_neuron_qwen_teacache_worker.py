"""The gloo rank worker of the Q2 probe-free TeaCache test (not collected by pytest; it lives
in an importable module for ``_neuron_gloo.run_ranks``)."""

from tests.unit.backends._neuron_qwen_app_toy import TINY_Z_DIM, tiny_text_inputs
from tests.unit.backends._neuron_qwen_toy import TINY_SHAPE, TINY_TEXT_SEQ_LEN


def q2_teacache_worker(rank: int, world_size: int, model_dir: str, teacache: dict) -> dict:
    """The composed application at TP=world on CPU fp32 with probe-free TeaCache: two 12-step
    requests on the fixed embeddings from seed 7; this rank's stdout, steps, forwards and
    stats."""
    import contextlib
    import hashlib
    import io

    import torch

    from difflet.models.qwen_image.neuron_application import TorchNeuronQwenImageApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = TorchNeuronQwenImageApplication(
        model_path=model_dir, parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32, shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager", device="cpu", enable_text_encoder=False, **teacache,
    )
    app.load()
    steps: list[None] = []
    forwards: list[float] = []
    app.step_hook = lambda: steps.append(None)
    app.transformer_adapter.forward_hook = forwards.append
    embeds, mask = tiny_text_inputs()
    shas = []
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for _ in range(2):
            latents = app(encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
                          num_inference_steps=12, guidance_scale=4.0, seed=7,
                          channels=TINY_Z_DIM).latents
            shas.append(hashlib.sha256(latents.contiguous().numpy().tobytes()).hexdigest())
    return {"stats": app.pipeline.teacache_controller.stats(), "out": out.getvalue(),
            "steps": len(steps), "forwards": len(forwards), "sha256": shas}
