from difflet.registry import resolve_model


def test_registry_lists_cuda_for_wan():
    for model_id in ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "Wan-AI/Wan2.1-T2V-14B-Diffusers"):
        entry = resolve_model(model_id)
        assert entry.name == "wan" and entry.backends == ("trainium", "tpu", "cuda")
        entry.require_backend("cuda")  # must not raise
