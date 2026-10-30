"""The Flux model module is backend-neutral: the AoT application lives in
difflet.backends.trainium.flux.backbone, and layer-boundary markers go through difflet.ops."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
import torch


def test_flux_model_imports_without_the_aot_backend():
    code = (
        "import sys\n"
        "import difflet.models.flux.modeling_flux as mf\n"
        "import difflet.models.flux.t5.modeling_t5 as t5\n"
        "import difflet.models.flux.clip.modeling_clip as clip\n"
        "import difflet.models.flux.vae.modeling_vae as vae\n"
        "assert hasattr(mf, 'NeuronFluxTransformer2DModel') and hasattr(t5, 'NeuronT5EncoderModel')\n"
        "assert hasattr(clip, 'NeuronCLIPTextModel') and hasattr(vae, 'PatchedGroupNorm')\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('difflet.backends.trainium')"
        " or m.startswith('neuronx_distributed') or m.startswith('torch_xla'))\n"
        "print(loaded)\n"
        "assert not loaded, loaded\n"
    )
    env = dict(os.environ, DIFFLET_BACKEND="neuron")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_markers_pass_inputs_through_on_the_neuron_backend():
    from difflet.backends.neuron.ops_impl.markers import (
        ModuleMarkerEndWrapper,
        ModuleMarkerStartWrapper,
    )

    a, b = torch.randn(2, 3), torch.randn(4)
    assert ModuleMarkerStartWrapper()(a) is a
    out = ModuleMarkerEndWrapper()(a, b)
    assert isinstance(out, tuple) and out[0] is a and out[1] is b


def test_unknown_names_still_raise_attribute_error():
    import difflet.models.flux.modeling_flux as mf

    with pytest.raises(AttributeError, match="no attribute 'NotAThing'"):
        mf.NotAThing  # noqa: B018


@pytest.mark.skipif(importlib.util.find_spec("neuronx_distributed") is None,
                    reason="the AoT (Trainium) stack is not installed")
def test_old_import_path_reaches_the_moved_aot_classes():
    import difflet.models.flux.modeling_flux as mf
    from difflet.backends.trainium.flux import backbone

    for name in ("FluxBackboneInferenceConfig", "ModelWrapperFluxBackbone",
                 "NeuronFluxBackboneApplication"):
        assert getattr(mf, name) is getattr(backbone, name)


@pytest.mark.skipif(importlib.util.find_spec("neuronx_distributed") is None,
                    reason="the AoT (Trainium) stack is not installed")
@pytest.mark.parametrize("model_mod, aot_mod, names", [
    ("difflet.models.flux.t5.modeling_t5", "difflet.backends.trainium.flux.t5",
     ("T5InferenceConfig", "ModelWrapperT5", "NeuronT5Application")),
    ("difflet.models.flux.clip.modeling_clip", "difflet.backends.trainium.flux.clip",
     ("CLIPInferenceConfig", "ModelWrapperClip", "CLIPEncoderOutput", "NeuronClipApplication")),
    ("difflet.models.flux.vae.modeling_vae", "difflet.backends.trainium.flux.vae",
     ("VAEDecoderInferenceConfig", "ModelWrapperVAEDecoder", "NeuronVAEDecoderApplication")),
])
def test_old_import_paths_reach_the_moved_encoder_and_vae_classes(model_mod, aot_mod, names):
    import importlib

    model, aot = importlib.import_module(model_mod), importlib.import_module(aot_mod)
    for name in names:
        assert getattr(model, name) is getattr(aot, name)
