"""aot15_graph keys the HV1.5 AOT stage cache, so it must import on any host: no Neuron SDK,
no trainium package (whose __init__ pulls in NxD)."""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import sys
import difflet.models.hunyuan_video.aot15_graph
bad = sorted(m for m in sys.modules
             if m.split(".")[0] in {"neuronx_distributed", "torch_neuronx", "torch_xla"}
             or m.startswith("difflet.backends."))
print(bad)
"""


def test_aot15_graph_imports_without_neuron_or_backends():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]"


def test_hv15_num_image_tokens_matches_the_vae_compression():
    from difflet.models.hunyuan_video.aot15_graph import hv15_num_image_tokens

    assert hv15_num_image_tokens(320, 512, 61) == 16 * 20 * 32
    assert hv15_num_image_tokens(480, 848, 121) == 31 * 30 * 53
