"""parse_generate must read both load-line formats difflet has emitted."""
from __future__ import annotations

from benchmark.parse_generate import parse

_OLD = """
Neuron: Done Sharding weights in 13.9
INFO:Neuron:Finished weights loading in 8.27 seconds
[text] encoded
INFO:Neuron:Finished weights loading in 6.52 seconds
[generate] latents
"""

_NEW = """
Neuron: Loading presharded checkpoints for ranks: 0...3
Neuron: Finished traced model weight initialization in 12.38s (device init 12.38s, total load_weights 12.41s)
[text] encoded
Neuron: Loading presharded checkpoints for ranks: 0...3
Neuron: Finished traced model weight initialization in 8.87s (device init 8.87s, total load_weights 9.16s)
[wan] latents saved
Neuron: Loading presharded checkpoints for ranks: 0...0
Neuron: Finished traced model weight initialization in 30.43s (device init 30.43s, total load_weights 30.43s)
[wan] video saved
"""


def test_parse_old_load_line_with_shard():
    eb = parse(_OLD, wall_total_s=60.0)
    assert [s["load_s"] for s in eb["stages"]] == [8.27, 6.52]
    assert eb["stages"][0]["shard_s"] == 13.9
    assert eb["weights_load_total_s"] == 14.79
    assert eb["compute_and_overhead_s"] == 45.21


def test_parse_presharded_load_line_uses_total_load_weights():
    eb = parse(_NEW, wall_total_s=84.6)
    assert [s["load_s"] for s in eb["stages"]] == [12.41, 9.16, 30.43]
    assert eb["weights_load_total_s"] == 52.0
    assert eb["stages"][-1]["stage"] == "vae_decoder"   # trailing stage default
    assert abs(eb["compute_and_overhead_s"] - 32.6) < 1e-9
