"""The structural checks the neuron DiT configs and applications share (CPU only, torch-free
except the dtype check): ``difflet/backends/neuron/core/config_checks.py``. Each model's own
tests pin its messages end to end; these pin the rules and the message templates once."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from difflet.backends.neuron.core import config_checks as cc


def test_layer_count():
    cc.check_layer_count("Wan", 2, 2)
    cc.check_layer_count("Wan", 1, None)
    with pytest.raises(ValueError, match=r"^Wan num_layers=0 must be >= 1$"):
        cc.check_layer_count("Wan", 0, None)
    with pytest.raises(ValueError, match=r"^LTX-2 num_layers=3 exceeds the checkpoint's 2 blocks$"):
        cc.check_layer_count("LTX-2", 3, 2)


def test_heads_divide_tp():
    cc.check_heads_divide_tp("Wan", 40, 4)
    with pytest.raises(ValueError, match=r"^Wan has 40 attention heads, which does not divide "
                                         r"tp=3$"):
        cc.check_heads_divide_tp("Wan", 40, 3)
    with pytest.raises(ValueError, match=r"^LTX-2 has 32 audio attention heads, which does not "
                                         r"divide tp=3$"):
        cc.check_heads_divide_tp("LTX-2", 32, 3, what="audio")


def test_cp_rules():
    cc.check_cp_mode("ring")
    with pytest.raises(ValueError, match="cp_mode must be one of"):
        cc.check_cp_mode("bogus")
    cc.check_cp_enabled(None, False)
    cc.check_cp_enabled(1, False)
    cc.check_cp_enabled(2, True)
    with pytest.raises(ValueError, match=r"cp_degree=2 needs context_parallel_enabled=True \("):
        cc.check_cp_enabled(2, False)
    cc.check_cp_split("Wan", "image_seq_len", 8, "2x2x2 latent", 2)
    with pytest.raises(ValueError, match=r"^Wan image_seq_len=9 \(3x3 packed\) is not divisible "
                                         r"by cp_degree=2$"):
        cc.check_cp_split("Wan", "image_seq_len", 9, "3x3 packed", 2)
    cc.check_even_split("Wan sequence parallelism splits image_seq_len=8", 8, "tp", 4)
    with pytest.raises(ValueError, match=r"^X splits n=6 over tp=4, which does not divide it$"):
        cc.check_even_split("X splits n=6", 6, "tp", 4)
    cc.check_ulysses_heads(heads=8, tp=2, cp=2)
    with pytest.raises(ValueError, match=r"this rank's 3 heads \(heads=6 / tp=2\) over cp_degree=2"):
        cc.check_ulysses_heads(heads=6, tp=2, cp=2)
    error = cc.exclusive_modes("Wan", "sp_enabled", "context_parallel_enabled", "why")
    assert str(error) == "Wan sp_enabled and context_parallel_enabled are mutually exclusive (why)"


@dataclass
class _Config:
    tp_degree: int = 2
    cp_degree: int | None = None
    context_parallel_enabled: bool = False
    cp_mode: str = "gather_kv"
    cfg_parallel_enabled: bool = False
    sp_enabled: bool = False
    num_layers: int = 2
    checkpoint_num_layers: int | None = None


def _parallel(**kw):
    return SimpleNamespace(**{"tp_degree": 2, "cp_degree": 1, "cp_mode": "gather_kv",
                              "cfg_parallel_enabled": False, **kw})


def test_config_against_parallel_config():
    cc.check_tp_matches("C", _Config(), _parallel())
    with pytest.raises(ValueError, match=r"^C has tp_degree=2 but the parallel config has "
                                         r"tp_degree=4; the module would be sharded"):
        cc.check_tp_matches("C", _Config(), _parallel(tp_degree=4))
    cc.check_flag_matches("C", _Config(), _parallel(), "sp_enabled")  # absent: off
    with pytest.raises(ValueError, match=r"^C has cfg_parallel_enabled=False but the parallel "
                                         r"config has cfg_parallel_enabled=True$"):
        cc.check_flag_matches("C", _Config(), _parallel(cfg_parallel_enabled=True),
                              "cfg_parallel_enabled")
    config = _Config(context_parallel_enabled=True)
    filled = cc.resolve_cp_degree("C", config, _parallel(cp_degree=2))
    assert filled.cp_degree == 2 and config.cp_degree is None  # a copy
    assert cc.resolve_cp_degree("C", _Config(cp_degree=1), _parallel()).cp_degree == 1
    with pytest.raises(ValueError, match="context_parallel_enabled=False but the parallel config "
                                         "has cp_degree=2"):
        cc.resolve_cp_degree("C", _Config(), _parallel(cp_degree=2))
    with pytest.raises(ValueError, match="cp_mode='gather_kv' but the parallel config has "
                                         "cp_mode='ring'"):
        cc.resolve_cp_degree("C", config, _parallel(cp_degree=2, cp_mode="ring"))
    with pytest.raises(ValueError, match="cp_degree=4 but the parallel config has cp_degree=2"):
        cc.resolve_cp_degree("C", _Config(context_parallel_enabled=True, cp_degree=4),
                             _parallel(cp_degree=2))


def test_device_dtype_and_truncated_blocks():
    torch = pytest.importorskip("torch")
    cc.require_bf16_on_device("Wan", "neuron", torch.bfloat16)
    cc.require_bf16_on_device("Wan", "cpu", torch.float32)
    with pytest.raises(ValueError, match="the Wan DiT runs in torch.bfloat16 on the neuron "
                                         "device, got torch.float32"):
        cc.require_bf16_on_device("Wan", "neuron", torch.float32)
    assert cc.truncated_block_prefixes(_Config(), "blocks.") == ()
    assert cc.truncated_block_prefixes(_Config(num_layers=1, checkpoint_num_layers=3),
                                       "blocks.") == ("blocks.1.", "blocks.2.")
