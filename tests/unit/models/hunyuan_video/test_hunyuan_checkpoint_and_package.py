"""Coverage for HunyuanVideo checkpoint helpers and package ``__getattr__``."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DIFFLET_BACKEND", "cpu")


def test_convert_vae_decoder_state_dict_filters_prefixes():
    from difflet.models.hunyuan_video.checkpoint import convert_vae_decoder_state_dict

    state = {
        "post_quant_conv.weight": 1,
        "decoder.block.0.weight": 2,
        "encoder.block.0.weight": 3,
        "quant_conv.weight": 4,
    }
    out = convert_vae_decoder_state_dict(state)
    assert set(out) == {"post_quant_conv.weight", "decoder.block.0.weight"}


def test_convert_vae_decoder_state_dict_ignores_config_arg():
    from difflet.models.hunyuan_video.checkpoint.vae import convert_vae_decoder_state_dict

    out = convert_vae_decoder_state_dict({"decoder.x": 1}, config={"anything": True})
    assert out == {"decoder.x": 1}


def test_checkpoint_package_exports():
    import difflet.models.hunyuan_video.checkpoint as ckpt

    assert "convert_vae_decoder_state_dict" in ckpt.__all__


def test_package_getattr_lazy_exports():
    import difflet.models.hunyuan_video as pkg

    for name in pkg.__all__:
        assert getattr(pkg, name) is not None


def test_package_getattr_unknown_raises():
    import difflet.models.hunyuan_video as pkg

    with pytest.raises(AttributeError, match="no attribute"):
        pkg.DefinitelyNotAReal_Symbol
