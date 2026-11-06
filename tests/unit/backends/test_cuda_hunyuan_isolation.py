"""CUDA overrides stay local to the loaded Hunyuan model instance."""

from dataclasses import fields

import torch


def test_cuda_mask_does_not_change_shared_model_classes(tmp_path):
    from difflet.backends.cuda.hunyuan_video.modeling import CudaHunyuanVideoTransformer3DModel
    from difflet.models.hunyuan_video import modeling_hunyuan_video as shared
    from tests.unit.backends.test_cuda_hunyuan_transformer import _component, _inputs

    original_forward = shared.HunyuanVideoAttention.forward
    assert "joint_attention_mask_mode" not in {
        field.name for field in fields(shared.HunyuanVideoTransformerConfig)
    }
    reference, component = _component(tmp_path, mask_mode="compact")
    assert isinstance(component.module, CudaHunyuanVideoTransformer3DModel)
    assert shared.HunyuanVideoAttention.forward is original_forward
    with torch.no_grad():
        actual = component(*_inputs(), return_dict=False)[0]
        expected = reference(*_inputs(), return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
