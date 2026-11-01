"""FP8 post-training quantization of DiT linears (W8A8, static per-tensor scales).

Backend-neutral: the quantization law (``fp8``), the spec that names what is quantized and how
(``spec``), the per-model target layers (``targets``), the activation calibration (``calibration``),
the CPU reference layer every device result is checked against (``fake_linear``) and the offline
quantized checkpoint (``checkpoint``). The device side lives with each backend
(``difflet/backends/neuron/ops_impl/fp8_linear.py``).

``spec`` is torch-free so the CLI driver can parse flags and build cache keys before torchrun
starts a rank; the other modules import torch.
"""
