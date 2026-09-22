"""Compile and execute a tiny synthetic Wan/LTX-2 probe on Neuron hardware.

Run from the repository root with PYTHONPATH=. and the Neuron Python environment:
    python tests/manual/check_wan_ltx2_teacache_probe.py --model wan --tp-degree 1
    python tests/manual/check_wan_ltx2_teacache_probe.py --model ltx_2 --tp-degree 2

Uses tiny test configurations, not downloaded model weights. Wan exercises two
shape buckets. Checks scalar-only output, persistent state, and CPU signal parity.
"""

import argparse
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from difflet.backends.trainium.core.config import NeuronConfig
from tests.unit.pipeline.test_wan_ltx2_device_probe import _ltx_app, _wan_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("wan", "ltx_2"), required=True)
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    root = args.work_dir or Path(tempfile.mkdtemp(prefix=f"difflet-{args.model}-probe-"))
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    kwargs = {"shapes": [(32, 32, 5), (16, 32, 5)]} if args.model == "wan" else {}
    parent = (_wan_app if args.model == "wan" else _ltx_app)(
        root / "source", teacache_fused=True, **kwargs
    )
    config = parent.teacache_probe.config
    config.neuron_config = NeuronConfig(
        batch_size=1,
        tp_degree=args.tp_degree,
        world_size=args.tp_degree,
        torch_dtype=torch.bfloat16,
    )
    app = type(parent.teacache_probe)(
        model_path=str(root / "source" / "transformer"), config=config
    )
    reference = app._model_cls(config).to(torch.bfloat16).eval()
    state = {
        k.removeprefix("transformer."): v.contiguous()
        for k, v in reference.state_dict().items()
        if k != "prev_mod"
    }
    save_file(state, str(root / "source" / "transformer" / "diffusion_pytorch_model.safetensors"))
    examples = app.model.input_generator()
    print(f"{args.model}: compiling at {root}", flush=True)
    app.compile(str(root / "compiled"))
    app.load(str(root / "compiled"), skip_warmup=True)
    for index, inputs in enumerate(examples):
        first = app.teacache_delta(*inputs)
        assert first.numel() == 1 and torch.isfinite(first).all()
        repeated = app.teacache_delta(*inputs)
        # Compiled bf16 reductions can have a small floor (observed on Wan's
        # padded bucket). This bound is below half one bf16 relative ULP.
        assert repeated.abs().item() < torch.finfo(torch.bfloat16).eps / 2
        changed = (inputs[0] * 1.2, inputs[1] + 100)
        with torch.no_grad():
            old = reference.teacache_mod_input(*inputs).float()
            new = reference.teacache_mod_input(*changed).float()
            expected = (new - old).abs().mean() / old.abs().mean().clamp_min(1e-8)
        actual = app.teacache_delta(*changed).float().reshape(())
        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.002)
        print(
            f"{args.model} tp={args.tp_degree} bucket={index}: "
            f"repeat={repeated.item():.6f}, changed={actual.item():.6f}, "
            f"cpu={expected.item():.6f}: PASS",
            flush=True,
        )


if __name__ == "__main__":
    main()
