"""CPU probe of the real HunyuanVideo 1.0 DiT (13B, bf16) with the real text conditioning.

1. Per target linear: input absmax over image rows / valid text rows / pad text rows, so we
   can see whether pad rows set the per-tensor dynamic FP8 activation scale.
2. Fake-quant forwards (same math as the device path, difflet.quant.fp8): bf16, weight-only,
   dynamic with raw pad rows, dynamic with zeroed pad rows, bf16 with zeroed pad rows;
   image-token output compared against bf16.
"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("DIFFLET_BACKEND", "cpu")
sys.path.insert(0, "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from difflet.quant.fp8 import dequantize, fake_quant_activation, quantize_weight  # noqa: E402
from difflet.quant.metrics import tensor_error_metrics  # noqa: E402
from difflet.quant.spec import QuantSpec  # noqa: E402
from difflet.quant.targets import HUNYUAN_VIDEO_TARGETS, device_targets  # noqa: E402

TEXT = "/home/ubuntu/.claude/jobs/b5f130d0/tmp/hv_text.pt"
OUT = "/home/ubuntu/.claude/jobs/b5f130d0/tmp/hv_dit_cpu_probe.json"
torch.manual_seed(0)
torch.set_num_threads(12)

snap = glob.glob("/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/*")[0]
tdir = f"{snap}/transformer"
import difflet.models.hunyuan_video.modeling_hunyuan_video as m  # noqa: E402

cfg = m.HunyuanVideoTransformerConfig.from_diffusers_dict(json.loads(open(f"{tdir}/config.json").read()))
t0 = time.time()
model = m.HunyuanVideoTransformer3DModel(cfg).to(torch.bfloat16).eval()
print("model built", round(time.time() - t0, 1), "s", flush=True)

from difflet.backends.trainium.core.modules.checkpoint import load_state_dict  # noqa: E402
from difflet.backends.trainium.hunyuan_video.backbone import NeuronHunyuanVideoBackboneApplication  # noqa: E402
from types import SimpleNamespace  # noqa: E402

t0 = time.time()
sd = load_state_dict(tdir)
sd = NeuronHunyuanVideoBackboneApplication.convert_hf_to_neuron_state_dict(
    sd, SimpleNamespace(num_attention_heads=cfg.num_attention_heads, attention_head_dim=cfg.attention_head_dim,
                        num_single_layers=cfg.num_single_layers, neuron_config=SimpleNamespace(world_size=1)))
missing, unexpected = model.load_state_dict(sd, strict=False)
print("weights loaded", round(time.time() - t0, 1), "s; missing", len(missing), missing[:5], "unexpected", len(unexpected), unexpected[:5], flush=True)
del sd

text = torch.load(TEXT)
ehs = text["encoder_hidden_states"].to(torch.bfloat16)
mask = text["encoder_attention_mask"]
pooled = text["pooled_projections"].to(torch.bfloat16)
n_valid = int(mask.sum())
# small latent: 2 latent frames x 12x12 patches = 288 image tokens
latents = torch.randn(1, 16, 2, 24, 24).to(torch.bfloat16)
n_img = 2 * 12 * 12
timestep = torch.tensor([900.0], dtype=torch.bfloat16)
guidance = torch.tensor([6000.0], dtype=torch.bfloat16)
print("inputs: image tokens", n_img, "text rows", ehs.shape[1], "valid", n_valid, flush=True)

spec = QuantSpec.for_model("hunyuan_video")
targets = device_targets(HUNYUAN_VIDEO_TARGETS)
spec_dev = QuantSpec(targets=tuple(targets))


class Switchable(nn.Module):
    """Target linear with a global mode: bf16 / wo (fp8 weight) / dyn (fp8 weight + per-tensor dynamic fp8 input)."""

    mode = "bf16"
    stats: dict | None = None
    name = ""

    def __init__(self, src: nn.Linear, name: str) -> None:
        super().__init__()
        self.name = name
        self.weight = src.weight
        self.bias = src.bias
        w8, s = quantize_weight(src.weight.detach(), "tensor")
        self.register_buffer("w8", w8)
        self.register_buffer("ws", s)

    def forward(self, x):
        if Switchable.stats is not None:
            rows = x.shape[1]
            a = x.detach().float().abs().amax(dim=-1)[0]  # per row
            if rows == ehs.shape[1]:
                img, valid, pad = None, a[:n_valid].max(), a[n_valid:].max()
            elif rows == n_img:
                img, valid, pad = a.max(), None, None
            elif rows == n_img + ehs.shape[1]:
                img, valid, pad = a[:n_img].max(), a[n_img:n_img + n_valid].max(), a[n_img + n_valid:].max()
            else:
                img, valid, pad = a.max(), None, None
            Switchable.stats[self.name] = {k: (None if v is None else float(v)) for k, v in
                                           (("img", img), ("valid", valid), ("pad", pad))}
        if Switchable.mode == "bf16":
            return nn.functional.linear(x, self.weight, self.bias)
        x32 = x.float()
        if Switchable.mode == "dyn":
            x32 = fake_quant_activation(x32)
        out = x32 @ dequantize(self.w8, self.ws).t()
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)


swapped = 0
for parent_name, parent in list(model.named_modules()):
    for child_name, child in list(parent.named_children()):
        q = f"{parent_name}.{child_name}" if parent_name else child_name
        if isinstance(child, nn.Linear) and spec_dev.matches(q):
            setattr(parent, child_name, Switchable(child, q))
            swapped += 1
print("target linears wrapped", swapped, flush=True)


def run(mode, zero_pads):
    Switchable.mode = mode
    e = ehs * mask.bool().unsqueeze(-1) if zero_pads else ehs
    t0 = time.time()
    with torch.no_grad():
        out = model(hidden_states=latents, timestep=timestep, encoder_hidden_states=e, encoder_attention_mask=mask,
                    pooled_projections=pooled, guidance=guidance, return_dict=False)[0]
    print(f"forward {mode} zero_pads={zero_pads}: {time.time() - t0:.1f}s finite={bool(torch.isfinite(out).all())}", flush=True)
    return out.float()


results = {}
Switchable.stats = {}
ref = run("bf16", False)
stats_raw = Switchable.stats
Switchable.stats = {}
_ = run("bf16", True)
stats_zero = Switchable.stats
Switchable.stats = None


def summarize(stats, label):
    worst = []
    for name, s in stats.items():
        if s["pad"] is None:
            continue
        base = max(v for v in (s["img"], s["valid"]) if v is not None)
        worst.append((s["pad"] / max(base, 1e-6), name, s))
    worst.sort(reverse=True)
    n_pad_dominated = sum(1 for r, _, _ in worst if r > 1.0)
    print(f"== {label}: {len(worst)} target linears see text rows; pad absmax > non-pad absmax in {n_pad_dominated}", flush=True)
    for r, name, s in worst[:12]:
        print(f"   x{r:8.1f}  {name}  img={s['img']} valid={s['valid']} pad={s['pad']}", flush=True)
    return {"n_text_linears": len(worst), "n_pad_dominated": n_pad_dominated,
            "worst": [(r, n, s) for r, n, s in worst[:40]]}


results["stats_raw_pads"] = summarize(stats_raw, "raw pad rows (as the CLI feeds them)")
results["stats_zeroed_pads"] = summarize(stats_zero, "zeroed pad rows (the fix)")

for label, mode, zero in (("bf16_zeroed_pads", "bf16", True), ("wo_raw", "wo", False), ("dyn_raw", "dyn", False),
                          ("dyn_zeroed", "dyn", True), ("wo_zeroed", "wo", True)):
    out = run(mode, zero)
    met = tensor_error_metrics(ref, out)
    results[label] = met
    print(f"   {label}: cosine={met['cosine']:.6f} rel_l2={met['rel_l2']:.4f} snr_db={met['snr_db']:.2f} max_abs={met['max_abs']:.3f}", flush=True)

json.dump(results, open(OUT, "w"), indent=1, default=str)
print("wrote", OUT, flush=True)
