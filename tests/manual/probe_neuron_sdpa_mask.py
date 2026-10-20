"""Does F.scaled_dot_product_attention honour attn_mask on the neuron backend?

    python tests/manual/probe_neuron_sdpa_mask.py

One core, eager. q, k, v [1, 4, 256, 128] bf16; key padding masks (the last 200 keys masked,
and all keys masked for a block of keys as with an empty glyph stream), as boolean (True =
attend) and as additive float (0 / -inf), in the shapes diffusers' processors build:
[B, 1, 1, S] and [B, 1, L, S]. Each against the CPU result on the same inputs; plus the
unmasked call, and the masked call against an explicit slice of the valid keys.
"""

from __future__ import annotations

import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_neuronx  # noqa: E402,F401

B, H, L, S, D = 1, 4, 256, 256, 128
g = torch.Generator().manual_seed(0)
q, k, v = (torch.randn(B, H, n, D, generator=g).to(torch.bfloat16) for n in (L, S, S))
valid = torch.zeros(S, dtype=torch.bool)
valid[:56] = True  # 56 valid keys, 200 masked

cases = {}
key_bool = valid.view(1, 1, 1, S)
cases["bool [B,1,1,S]"] = key_bool
cases["bool [B,1,L,S]"] = key_bool.expand(B, 1, L, S).contiguous()
add = torch.zeros(S, dtype=torch.bfloat16).masked_fill(~valid, float("-inf"))
cases["additive [B,1,1,S]"] = add.view(1, 1, 1, S)
cases["additive [B,1,L,S]"] = add.view(1, 1, 1, S).expand(B, 1, L, S).contiguous()

# Query-dependent masks, as a token refiner builds them: valid rows attend to valid keys; padded
# rows would attend to nothing, so key 0 is forced on for every row (no fully masked row).
qvalid = torch.zeros(L, dtype=torch.bool)
qvalid[:56] = True
sym = (qvalid[:, None] & valid[None, :]).clone()
sym[:, 0] = True
cases["bool symmetric [B,1,L,S]"] = sym.view(1, 1, L, S)
cases["bool symmetric [B,H,L,S]"] = sym.view(1, 1, L, S).expand(B, H, L, S).contiguous()
cases["additive symmetric [B,H,L,S]"] = torch.zeros(L, S, dtype=torch.bfloat16).masked_fill(
    ~sym, float("-inf")).view(1, 1, L, S).expand(B, H, L, S).contiguous()

dev = [t.to("neuron") for t in (q, k, v)]


def rel(a, b):
    return ((a.float() - b.float()).abs().mean() / b.float().abs().mean()).item()


ref_none = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
out_none = F.scaled_dot_product_attention(*dev).cpu()
print(f"no mask:            rel vs CPU {rel(out_none, ref_none):.2e}", flush=True)
sliced = F.scaled_dot_product_attention(q.float(), k[:, :, :56].float(), v[:, :, :56].float())
for name, mask in cases.items():
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                         attn_mask=mask if mask.dtype == torch.bool else mask.float())
    try:
        out = F.scaled_dot_product_attention(*dev, attn_mask=mask.to("neuron")).cpu()
        finite = bool(torch.isfinite(out).all())
        rows = rel(out[:, :, :56], ref[:, :, :56]), rel(out[:, :, 56:], ref[:, :, 56:])
        print(f"{name:30s} finite={finite} rel vs CPU {rel(out, ref):.2e} (valid rows {rows[0]:.1e}, "
              f"padded rows {rows[1]:.1e}); device vs unmasked {rel(out, ref_none):.2e}", flush=True)
    except Exception as exc:  # report and continue
        print(f"{name:20s} ERROR {type(exc).__name__}: {str(exc)[:150]}", flush=True)
