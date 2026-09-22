# Fused on-device TeaCache probe for Wan and LTX-2 — verification evidence

**Campaign date:** 2026-09-22
**Branch:** `feat/wan-ltx2-teacache-device-probe`
**Host:** trn2.3xlarge, 1 Neuron device, 4 cores, 96 GB device memory
**Runtime env:** `/home/ubuntu/difflet-vj/.venv` (torch 2.9.1, neuronx_distributed present).
Note that this is *not* `<repo>/.venv`; that path does not exist in this checkout, so every
command below sets `PYTHONPATH=/home/ubuntu/Difflet` to put the working tree ahead of the
`difflet` package installed in that venv from a different checkout.

## What changed and why it needed verifying

Before this branch, Wan and LTX-2 were the only models computing the block-0 TeaCache signal on
the host CPU. Wan used a lean shadow (one layer, only the patch-embed / condition-embedder /
block-0 norm weights). LTX-2 single mode loaded a **full** bf16 host copy of the 48-block
transformer whose only caller was `teacache_mod_input`.

This branch gives both the fused-A probe that HunyuanVideo, Qwen-Image and Flux already use:
`prev_mod` is a persistent on-device `nn.Parameter` aliased to output 1, the forward returns
`(rel_l1, mod_input)`, and only the 4-byte scalar crosses to host.

The risk the verification targets is specific. Both probes **subclass** their backbone so their
traced parameter names stay byte-identical to the backbone's, which is what lets the shared
weight store hand them the backbone's pre-sharded checkpoint instead of making a second copy.
If that parity is wrong, the failure is a hard runtime error at `nxd_model.initialize`:
`Missing weight tensor with key ...`. Flux hit exactly that on 2026-08-30 with a wrapper design
(GitHub issue #39). So the load step is the real test, not the compile step.

## Phase 1 — host-side weight-name contract (no device)

Two levels: synthetic tiny configs in the unit suite, then the **real** downloaded checkpoints.

### Unit tests

```
PATH=/home/ubuntu/difflet-vj/.venv/bin:$PATH \
/home/ubuntu/difflet-vj/.venv/bin/python -m pytest \
  tests/unit/models/wan/test_wan_teacache_probe_keys.py \
  tests/unit/models/ltx_2/test_ltx_2_teacache_probe_keys.py -q
```

| Suite | Result |
|---|---|
| `test_wan_teacache_probe_keys.py` | 11 passed |
| `test_ltx_2_teacache_probe_keys.py` | 11 passed |

Each suite pins: the probe is a subclass rather than a wrapper; its state dict equals the
backbone's plus exactly `prev_mod`; the backbone's converted checkpoint serves every probe
weight; the aliased tensors are exactly the declared state; and the probe's rel-L1 equals the
host formula on the same weights.

### Real checkpoints

`checkpoint_missing_weights(probe, converted_real_keys, {"prev_mod"})` against the key set in
each model's `diffusion_pytorch_model.safetensors.index.json`. This is the host-side form of the
on-device error, run before spending any device time.

| Model | Probe tensors | Checkpoint keys | Missing with `prev_mod` declared | Missing without |
|---|---|---|---|---|
| Wan 2.1 T2V 14B | 1096 | 1096 | none | `prev_mod` |
| LTX-2 | 3511 | 3511 | none | `prev_mod` |

The right-hand column proves the check is not vacuous: `prev_mod` is the single tensor that is
NEFF state rather than a checkpoint weight, exactly as intended.

## Phase 2 — on-device compile, load and parity

Runners added by this branch, both modelled on `scripts/run_hv_teacache_fused_smoke.py`:

- `scripts/run_wan_teacache_fused_smoke.py`
- `scripts/run_ltx2_teacache_fused_smoke.py`

Each one compiles **only** the probe component, loads it, and then checks four things:

1. **Compile.** The alias on the `prev_mod` Parameter is accepted.
2. **Load.** No missing-weight error, so the probe's names resolved against the backbone's shards.
3. **`prev_mod` persists on device.** Call the probe twice with the *same* input. The first call
   sees a zero-filled `prev_mod` and returns a large garbage value, which the controller's warmup
   window absorbs in real use. The second call must return approximately zero, because the alias
   wrote `mod_input` into `prev_mod` in place. A second call that does *not* collapse means the
   alias silently did nothing, and the whole fused design is unsound.
4. **Parity with the host path.** With `prev_mod` holding `mod_input(A)`, probing latent B yields
   the device's rel-L1 of B against A. The same two latents go through the host path being
   replaced (Wan's CPU shadow; LTX-2's host CPU transformer), and the host rel-L1 is computed with
   the pipeline's own formula, `mean|cur - prev| / mean|prev|`. These two numbers are what the
   TeaCache controller consumes, so they are the ones that must agree.

Timings are median over 20 calls for the device probe and over the host path, measured in the
same process on the same inputs.

### Reproduce

```
cd /home/ubuntu/Difflet
PATH=/home/ubuntu/difflet-vj/.venv/bin:$PATH PYTHONPATH=/home/ubuntu/Difflet \
HF_HOME=/home/ubuntu/hf NEURON_RT_NUM_CORES=4 \
/home/ubuntu/difflet-vj/.venv/bin/python scripts/run_wan_teacache_fused_smoke.py \
  --result artifacts/verification-2026-09-22/wan_teacache_fused_smoke.json
```

```
cd /home/ubuntu/Difflet
PATH=/home/ubuntu/difflet-vj/.venv/bin:$PATH PYTHONPATH=/home/ubuntu/Difflet \
HF_HOME=/home/ubuntu/hf NEURON_RT_NUM_CORES=4 \
/home/ubuntu/difflet-vj/.venv/bin/python scripts/run_ltx2_teacache_fused_smoke.py \
  --result artifacts/verification-2026-09-22/ltx2_teacache_fused_smoke.json
```

### Results

#### Wan 2.1 T2V 14B — PASS

Evidence: `artifacts/verification-2026-09-22/wan_probe_smoke.log`,
`artifacts/verification-2026-09-22/wan_teacache_fused_smoke.json`.
Shape 480x832, 21 latent frames, sequence length 32760, inner dim 5120, tp 4. Runner exit 0.

| Check | Result |
|---|---|
| Compile | 448.9 s |
| Load | 11.6 s, no missing-weight error |
| Call 1 delta, `prev_mod` zero-filled | 43654672.0 |
| Call 2 delta, same input | 0.0 |
| `prev_mod` persists on device | yes |
| Device rel-L1(B vs A) | 0.229369 |
| Host shadow rel-L1(B vs A) | 0.229291 |
| Relative difference | 0.034% |
| Device probe, median of 20 | 27.42 ms/call |
| Host CPU shadow, median of 20 | 941.31 ms/call |

Call 2 returning exactly 0.0 is the strongest single result here: it can only happen if the
`input_output_aliases` entry wrote `mod_input` into `prev_mod` in place on device between calls.
Had the alias silently done nothing, `prev_mod` would still be zero and call 2 would repeat call
1's value.

The 0.034% gap between device and host is bf16 rounding, not a semantic difference. The probe
inherits `WanTransformer3DModel.teacache_mod_input` rather than reimplementing it, so the two
paths run the same arithmetic on the same weights and differ only in accumulation order.

**The timing result contradicts the prior assumption in this repo and is worth stating plainly.**
`teacache_cpu_shadow.py` records that the HunyuanVideo probe NEFF cost ~51 ms/step of dispatch
against single-digit milliseconds on host, and that measurement is why Wan and LTX-2 were built
with host shadows. It does not transfer to Wan at this shape. The shadow has to run the patch
embedding and a norm over a 32760 x 5120 activation on CPU, which costs 941 ms/call, while the
device probe costs 27 ms/call — the device path is about 34x faster. The earlier figure was
measured on a different model at a different sequence length; it was never a general claim about
host-versus-device probes, and it should not be read as one.

#### LTX-2 single mode — PASS

Evidence: `artifacts/verification-2026-09-22/ltx2_probe_smoke.log`,
`artifacts/verification-2026-09-22/ltx2_teacache_fused_smoke.json`.
Shape 512x768, 121 frames, sequence length 6144, inner dim 4096, tp 4. Runner exit 0.

| Check | Result |
|---|---|
| Compile | 564.8 s |
| Load | 118.4 s, no missing-weight error |
| Call 1 delta, `prev_mod` zero-filled | 28423100.0 |
| Call 2 delta, same input | 0.0 |
| `prev_mod` persists on device | yes |
| Device rel-L1(B vs A) | 0.144422 |
| Host CPU transformer rel-L1(B vs A) | 0.144454 |
| Relative difference | 0.022% |
| Device probe, median of 20 | 3.05 ms/call |
| Host CPU transformer, median of 5 | 26.70 ms/call |

The clean load is the result that mattered most for LTX-2, because its backbone is *already* a
wrapper: `_LTX2TransformerTraceModule` holds the diffusers model at `self.transformer`, and the
converter prefixes every checkpoint key with `transformer.`. The probe subclasses that trace
module and adds only `prev_mod`, so its keys land at exactly the depth the converter emits. Had
the probe wrapped the trace module instead, every key would have gained a second level and the
load would have failed with `Missing weight tensor with key ...`.

The device path is about 8.8x faster per call here. Its benefit is not only latency: in single
mode `_load_cpu_transformer` had exactly one caller, `teacache_mod_input`, so with the probe
mounted the runtime no longer needs a full bf16 host copy of the 48-block transformer at all.
That host copy is 37.8 GB of weights.

### Summary

| Model | Compile | Load | `prev_mod` persists | Device vs host rel-L1 | Device ms/call | Host ms/call | Outcome |
|---|---|---|---|---|---|---|---|
| Wan 2.1 T2V 14B | 448.9 s | 11.6 s | yes | 0.034% apart | 27.42 | 941.31 | PASS |
| LTX-2 single | 564.8 s | 118.4 s | yes | 0.022% apart | 3.05 | 26.70 | PASS |

Both device probes produce the number the TeaCache controller consumes, to within bf16 rounding
of the host path they replace, and both are faster than that host path on this hardware.

## Host budget after the campaign

Disk went from 80 GB free to **15 GB free (98% full)**. Nothing was deleted.

**The shared weight store worked.** An earlier revision of this document said each probe "wrote
its own shard set rather than attaching to a backbone's weight-store entry". That was wrong, and
it was wrong because `du` counts a hardlinked inode only once per invocation, so measuring a
directory on its own attributes the shared bytes to it. The shards exist **once**; the probe
artifact directory holds hardlinks to them:

```
wan21_teacache_probe_fused/weights/tp0_sharded_checkpoint.safetensors   2 links  ino=567020
_shared_weights/Wan-AI--…__38ec498c__bfloat16__tp4__f12d…/shard0.safetensors  2 links  ino=567020
```

Same inode, link count 2, 7498045044 bytes each across 4 shards. `du --total` over the two probe
directories *and* `_shared_weights` together reports 135 GB, not the ~203 GB that independent
copies would produce.

The practical consequence matters more than the bookkeeping: **deleting a probe artifact
directory reclaims almost nothing**, because the `_shared_weights` link keeps the inode alive,
and vice versa. Freeing those bytes requires removing both links.

| Store entry | Size | Written |
|---|---|---|
| `Qwen--Qwen-Image__transformer__75e0b4be__bfloat16__tp4__…` | 39 GB | 2026-09-21 |
| `Lightricks--LTX-2__transformer__dfcc2108__bfloat16__tp4__…` | 38 GB | this campaign |
| `Wan-AI--Wan2.1-T2V-14B-Diffusers__38ec498c__bfloat16__tp4__…` | 28 GB | this campaign |
| `black-forest-labs--FLUX.1-dev__transformer__3de623fc__bfloat16__tp4__…` | 23 GB | 2026-09-21 |

### Known defect in this campaign's Wan artifact

The Wan entry above has **no `__transformer__` component segment**, while LTX-2, Qwen and FLUX
all have one. That is not a store bug — it is a defect in this campaign's runner.

`_key_inputs` keys on `os.path.realpath(app.model_path)` (shared_weights.py:96), and
`difflet/models/wan/application.py` builds both the backbone and the probe with
`model_path=self.transformer_path`. The first version of
`scripts/run_wan_teacache_fused_smoke.py` passed the **model root** instead. The weights still
resolved correctly — the 0.034% parity against a shadow built from `<snap>/transformer` proves
the probe read the right tensors — but the entry was filed under a key that **no production Wan
run will ever look up**. A real Wan backbone or application-mounted probe at bf16/tp4 will miss
it and re-shard the same 28 GB under the correct key.

So that 28 GB entry is effectively orphaned, and is the one item in the table above that is safe
to reclaim on those grounds. The runner is fixed to pass the transformer directory; the fix is
**not re-verified on device**, because a rerun would write a second 28 GB entry and this host has
15 GB free. The Wan functional results are unaffected: compile, load, alias persistence and
host parity all hold regardless of the store key.

The LTX-2 runner passed `<snap>/transformer` and matches `difflet/models/ltx_2/application.py`,
so its entry is correctly keyed and would be reused.

### Still not measured

Store **sharing between a probe and its backbone** — the probes were compiled with no backbone
alongside, so each created the entry rather than attaching to an existing one. Proving the attach
path needs the backbone compiled first and a link-count rise on the topology's
`shard0.safetensors`. That is **NOT MEASURED** here.

## Scope and limits

- **Wan is 2.1, not 2.2.** The downloaded snapshot is `Wan-AI/Wan2.1-T2V-14B-Diffusers`, which has
  a single `transformer` directory and no `transformer_2`. The two-expert path, where a probe is
  mounted per stage and the controller resets its residual on the stage switch, is therefore
  **NOT MEASURED** here. The code builds a probe per present stage, so 2.2 would mount two.
- **One compiled shape per probe.** `prev_mod` is a fixed-shape Parameter, so the Wan probe covers
  the primary compile shape only. Other buckets fall back to the host CPU shadow at runtime via
  `_probe_covers`. Multi-shape behaviour is **NOT MEASURED**.
- **LTX-2 segmented mode is out of scope by design.** There the host CPU transformer is
  load-bearing for `_prepare_frontend` and `_final_projection`, so the signal is already free and
  the application rejects `teacache_fused` with an explanatory error.
- **CFG parallel is rejected for Wan**, matching Flux: a single probe and a single skip decision
  cannot represent branches scattered across DP ranks, and the Wan pipeline already refuses
  TeaCache in that mode.
- **No end-to-end quality run.** These runners verify the signal, not generated video. A full
  generate with an adaptive calibration, and the resulting speedup and fidelity, is a separate
  phase and is **NOT MEASURED**.
