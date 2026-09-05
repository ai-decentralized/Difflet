# Device evidence — TeaCache probes with canonical weight keys

Branch: `refactor/teacache-probe-canonical-keys` (based on `main` @ `c9efd8d`).
Hardware: trn2.3xlarge (`i-0672d11bbf8e50111`), 4 NeuronCores, LNC=2, tp=4.
Date: 2026-09-03.

**The raw logs are not committed.** They exist only on the verification
host, under `/tmp/logs/`:

| What | Host path |
|---|---|
| HunyuanVideo probe compile + load | `/tmp/logs/hv_probe.log` |
| FLUX adaptive end-to-end | `/tmp/logs/flux_e2e.log` |
| Qwen-Image probe compile | `/tmp/logs/qwen_probe.log` |
| Prior store-collision incident | `/tmp/logs/ab/hv_failed_store_collision/` |
| FLUX harness JSON results | `cclogs/m9-teacache/` in the run's worktree |

The store listings, inode/link-count dumps and safetensors-header dumps
quoted below came from read-only commands run during the session and were
kept only in an uncommitted scratch tree; "How to inspect manually" at the
end gives the commands that regenerate them. Every number in this document
is transcribed from those sources.

Plan this executes: `docs/verification/teacache-probe-canonical-keys-plan.md`.

## What was being verified

The TeaCache probe applications now **subclass** their backbone model classes
instead of wrapping them, so a probe's traced weight names are exactly its
backbone's shard keys. Therefore the shared weight store
(`difflet/backends/trainium/core/shared_weights.py`) can serve a probe from the
backbone's pre-sharded checkpoint:

* no `weights_layout_tag` in the store key (supersedes campaign `3f04080` and
  the independently re-applied `3972c8c` / `698449e`), and
* one physical copy of the transformer shards instead of two (issue #39).

The single probe-only tensor, `prev_mod`, is aliased to an output, so
`torch_neuronx` classifies it as INPUT_STATE: NxD's `StateInitializer`
allocates it zero-filled at load and never looks it up in the checkpoint. It is
declared in `<ProbeApp>.state_tensor_names`.

## The bug this replaces, as it actually manifested on this machine

Before this branch the shared store keyed probe and backbone identically **but
their checkpoints used different key namespaces**, so whichever compiled first
published its shards and the other was handed weights it could not read. Both
directions were observed here:

1. **Backbone poisoned by probe-layout shards.** The HunyuanVideo A/B run at
   09:36–09:40 failed on every generate with

   ```
   RuntimeError: Missing weight tensor with keyproj_out.bias
   ```

   raised from `neuronx_distributed/trace/spmd.py:185` inside
   `nxd_model.initialize`. Preserved verbatim on the host under
   `/tmp/logs/ab/hv_failed_store_collision/` (`baseline.log`,
   `baseline_warm.log`, `cadence2.log`, `online06.log`) — not committed. The
   recovery left a quarantined directory still on disk, which is itself
   evidence:

   ```
   .../hunyuan_video_dit/c665594454b2a584/transformer/weights.quarantined-probe-layout-20260903/
   ```

2. **Two store entries for one transformer**, once the layout tag was
   introduced to work around (1) — the duplication issue #39 tracks.

### The two entries, before this branch's probe was compiled

Store listing taken before the probe compile, and the key namespaces read
straight out of the safetensors headers (see "How to inspect manually"):

| Store entry | Per-shard size | Keys | Namespace | Written by |
|---|---:|---:|---|---|
| `…__tp4__7630a3d168d47af0` | 10,690,487,632 B | 1184 | **all `model.`-prefixed** | the old **probe** |
| `…__tp4__ad736a671a4f48be` | 11,634,459,808 B | 1304 | backbone names | the **backbone** |

Both are the same HunyuanVideo transformer at the same dtype and topology:
**40 GB + 44 GB = 84 GB for one model's weights.**

The 120-key difference is a second, quieter defect of the old design: the HV
probe's converter only prefixed keys with `model.` and never applied the
backbone's single-block `proj_out` → `proj_out_attn` / `proj_out_mlp` split, so
its checkpoint carried 40 blocks' worth of tensors the probe model does not
have and lacked the ones it does. NxD's `preprocess_checkpoint` silently drops
unknown checkpoint keys, and the probe NEFF dead-code-eliminates the single
blocks, so this never surfaced as an error — the probe simply ran with an
incomplete checkpoint. Reusing the backbone's converter object fixes it by
construction.

## Proof 1 — one store entry, by key, with production configs

`store_key` computed from the **real** production config objects (no device,
no weights loaded):

```
HunyuanVideo   backbone / probe_fused / probe_v1  -> ad736a671a4f48be   (ONE entry)
FLUX.1-dev     backbone / probe_fused             -> 725b456c1a7101e9   (ONE entry)
converter identity (probe is backbone's function object): True
state_tensor_names: backbone=set()  probe_fused={'prev_mod'}
```

`ad736a671a4f48be` is the **existing backbone entry**. The old probe entry
`7630a3d168d47af0` is not reachable from any application on this branch — it is
orphaned, and is exactly the ~40 GB issue #39 is about.

Reproduce: build the backbone and probe applications from the production
configs and compare `store_key()`. The same property is asserted on CPU by
`tests/unit/backends/test_shared_weights.py`
`::test_probe_and_backbone_apps_share_one_store_entry`.

## Proof 2 — HunyuanVideo: one entry, shared inodes, on device

Job: `DiffletPipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo",
application_kwargs={"teacache_fused": True})` at 320×512×121, tp=4, bf16.
Full log (host only): `/tmp/logs/hv_probe.log`. The stale probe component
from the previous design was moved aside first (to
`/tmp/stale_hv_teacache_probe_20260903`, not deleted) so the probe recompiled
against this branch's code.

Both components of the new artifact `~/.cache/difflet/hunyuan_video/f3739da90408fde1/`
linked **the same** store entry — the backbone's:

```
:12  Saving the neuron_config to .../f3739da90408fde1/transformer/
:50  Reusing pre-sharded checkpoints from .../HunyuanVideo__transformer__…__ad736a671a4f48be.
:51  Saving the neuron_config to .../f3739da90408fde1/teacache_probe/
:91  Reusing pre-sharded checkpoints from .../HunyuanVideo__transformer__…__ad736a671a4f48be.
```

No `Pre-sharding checkpoints.` for either component: nothing was re-sharded and
no new bytes were written. The link count on that entry's shards went
**2 → 4** (`hv_store_before.log` vs `hv_store_after.log`), and the two
components' shard files are literally the same inodes:

```
4 links  inode 525519  11634459808 B  .../f3739da90408fde1/transformer/weights/tp0_sharded_checkpoint.safetensors
4 links  inode 525519  11634459808 B  .../f3739da90408fde1/teacache_probe/weights/tp0_sharded_checkpoint.safetensors
   (likewise tp1→525520, tp2→525521, tp3→525625)
```

The probe's shard, read through the probe's own path
(`hv_probe_shard_header.log`), carries backbone names and no state tensor:

```
keys=1304   nested(model.)=0   nested(trace_module.)=0   prev_mod present: False
```

The orphaned probe-layout entry `7630a3d168d47af0` gained no references: on this
branch nothing can address it. **Deleting it reclaims ~40 GB** (left in place —
cache deletion needs the user's approval).

**Weight init succeeded.** Both components read the identical 44,381.9 MB of
shards, and the first `nxd_model.initialize` completed with no
`Missing weight tensor`:

```
:93  Presharded file read: 3.72s for 4 shard(s) (44381.9 MB total)
:95  Finished traced model weight initialization in 634.19s
:98  Presharded file read: 0.14s for 4 shard(s) (44381.9 MB total)   <- the probe, same bytes
```

### Known limit found (not a regression): HV probe + backbone exceed device HBM

The second `initialize` — the probe's — aborted with

```
Failure: NRT_RESOURCE in nrt_tensor_allocate   /   what(): nrt_tensor_allocate status=4
```

This is device memory, not weight naming — the shard *read* succeeded twice
(same 44,381.9 MB both times) and no key was reported missing. The fused probe
holds a **full** transformer parameter set on device: XLA dead-code-eliminates
the unused layers from the NEFF, but `nxd_model.initialize` still materialises
the checkpoint it is handed. The arithmetic, all measured on this machine:

| Quantity | Value |
|---|---:|
| Per-rank shard | 11,634,459,808 B = 11.63 GB |
| Backbone, 4 ranks | 46.54 GB |
| Probe, 4 ranks (this branch, same shards) | 46.54 GB |
| Co-resident total | **93.1 GB** |
| Device HBM (trn2.3xlarge, 4 cores × 24 GB) | **96 GB** |
| Left for activations, text encoders, VAE | ~2.9 GB → **insufficient** |

It predates this branch rather than being caused by it: the *old* probe's
checkpoint was 10.69 GB/shard = 42.8 GB, so the pair was 46.54 + 42.8 =
**89.3 GB**, also leaving too little headroom. It is consistent with the earlier
HunyuanVideo adaptive attempt on this machine aborting with SIGABRT after 595 s
(`/tmp/logs/hv_adaptive.supervised.log`).

**Conclusion for HV:** the weight-key and store-dedup contract is verified on
device; adaptive (probe-based) TeaCache for HunyuanVideo is blocked on this
instance by HBM capacity and should use the probe-free fixed cadence.

## Proof 3 — FLUX: one entry, and an end-to-end run that matches the reference

Job: `scripts/run_flux_teacache_e2e.py` (harness fixes `4a16b0a` and `3dcc2d1`
cherry-picked onto this branch), 1024×1024, 28 steps, tp=4, bf16, seed 0.
Full log (host only): `/tmp/logs/flux_e2e.log`; the harness wrote
`flux_teacache_e2e.json` and `calibration_flux_1024_28step.json` into
`cclogs/m9-teacache/` in the run's worktree (gitignored scratch, not
committed).

FLUX was compiled from scratch, so the transformer sharded and **published** the
entry, and the probe then **reused** it:

```
:120  Saving the neuron_config to .../flux/6f06610c4db43fb0/transformer/
:175  Pre-sharding complete in 217.0s: 4 shard file(s), 22708.9 MB total under .../transformer/weights/
:227  Saving the neuron_config to .../flux/6f06610c4db43fb0/teacache_probe/
:265  Reusing pre-sharded checkpoints from .../FLUX.1-dev__transformer__…__725b456c1a7101e9.
```

That `Reusing` at :265 is the **only** one in the FLUX log: the transformer
published the entry, the probe consumed it, and nothing was written twice.

**22,708.9 MB is exactly the ~22.7 GB issue #39 is about, and it was written
once.** All three paths are the same inodes (`flux_hardlinks.log`):

```
3 links  inode 531875  5953008676 B  _shared_weights/…__725b456c1a7101e9/shard0.safetensors
3 links  inode 531875  5953008676 B  flux/6f06610c4db43fb0/transformer/weights/tp0_sharded_checkpoint.safetensors
3 links  inode 531875  5953008676 B  flux/6f06610c4db43fb0/teacache_probe/weights/tp0_sharded_checkpoint.safetensors
   (likewise tp1→531876, tp2→531877, tp3→531878)
```

The pipeline then **loaded** — probe included — in 1359 s with no
`Missing weight tensor`, and ran the full gate → calibrate → A/B:

| Metric | This branch (shared shards) | Campaign `920585a` (duplicated shards) |
|---|---|---|
| Signal gate | 81 pairs, Pearson **0.6847** | 81 pairs, Pearson **0.6847** |
| Calibration threshold | **0.170** | **0.170** |
| Steps skipped | **52 / 112** (13 per prompt) | **52 / 112** |
| Final-latent cosine | **0.875093** | **0.8751** |
| Wall-clock speedup | **1.830×** (30.51 s → 16.67 s) | 1.881× |

Every deterministic quantity — signal correlation, fitted threshold, skip
schedule, output cosine — is identical to the reference. Only wall-clock
speedup differs (1.830× vs 1.881×, 2.7%), which is timing variance, not a
behavioural change. **The canonical-key probe is behaviour-preserving and
costs one copy of the weights instead of two.**

## Proof 4 — Qwen-Image

Job: compile + shard with `teacache_fused=True`, `load=False`
(host log `/tmp/logs/qwen_probe.log`). Qwen-Image's MMDiT is ~20B: backbone
and probe would each materialise a full parameter set on device, which exceeds
this instance's HBM for the same reason as HunyuanVideo, so this job verifies
the part under test — compile-time weight naming and store dedup — and does not
attempt a co-resident load. The campaign's own Qwen adaptive result was
obtained through a transformer-only harness for the same reason, and reported a
weak signal (fit R² = 0.125, 0/400 steps skipped), which is a property of Qwen's
text-independent block-0 modulation rather than of the weight path.

**Result: PASS.** The transformer sharded once and the probe reused it
(`/tmp/logs/qwen_probe.log`):

```
:54   Pre-sharding checkpoints.
:71   Pre-sharding complete in 474.3s: 4 shard file(s), 38987.1 MB total under
        .../qwen_image/b6f5bb86addcd1f6/transformer/weights/
:72   Saving the neuron_config to .../qwen_image/b6f5bb86addcd1f6/teacache_probe/
:116  Reusing pre-sharded checkpoints from .../Qwen--Qwen-Image__transformer__…__8ad7cab52bbb1b05.
```

There is no second `Pre-sharding complete` for `teacache_probe/weights/`:
**38,987.1 MB (39.0 GB) was written once and serves both components.** The
application objects agree on the key, and the probe reuses the backbone's
converter function object:

```
store_key backbone=8ad7cab52bbb1b05
store_key probe   =8ad7cab52bbb1b05
SAME ENTRY: True        converter identity: True
state_tensor_names={'prev_mod'}
```

Inodes (`qwen_hardlinks.log`) — store, backbone artifact and probe artifact are
one physical copy, 3 links each:

```
3 links  inode 528575  10220246384 B  _shared_weights/…__8ad7cab52bbb1b05/shard0.safetensors
3 links  inode 528575  10220246384 B  qwen_image/b6f5bb86addcd1f6/transformer/weights/tp0_sharded_checkpoint.safetensors
3 links  inode 528575  10220246384 B  qwen_image/b6f5bb86addcd1f6/teacache_probe/weights/tp0_sharded_checkpoint.safetensors
   (likewise tp1→528576, tp2→528581, tp3→528582)
```

The probe's shard (`qwen_probe_shard_header.log`) carries the backbone's
namespace and no state tensor:

```
keys=1933   top-level prefixes: {'transformer': 1933}
nested(model.)=0   nested(trace_module.)=0   prev_mod present: False
```

## Summary — store dedup per model

| Model | Store entries for the topology | Entry | Shard inodes (backbone = probe) | Link count | Probe shard keys | `Missing weight tensor`? |
|---|---|---|---|---|---:|---|
| HunyuanVideo | **1** (was 2) | `ad736a671a4f48be` | 525519 / 525520 / 525521 / 525625 | 2 → **4** | **1304**, backbone-named, no `prev_mod` | **none** |
| FLUX.1-dev | **1** | `725b456c1a7101e9` | 531875 / 531876 / 531877 / 531878 | **3** | backbone-named | **none** |
| Qwen-Image | **1** | `8ad7cab52bbb1b05` | 528575 / 528576 / 528581 / 528582 | **3** | **1933**, all `transformer.`-prefixed, no `prev_mod` | **none** (compile-only; see Proof 4) |

End-to-end, FLUX (the only model whose backbone + probe fit in HBM):

| Metric | This branch | Campaign reference | Verdict |
|---|---|---|---|
| Signal Pearson | 0.6847 | 0.6847 | identical |
| Threshold | 0.170 | 0.170 | identical |
| Steps skipped | 52/112 | 52/112 | identical |
| Final cosine | 0.875093 | 0.8751 | identical |
| Speedup | 1.830× | 1.881× | −2.7 %, timing jitter |

Bytes saved, measured — in each case the second copy was simply never written:

| Model | Duplicate avoided |
|---|---:|
| FLUX.1-dev | 22,708.9 MB (**22.7 GB**) |
| Qwen-Image | 38,987.1 MB (**39.0 GB**) |
| HunyuanVideo | 44,381.9 MB (**44.4 GB**) |

Plus the pre-existing HunyuanVideo probe-layout entry `7630a3d168d47af0`
(**~40 GB**), which this branch orphans. It is **left in place on disk —
deleting it requires the user's approval.**

## Layout tag: no longer needed anywhere

This branch contains **no** `weights_layout_tag`. `shared_weights._key_inputs`
has no layout field, and `tests/unit/backends/test_shared_weights.py::
test_probe_and_backbone_apps_share_one_store_entry` pins that even a stray
`weights_layout_tag` attribute left on an application is ignored. The device
runs above show the store correctly serving probe and backbone from one entry
without it, so `3f04080` and the independently re-applied `3972c8c` / `698449e`
are superseded and can be dropped.




## How to inspect manually

Everything below is read-only and needs no device. The helper scripts used
during the run were scratch and are not committed; the equivalent commands are
inlined here.

1. **Which namespace is in a shard?** Dump the safetensors header:

   ```bash
   python - ~/.cache/difflet/<component>/<key>/teacache_probe/weights/tp0_sharded_checkpoint.safetensors <<'PY'
   import collections, json, struct, sys
   for path in sys.argv[1:]:
       with open(path, 'rb') as f:
           h = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
       keys = [k for k in h if k != '__metadata__']
       print(path, 'keys=%d' % len(keys))
       print('  top-level prefixes:',
             dict(collections.Counter(k.split('.')[0] for k in keys).most_common(8)))
       print('  nested(model.)=%d nested(trace_module.)=%d' % (
           sum(k.startswith('model.') for k in keys),
           sum(k.startswith('trace_module.') for k in keys)))
       print('  prev_mod present:', any(k == 'prev_mod' or k.endswith('.prev_mod') for k in keys))
   PY
   ```

   The probe's shard must report `nested(model.)=0`, `nested(trace_module.)=0`
   and `prev_mod present: False` — backbone names only, and no state tensor in
   the checkpoint. A non-zero `nested(...)` count is the old design.

2. **Are the bytes shared?** Compare inode numbers and link counts:

   ```bash
   STORE=~/.cache/difflet/_shared_weights
   ART=~/.cache/difflet/<component>/<artifact-key>
   stat -c '%h links  inode %i  %s B  %n' \
     "$STORE"/*__tp4__*/shard*.safetensors \
     "$ART"/transformer/weights/tp*_sharded_checkpoint.safetensors \
     "$ART"/teacache_probe/weights/tp*_sharded_checkpoint.safetensors
   du -sh "$STORE"/*/
   ```

   The `inode` of `transformer/weights/tp{r}_sharded_checkpoint.safetensors`,
   of `teacache_probe/weights/tp{r}_sharded_checkpoint.safetensors`, and of the
   store's `shard{r}.safetensors` must all be **the same number**. The link
   count is the number of references to that one physical copy; it rises as
   artifacts are added and no bytes are written.

3. **Is there only one entry per (model, component, dtype, topology)?**

   ```bash
   ls -1d ~/.cache/difflet/_shared_weights/*/ | sed 's/__[0-9a-f]\{16\}\/$//' | sort | uniq -c
   ```

   Any count > 1 is a duplicated model. Before this branch HunyuanVideo's
   transformer showed 2.

4. **Do the keys agree without hardware?** The unit tests state the same
   contract on CPU:

   ```bash
   source .venv/bin/activate
   python -m pytest tests/unit/models/flux/test_flux_teacache_probe_keys.py \
     tests/unit/models/qwen_image/test_qwen_teacache_probe_keys.py \
     tests/unit/models/hunyuan_video/test_hunyuan_video_teacache_probe_keys.py \
     tests/unit/backends/test_shared_weights.py -q
   ```
