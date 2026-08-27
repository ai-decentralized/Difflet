# Task 2 Verification — Shape-set artifact identity (hash dirs + manifests + cache ls)

**Status: VERIFIED on device 2026-08-16 (chain log below).**

Branch: `feat/bucketed-artifacts` · Host: trn2.3xlarge · Model: HunyuanVideo tp4 bf16
Scratch: `/home/ubuntu/.claude/jobs/e22aa8d9/tmp/` (`$LOG`; logs `t2_*.log`)

## What changed

- **Schema v5** (`difflet/pipeline/compile_cache.py`): the cache key's single
  `"shape": {h, w, f}` is replaced by a canonical `"shapes": [[h, w, f], ...]` list — deduped,
  sorted largest-first via the same `canonicalize_shapes` the compile side uses, so set order and
  duplicates can never change the key. **K=1 uses the same list form** (unified naming decision:
  all v4 manifests intentionally read as cache misses and recompile once).
- **CLI artifact dirs are now pure hashes**: `<cache>/<component>/<sha256[:16]>`
  (e.g. `hunyuan_video_dit/9f3ab2c1d4e5f6a7`), replacing human-readable names like
  `hunyuan_video_dit_tp4cp1_h320w512f61`. The authoritative record is the `manifest.json` written
  into each dir (component, model, tp/cp/sp/cfg, dtype, shapes, toolchain). Helpers in
  `difflet/cli/orchestrators/base.py`; wired for HunyuanVideo + Wan stages (compile writes the
  manifest; generate refuses to load without a matching one).
- **Serving identities**: HunyuanVideo `_compile_specs` bumps `compile_contract_version` to 2 and
  keys denoiser/decoder on the canonical `shapes` list; llama/clip stay shape-independent. Wan/LTX-2
  serving identities inherit the v5 `shapes` form automatically through `CacheSpec.cache_inputs()`.
- **`difflet cache ls`** (new, `difflet/cli/cache_cmd.py`): human-readable index over the hash
  dirs — DIR / COMPONENT / MODEL / SHAPES / TP / DTYPE / SCHEMA / SIZE / UPDATED, plus `--json`.
- **`DiffletPipeline.from_pretrained(shapes=[...])`** (Flux/Qwen path): shape set flows into the
  CacheSpec key and the application (excluded from the app-kwargs hash to avoid double counting).
- **Concurrency fix found during verification** (`difflet/cli/runner.py`): every stage subprocess
  now gets a unique compiler scratch dir. The vendor `ModelBuilder.trace()` rmtree-s its workdir at
  start, and the old layout keyed scratch by component name only (`/tmp/nxd_model/transformer`), so
  two concurrent difflet compiles of different models deleted each other's in-flight scratch —
  observed as an NCC internal error killing a running Wan compile. Scratch is deleted after a
  successful stage and kept for post-mortem on failure.

## Unit tests

```bash
PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python -m pytest \
  tests/unit/cli/test_cache_identity.py tests/unit/pipeline/test_compile_cache.py tests/unit/cli -q
```
13 new tests (`test_cache_identity.py`): K=1 list form, order/dup invariance of the key, key
sensitivity to added shapes, app-kwargs exclusion, manifest round-trip/tamper/corrupt, hash-dir
layout, `cache ls` collection + JSON. Naming-scheme tests updated from literal names to
scheme+stability+sensitivity assertions. Full suite: 1992 passed (pre-existing failures on main:
1 teacache-kwargs test + video_service/storage timing tests, deselected — verified identical on
pristine main).

## On-device verification (`bash $LOG/task2_verify.sh`, logs `t2_*.log`)

| check | expectation | result |
|---|---|---|
| K=1 compile lands in hash dirs | `hunyuan_video_{clip,llama,dit}/<16-hex>/` each with manifest.json | ✅ `dit/4fd6868c9c51f80f`, `llama/4ff7e840828c0a45`, `clip/826aaffa4739891b` (75 min) |
| identical rerun | all stages skip, minutes not hours | ✅ 5.5 min (vs 75) — pure skip/validation pass |
| `--shapes A,B` then `--shapes B,A` | second run resolves the SAME dir and skips (order invariance on device) | ✅ swapped rerun 5.3 min, no new dir |
| dit/ dir count | exactly 2 hash dirs: the K=1 and the K=2 artifact | ✅ `4fd6868c9c51f80f` + `baff42a213213ac6` |
| generate shape B through new layout | bucketed artifact serves B | ✅ output sha256 `662e0fc9…c278` — **bit-identical to Task 1's fixed bucketed-B output** (relocation changed nothing) |
| manifest tamper | generate rejects; rerunning compile repairs the manifest | ✅ clear "`no valid compiled artifact for stage 'generate' at …`" then `manifest repaired: tp=4` |
| `difflet cache ls` | table shows hash → shapes/tp/dtype for all artifacts | ✅ see below / `$LOG/t2_cache_ls.txt` |

```
DIR                                   COMPONENT            SHAPES                 TP  DTYPE     SCHEMA  SIZE
hunyuan_video_clip/826aaffa4739891b   hunyuan_video_clip   -                      -   bfloat16  5       236.3 MiB
hunyuan_video_dit/4fd6868c9c51f80f    hunyuan_video_dit    320x512x61             4   bfloat16  5       43.8 GiB
hunyuan_video_dit/baff42a213213ac6    hunyuan_video_dit    320x512x61+320x512x33  4   bfloat16  5       43.8 GiB
hunyuan_video_llama/4ff7e840828c0a45  hunyuan_video_llama  -                      4   bfloat16  5       15.0 GiB
```
(The two 43.8 GiB dit rows share one physical weight set via the hardlink store.)

## Notes

- Old-named artifacts from Task 1 remain on disk but are no longer addressed; disk is reclaimable
  (weights are hardlinked into `_shared_weights`, so deleting old dirs frees little until the store
  itself is pruned). Legacy leftovers can sit next to the new prefix dirs (e.g. the old
  `hunyuan_video_clip/` artifact files beside `hunyuan_video_clip/<hash>/`) — `cache ls` reads only
  `*/*/manifest.json` and is not confused by them.
- Hash-dir migration cost: one recompile per configuration (measured 75 min for the full K=1
  HunyuanVideo artifact set, 72 min for the 2-shape bucketed artifact); subsequent runs are
  5-minute validation passes.
