# HunyuanVideo 1.0 on the neuron (torch-native) backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `difflet generate --model hunyuan_video` runs end to end on the neuron backend (TorchNeuron, one process per NeuronCore, torchrun) with TP, SP and CP (gather_kv, ulysses, ring), TeaCache and a device VAE, matching diffusers within the repo's parity gate.

**Architecture:** Follow the registry-application pattern Wan / Qwen-Image / LTX-2 use (not HV1.5's registry-bypassing pipeline): a `TorchNeuronApplicationBase` DiT that wraps Difflet's **backend-neutral** `HunyuanVideoTransformer3DModel` (`difflet/models/hunyuan_video/modeling_hunyuan_video.py`, which already builds TP/SP/CP from `difflet.ops`, and `difflet.ops` already dispatches to `difflet/backends/neuron/ops_impl`), composed by a `TorchNeuronModelApplication` with a rank-0 host text encoder and the existing `HunyuanVideoOrchestrator`. From HV1.5's neuron port we reuse the VAE application shape (diffusers' autoencoder, eager, one core, tiled), the stage layout (DiT launch, then a one-core vae launch), the TeaCache probe wiring and the manual-check/oracle methodology.

**Tech Stack:** torch 2.14 + torch-neuronx native (`/home/ubuntu/venvs/native-312`), diffusers 0.38, gloo CPU ranks for unit tests, trn2.3xlarge (4 logical cores × 24 GB HBM) for device checks.

**Spec:** none written — the user asked for "HunyuanVideo 1.0 torch-native (neuron) backend, studying HV1.5's implementation". The design decisions below are the spec.

## Design decisions (the spec)

1. **DiT module = Difflet's port, not diffusers' class.** HV1.5 wraps diffusers only because no Difflet port exists. HV1.0's port is the class the AoT path runs, is parity-tested against diffusers on CPU (`tests/numerical/test_hunyuan_video_vs_diffusers.py`), and its single blocks already split the fused `proj_out` (loaded via `CheckpointSlice`, exactly as Flux does in `difflet/backends/neuron/flux/transformer.py:78-85`). Its attention already routes the text padding through `attention(..., bound_min, bound_max)`, which the neuron op serves with attention_cte's key-bounded flash kernel.
2. **No CFG.** HV1.0 is guidance-distilled (`guidance = scale × 1000` into the timestep embedding). `cfg_parallel_enabled` is refused, as `entry.py` already does for Trainium.
3. **Text encoding on rank 0's host, inside the generate launch** (LTX-2's `HostTextEncoder` pattern), using diffusers' `HunyuanVideoPipeline.encode_prompt` (Llama template, crop 95, hidden layer −3, 256 tokens; CLIP pooler at 77 tokens): the same tensors the AoT `llama`/`clip` stages produce, without NxD. No device text encoder (YAGNI).
4. **Decode in a separate one-core vae launch** (HV1.5/LTX-2/Wan layout) with diffusers' `AutoencoderKLHunyuanVideo`, which tiles in space **and time** — so HV1.5's long-clip VAE OOM does not recur.
5. **Topology floor TP2.** The 24 GB checkpoint does not fit one 24 GB core; `hbm.dit_fits` refuses tp1 before torchrun starts. Supported layouts on one trn2.3xlarge: TP2, TP4, TP2+SP, TP4+SP, TP2×CP2 (gather_kv / ulysses / ring). 24 heads ⇒ `tp ∈ {2, 3, 4, 6, …}`.
6. **Ring CP needs one cross-backend change:** the port's ring branch raises on any mask, and the port always builds one. Pass the valid-text count as `text_valid_len` (the neuron op already takes it, added for HV1.5); add the keyword to the `difflet.ops` facade, implement it in the CPU reference, refuse a non-None value on Trainium (whose behaviour is unchanged: it raised before too).
7. **Out of scope:** the planner (`difflet/planner/*`, Trainium-only model of feasibility), serving (`difflet serve` metadata), multi-shape bucketing, batch > 1, device text encoder, I2V (`image_condition_type`).

## Amendments from the reuse survey (2026-10-24) — authoritative over the task text

A survey of every model on the neuron backend (verified claims only) found these. Each is tagged with its owning task. **If the owning task is already committed, the next task to run applies the amendment first, as its own commit.**

- **A1 (Task 2) Share the fused proj_out helpers.** Hoist Flux's single-block slicing into `difflet/backends/neuron/core/checkpoint.py`: `single_block_proj_out_slices(num_single_layers: int, inner_dim: int, mlp_dim: int) -> dict[str, CheckpointSlice]` and `single_block_proj_out_key(name: str) -> str` (`…proj_out_attn.bias` → `…proj_out.bias`). Flux's `checkpoint_slices`/`checkpoint_key` call them (a `refactor(neuron):` commit; `tests/unit/backends/test_neuron_flux_dit.py` stays green); HV1.0 uses them with `mlp_dim = int(inner_dim × mlp_ratio)`. No `global_rank.rank` key on neuron.
- **A2 (Task 1/2) Truncated depth.** Add `checkpoint_num_layers` and `checkpoint_num_single_layers` to `NeuronHunyuanVideoConfig` (default: the checkpoint's 20 / 40), and in the DiT application set `skipped_checkpoint_prefixes` from `config_checks.truncated_block_prefixes` for both stacks (`"transformer_blocks."`, `"single_transformer_blocks."`), as Wan does (`wan/transformer.py:89-92`). Task 8 uses it for fast N-block device parity.
- **A3 (Task 5) TeaCache metric is absolute L2.** HV1.0's calibrations were fit on `vector_norm(m_t − m_{t−1})` (`trainium/hunyuan_video/teacache_probe_model.py:86-87`, `pipeline/teacache.py:190,250`); the shared `NeuronTeaCacheProbe` returns rel-L1. Subclass it (`_HunyuanVideoTeaCacheProbe` in `neuron_application.py`) overriding `teacache_delta` to `torch.linalg.vector_norm((m.float() − prev.float()).reshape(-1))`, keep the first-step zero-prev behaviour, and call `probe.reset()` at the start of every `generate()`. Test: the delta equals the host-side L2 the controller computes for the same two mod inputs. The orchestrator's fused branch is at `pipeline.py:230-255` (not 220-226).
- **A4 (Task 5) Keep the mask int64 at the DiT boundary.** `OnDeviceDiT._forward` casts every tensor to the DiT dtype; override it in `_OnDeviceHunyuanVideoTransformer` so floats go to bf16 and integer tensors keep their dtype (HV1.5 `_positional`, `hunyuan_video15/transformer.py:144`), matching the warm-up's int64 mask (else every request compiles a new block graph). Test: the mask reaching the module is int64.
- **A5 (Task 5) Host encoder warm-up and shape check.** Copy LTX-2/Qwen's discarded first encode (`warmup_args`, `models/ltx_2/neuron_application.py:136-187`) — the bf16 first encode is nondeterministic. Model the DiT shape check on `_check_dit_shapes` (`models/ltx_2/neuron_application.py:354`).
- **A6 (Task 6) VAE dtype.** Default `dtype=torch.float32` (the checkpoint is fp32; Wan and Flux decode in fp32); Task 8 measures bf16 and may switch the default. Post-process with diffusers' `VideoProcessor(vae_scale_factor=8)` where the CLI saves frames.
- **A7 (Task 7) CLI reuse details.** Filter the `--teacache*` options out of `torchrun_unsupported_options` exactly as `cli/orchestrators/wan.py:257` does, or the shared refusal blocks them. Mirror LTX-2's decode flags exactly (its default and its opt-in flag for the device vae launch) instead of inventing `--host-vae` semantics; Task 8 measures both. Use a distinct stage component name for the neuron generate stage and drop the clip/llama keys there. Add a HunyuanVideo fake to `tests/unit/cli/_fake_neuron_pipeline.py`. Pin the tp1 refusal with a header-only safetensors snapshot, as `tests/unit/cli/test_ltx2_neuron_cli.py:133` does.
- **A8 (Task 8) Reuse the parity and benchmark harnesses.** The check scripts import `tests/numerical/neuron_parity.py` (`run_parity`, `check`, `device_evidence`, `_cast`) and `tests/numerical/neuron_benchmark.py` (`run`) instead of re-deriving metrics. Prefer truncated-depth parity (A2; pattern `tests/manual/check_neuron_ltx2_preflight.py`) over the 52 GB full fp32 oracle; keep one full-depth run. Re-target `tests/manual/check_hunyuan_video_mask_bounds.py` at `torch.compile(backend="neuron")` as a regression for the fp32 valid-key count. Opt the generate stage into the NKI trace cache as `cli/orchestrators/hunyuan_video_15.py:95-108` does if the bounded-kernel trace time is material.
- **A9 (Task 8 contingencies, only if a device check fails).** HBM at TP2: the replicated AdaLN modulation linears put TP2 at about 15.2 GiB of weights per rank (Task 1's ÷tp estimate under-counts; the base re-checks exact bytes at load). If 320×512×61 overflows, make `norm1.linear`/`norm1_context.linear`/`norm.linear` column-parallel with a gathered output (HV1.5 `hunyuan_video15/transformer.py:414`), about 3.2 GiB saved per rank. Accuracy: wrap the GELUs in `core/activations.py:35 Fp32GELU` (`use_fp32_gelu` does not recognise `HunyuanVideoGELU`; the single blocks' bare `nn.GELU` needs the same).

## Global Constraints

- Device dtype bf16 only (`require_bf16_on_device`); CPU tests run fp32 on gloo with `DIFFLET_BACKEND=neuron`.
- One static shape per process; default `height=320, width=512, num_frames=61`; `num_frames % 4 == 1`; `height % 16 == 0` and `width % 16 == 0` (VAE ×8, patch 2).
- Text: `text_seq_len = 256`, `encoder_attention_mask` int64 right-padded; pooled projection `[B, 768]`; guidance tensor `[B]` = `guidance_scale × 1000.0`.
- Image tokens `N = ((F−1)/4+1) × (H/16) × (W/16)` (10,240 at the default shape) must divide by `cp` (CP) or `tp` (SP).
- Parity gate on device (repo convention, `tests/manual/check_neuron_hv15_dit.py`): rel-L1 vs fp32 diffusers ≤ 1.15 × the bf16-diffusers control's rel-L1.
- Commit per task; messages in the repo's style (`feat(neuron): HunyuanVideo …`), ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run unit tests with `/home/ubuntu/venvs/native-312/bin/python -m pytest`.

## Review Focus

1. **Very short prompts** (3 valid text tokens of 256): the joint key bound is `N + 3`; the token refiner's symmetric mask has almost all rows padded (column 0 forced True). Expected: finite output matching diffusers. Pinned in Task 2 (`test_dit_matches_diffusers_short_prompt`) and on device in Task 8.
2. **Full-length prompts** (256 valid, no padding): bound equals the full key length. Pinned in Task 2 (`valid_text=256` case).
3. **Shapes CP/SP cannot split** (e.g. 33 frames at 480×848 = 9×30×53 = 14,310 tokens at cp=4): a `ValueError` naming the token count and a divisor before anything loads. Pinned in Task 1.
4. **A request at a shape other than the compiled one**: a clear `ValueError` from the DiT adapter, not a silent recompile. Pinned in Task 5 (`test_dit_adapter_rejects_other_shape`).
5. **Guidance scaling**: the DiT must receive `guidance_scale × 1000`; a missed factor gives plausible-looking but wrong video. Pinned in Task 5 (`test_generate_sends_guidance_times_1000`).

---

### Task 1: `NeuronHunyuanVideoConfig` and the HBM estimate

**Files:**
- Create: `difflet/backends/neuron/hunyuan_video/__init__.py` (docstring only, like `hunyuan_video15/__init__.py`)
- Create: `difflet/backends/neuron/hunyuan_video/config.py`
- Create: `difflet/backends/neuron/hunyuan_video/hbm.py`
- Test: `tests/unit/backends/test_neuron_hv1_config.py`

**Interfaces:**
- Produces: `NeuronHunyuanVideoConfig` (dataclass, modelled on `difflet/backends/neuron/ltx_2/config.py:NeuronLtx2Config`): shape fields `height=320, width=512, num_frames=61, batch_size=1, text_seq_len=256`; every key of the transformer `config.json` (`in_channels … rope_axes_dim`, defaults = the HunyuanVideo snapshot's values); parallel fields `tp_degree=1, cp_degree=1, context_parallel_enabled=False, cp_mode="gather_kv", sp_enabled=False, cfg_parallel_enabled=False`. Properties `latent_frames, latent_height, latent_width` (÷4+1, ÷8, ÷8), `num_image_tokens`, `inner_dim`. `validate() -> None`. `from_pretrained(transformer_path, **overrides) -> NeuronHunyuanVideoConfig`. The config is also what `HunyuanVideoTransformer3DModel(config)` reads (it takes any object with those attributes, `modeling_hunyuan_video.py:1022-1043`).
- Produces: `hbm.dit_fits(model_dir: str, *, tp_degree: int) -> tuple[bool, int, int]` (same contract as `difflet/backends/neuron/ltx_2/hbm.py`, built on `difflet/backends/neuron/hbm.py` `Component`/`check_fits`; reads safetensors headers only).

- [ ] **Step 1: Write the failing tests**

```python
def test_defaults_match_the_snapshot_shape():
    c = NeuronHunyuanVideoConfig()
    assert (c.latent_frames, c.latent_height, c.latent_width) == (16, 40, 64)
    assert c.num_image_tokens == 16 * 20 * 32 == 10240

def test_validate_refuses_what_hunyuan_video_does_not_support():
    with pytest.raises(NotImplementedError, match="guidance-distilled"):
        NeuronHunyuanVideoConfig(cfg_parallel_enabled=True).validate()
    with pytest.raises(ValueError, match="4k\\+1"):
        NeuronHunyuanVideoConfig(num_frames=60).validate()
    with pytest.raises(ValueError, match="multiple of 16"):
        NeuronHunyuanVideoConfig(height=328).validate()
    with pytest.raises(ValueError, match="tp=5"):
        NeuronHunyuanVideoConfig(tp_degree=5).validate()
    with pytest.raises(ValueError, match="mutually exclusive"):
        NeuronHunyuanVideoConfig(tp_degree=2, sp_enabled=True, cp_degree=2,
                                 context_parallel_enabled=True).validate()

def test_validate_names_an_unsplittable_token_count():
    c = NeuronHunyuanVideoConfig(height=480, width=848, num_frames=33, cp_degree=4,
                                 context_parallel_enabled=True)
    with pytest.raises(ValueError, match="14310"):
        c.validate()

def test_ulysses_needs_heads_per_rank_divisible_by_cp():
    NeuronHunyuanVideoConfig(tp_degree=2, cp_degree=2, context_parallel_enabled=True,
                             cp_mode="ulysses").validate()          # 12 heads / 2: ok
    with pytest.raises(ValueError, match="ulysses"):
        NeuronHunyuanVideoConfig(tp_degree=8, cp_degree=4, context_parallel_enabled=True,
                                 cp_mode="ulysses").validate()      # 3 heads / 4

def test_dit_fits_refuses_tp1_and_accepts_tp2(tmp_path, monkeypatch):
    # a fake snapshot whose safetensors header declares 12.8e9 bf16 params
    ...
    assert hbm.dit_fits(str(snapshot), tp_degree=1)[0] is False
    assert hbm.dit_fits(str(snapshot), tp_degree=2)[0] is True
```

(Build the fake snapshot the way `tests/unit/backends/test_neuron_hbm.py` does.)

- [ ] **Step 2: Run to verify they fail** — `pytest tests/unit/backends/test_neuron_hv1_config.py -v` → ImportError.
- [ ] **Step 3: Implement** `config.py` with the checks from `difflet/backends/neuron/core/config_checks.py` (`check_heads_divide_tp`, `check_cp_mode`, `check_cp_split`, `check_ulysses_heads`, `exclusive_modes`, `check_even_split` for SP); and `hbm.py` (DiT component = whole transformer checkpoint ÷ tp, plus the activation margin `ltx_2/hbm.py` uses).
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(neuron): HunyuanVideo config and HBM estimate (tp1 refused: 24 GB DiT, 24 GB core)`.

---

### Task 2: The DiT application (TP, eager and compile)

**Files:**
- Create: `difflet/backends/neuron/hunyuan_video/transformer.py`
- Create: `tests/unit/backends/_neuron_hv1_toy.py` (tiny snapshot writer + diffusers reference)
- Modify: `tests/unit/backends/_neuron_workers.py` (add `hv1_dit_worker`)
- Test: `tests/unit/backends/test_neuron_hv1_dit.py`

**Interfaces:**
- Consumes: `NeuronHunyuanVideoConfig` (Task 1).
- Produces: `checkpoint_key(name: str) -> str` (module → checkpoint: `single_transformer_blocks.{i}.proj_out_attn.bias` → `single_transformer_blocks.{i}.proj_out.bias`, identity otherwise).
- Produces: `NeuronHunyuanVideoTransformerApplication(TorchNeuronApplicationBase)`:
  - `__init__(*, model_path, config: NeuronHunyuanVideoConfig | None = None, parallel=None, dtype=torch.bfloat16, exec_mode=None, device="neuron", **overrides)` — builds the config with `from_pretrained` when None, then `check_tp_matches` / `resolve_cp_degree` / `check_flag_matches(..., "sp_enabled")` / `validate()` / `require_bf16_on_device`, as `difflet/backends/neuron/wan/transformer.py:80-92` does.
  - `block_attrs = ("transformer_blocks", "single_transformer_blocks")`.
  - `build_module()` → a module-private subclass `_HunyuanVideoTensorSignature(HunyuanVideoTransformer3DModel)` whose `forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance) -> Tensor` returns `super().forward(..., return_dict=False)[0]` (HV1.5's `_tensor_signature_model` idea: the base warms up and compiles with positional tensors).
  - `checkpoint_slices() -> dict[str, CheckpointSlice]`: per single block, `proj_out_attn.weight = CheckpointSlice("single_transformer_blocks.{i}.proj_out.weight", 1, 0, inner_dim)` and `proj_out_mlp.weight = CheckpointSlice(same, 1, inner_dim, mlp_ratio × inner_dim)`.
  - `get_example_inputs()` → the six tensors at the static shape: latents `[B,16,F',H/8,W/8]`, timestep `[B]`, text `[B,256,4096]`, mask int64 `[B,256]` with the first 64 ones, pooled `[B,768]`, guidance `[B]` (6000.0).
  - `teacache_mod_input(hidden_states, timestep, pooled_projections, guidance) -> Tensor` — `module.teacache_mod_input(x, t, None, None, pooled, guidance)` on this rank's device (the port's method drops the text arguments; the modulation linears are diffusers' replicated `nn.Linear`, so the result is whole on every rank).

- [ ] **Step 1: Write the tiny toy.** `_neuron_hv1_toy.write_tiny_hv1(root) -> str` saves a diffusers `HunyuanVideoTransformer3DModel(in_channels=4, out_channels=4, num_attention_heads=4, attention_head_dim=16, num_layers=1, num_single_layers=2, num_refiner_layers=1, patch_size=2, patch_size_t=1, text_embed_dim=32, pooled_projection_dim=16, rope_axes_dim=(4, 6, 6), guidance_embeds=True)` with a fixed seed; `tiny_inputs(valid_text: int)` builds a `[1,4,5,8,8]` latent (5×4×4 = 80 image tokens after the 2×2 patch: divisible by cp 2 and 4), 16-token text with `valid_text` valid, pooled, timestep 500, guidance 6000; `diffusers_reference(path, inputs)` runs diffusers' forward in fp32.
- [ ] **Step 2: Write the failing tests**

```python
def test_checkpoint_mapping_and_slices(): ...            # as test_neuron_flux_dit.py:test_checkpoint_mapping
@pytest.mark.parametrize("valid_text", [3, 9, 16])
def test_dit_matches_diffusers_tp1(tiny, valid_text):    # eager, cpu, fp32
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-4)
def test_dit_matches_diffusers_short_prompt(tiny): ...   # valid_text=1: finite and allclose
@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_dit_matches_diffusers_tp2_gloo(tiny, exec_mode):
    outs = run_ranks(hv1_dit_worker, 2, tiny, {"tp_degree": 2}, exec_mode)
    assert all(torch.allclose(o, ref, atol=1e-5, rtol=1e-4) for o in outs)
def test_teacache_mod_input_matches_the_port(tiny): ...  # equals HunyuanVideoTransformer3DModel.teacache_mod_input on CPU
```

- [ ] **Step 3: Run to verify they fail.**
- [ ] **Step 4: Implement** `transformer.py` (module docstring in the house style: what it wraps, the lifecycle, why the port and not diffusers, numerics workarounds relied on — fp32 LayerNorm and the bool bitwise guard installed by `prepare_runtime`).
- [ ] **Step 5: Run to verify they pass** (compile mode on CPU uses the `aot_eager` backend).
- [ ] **Step 6: Commit** — `feat(neuron): HunyuanVideo DiT application (Difflet's port, TP, per-block compile, sliced proj_out)`.

---

### Task 3: Context parallelism (gather_kv, ulysses) and sequence parallelism

**Files:**
- Modify: `tests/unit/backends/_neuron_workers.py` (reuse `hv1_dit_worker` with the parallel dict)
- Test: `tests/unit/backends/test_neuron_hv1_dit.py` (new cases)
- Modify only if a test fails: `difflet/backends/neuron/hunyuan_video/transformer.py`, `difflet/models/hunyuan_video/modeling_hunyuan_video.py`

**Interfaces:**
- Consumes: Task 2's application with `parallel=DiffletParallelConfig(tp_degree=…, cp_degree=…, cp_mode=…, sp_enabled=…)`.
- Produces: nothing new — this task proves the port's existing CP/SP code paths on the neuron collectives (`scatter_to_process_group_spmd`, `SPMDRank` as a Python-int rank, `seq_to_heads`, `reduce_scatter_to_sequence_parallel_region`, `_sp_unbias` with the neuron `RowParallelLinear`'s full-bias partials).

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("parallel,world", [
    ({"tp_degree": 1, "cp_degree": 2, "cp_mode": "gather_kv"}, 2),
    ({"tp_degree": 1, "cp_degree": 2, "cp_mode": "ulysses"}, 2),
    ({"tp_degree": 2, "cp_degree": 2, "cp_mode": "gather_kv"}, 4),
    ({"tp_degree": 2, "cp_degree": 2, "cp_mode": "ulysses"}, 4),
    ({"tp_degree": 2, "sp_enabled": True}, 2),
])
@pytest.mark.parametrize("valid_text", [3, 16])
def test_dit_parallel_matches_diffusers(tiny, parallel, world, valid_text):
    outs = run_ranks(hv1_dit_worker, world, tiny, parallel, "eager", valid_text)
    assert all(torch.allclose(o, ref, atol=1e-5, rtol=1e-4) for o in outs)
```

Plus one compile-mode case for `tp2 cp2 ulysses` (the collectives must stay functional inside the block graphs).

- [ ] **Step 2: Run them.** Any that pass need no code. For each failure, use superpowers:systematic-debugging; fix in the smallest owning file; one commit per bug (`fix(neuron): …` or `fix(hunyuan_video): …`, the message naming the failing layout).
- [ ] **Step 3: Run the whole file to verify it passes.**
- [ ] **Step 4: Commit** — `test(neuron): HunyuanVideo DiT under CP (gather_kv, ulysses) and SP against diffusers`.

---

### Task 4: Ring context parallelism with the text mask

**Files:**
- Modify: `difflet/ops/attention.py:62` (facade gains `text_valid_len=None`)
- Modify: `difflet/backends/cpu/ops_impl/attention.py:73` (reference: mask text keys `>= text_valid_len[b]` in the joint softmax)
- Modify: `difflet/backends/trainium/ops_impl/attention.py:260` (accept the keyword; `NotImplementedError` if not None)
- Modify: `difflet/backends/tpu/ops_impl/attention.py:298` (signature unchanged: it already raises for everything)
- Modify: `difflet/models/hunyuan_video/modeling_hunyuan_video.py:468-487` (ring branch: replace the `attention_mask is not None` refusal with `text_valid_len = attention_mask.reshape(B, -1)[:, -S_txt:].sum(-1, dtype=torch.float32).to(torch.int32)` — the joint mask's text columns (it is built over `[latent_full ‖ text]`, so the last `S_txt = context_k.shape[1]` columns are the text), with the float-sum idiom of the ulysses branch (Trainium LNC2 integer-reduction bug) — and pass it)
- Test: `tests/unit/ops/test_joint_ring_text_valid_len.py`, `tests/unit/backends/test_neuron_hv1_dit.py` (ring cases)

**Interfaces:**
- Produces: `difflet.ops.joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale, causal=False, text_valid_len=None)`; `text_valid_len` is int32 `[B]` counting valid text keys (text right-padded).

- [ ] **Step 1: Write the failing tests** — CPU reference with `text_valid_len=[5]` equals dense SDPA over `[image ‖ text[:5]]`; Trainium impl raises `NotImplementedError` for a non-None value (import-guarded like the other trainium op tests); and in `test_neuron_hv1_dit.py` the ring layouts `{"cp_degree": 2, "cp_mode": "ring"}` (world 2) and `{"tp_degree": 2, "cp_degree": 2, "cp_mode": "ring"}` (world 4) at `valid_text ∈ {3, 16}` match diffusers.
- [ ] **Step 2: Run to verify they fail** (the modeling raises "ring cp_mode does not support attention_mask").
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run these tests plus `tests/unit/models/hunyuan_video/` to verify they pass and nothing on the AoT side regressed.**
- [ ] **Step 5: Commit** — `feat(hunyuan_video): ring context parallelism masks the padded text by count (neuron); difflet.ops.joint_ring_attention gains text_valid_len`.

---

### Task 5: The composed application, the host text encoder, TeaCache and the registry

**Files:**
- Create: `difflet/models/hunyuan_video/neuron_application.py`
- Modify: `difflet/models/hunyuan_video/entry.py:19-22` (a `backend == "neuron"` branch returning `TorchNeuronHunyuanVideoApplication`, after the existing CFG refusal; as `difflet/models/ltx_2/entry.py:27-38`)
- Modify: `difflet/registry.py:324` (`backends=("trainium", "neuron")`)
- Modify: `tests/unit/models/hunyuan_video/test_hunyuan_entry.py:49,56` (neuron is now accepted; another backend still rejected)
- Modify: `tests/unit/backends/_neuron_hv1_toy.py` (`write_tiny_hv1_pipeline(root)`: tiny transformer + tiny `AutoencoderKLHunyuanVideo` + scheduler + tiny random-weight Llama and CLIP text models with the snapshot's real tokenizers copied from the local HF cache — the Llama template's crop of 95 tokens needs the real tokenizer; skip when the snapshot is absent — as `_neuron_ltx2_toy.py` builds its tiny pipeline)
- Test: `tests/unit/backends/test_neuron_hv1_app.py`

**Interfaces:**
- Consumes: Tasks 1–2; `HunyuanVideoOrchestrator`, `HunyuanVideoDiTInputBundle`, `_bundle_from_tensors` (`difflet/models/hunyuan_video/pipeline.py`); `TorchNeuronModelApplication`, `HostTextEncoder`, `OnDeviceDiT`, `broadcast_from_rank0` (`core/model_application.py`); `NeuronTeaCacheProbe` (`core/teacache.py`).
- Produces: `TorchNeuronHunyuanVideoApplication(TorchNeuronModelApplication)`:
  - `MODEL_LABEL = "HunyuanVideo"`, `DIT_BUNDLE = HunyuanVideoDiTInputBundle`, `TEACACHE_KWARGS = ()` (TeaCache is implemented, as Wan's).
  - `KWARG_DEFAULTS`: `device="neuron", exec_mode=None, text_seq_len=256, shapes=None, batch_size=1, text_encoder_device="host", encoder_threads=<Wan's default>, enable_vae_decoder=False, teacache_speedup=None, teacache_calibration_path=None, teacache_cadence=None, teacache_online_delta_alpha=None` plus whatever the HV CLI stage passes the Trainium application today (read `_stage_generate` in `difflet/cli/orchestrators/hunyuan_video.py:247-352` and accept exactly those keys).
  - `_reject_model_options`: `text_encoder_device != "host"` → `NotImplementedError`; `shapes` with more than one shape → `NotImplementedError`.
  - `generate(*, prompt, seed=None, num_inference_steps=4, guidance_scale=6.0, output_type="latent", **kwargs)` → host-encode on rank 0 and broadcast, initial latents drawn on rank 0 (`_draw_on_rank0`), then `self.pipeline(latents=…, encoder_hidden_states=…, encoder_attention_mask=…, pooled_projections=…, guidance_scale=…, num_inference_steps=…, output_type=…)`.
- Produces: `_HostHunyuanVideoTextEncoder(HostTextEncoder)`: diffusers `HunyuanVideoPipeline.from_pretrained(model_path, transformer=None, vae=None, torch_dtype=bf16)` on rank 0; `__call__(prompt) -> (embeds [1,256,4096] bf16, mask [1,256] int64, pooled [1,768] bf16)` via `pipe.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)`; the mask is cast to int64 before the broadcast.
- Produces: `_OnDeviceHunyuanVideoTransformer(OnDeviceDiT)`: `__call__(bundle)` checks every tensor's shape against the config first (`ValueError` naming the input, its shape and the compiled shape), then `_forward(bundle.as_model_inputs())`; attributes `teacache_probe_fused` (True when a probe is set) and `teacache_delta(bundle)` → `NeuronTeaCacheProbe(app.teacache_mod_input).teacache_delta(hidden, timestep, pooled, guidance)` — the orchestrator's existing fused branch (`pipeline.py:220-226`) then needs no change.

- [ ] **Step 1: Write the failing tests**

```python
def test_registry_builds_the_neuron_application(tiny_pipeline): ...   # create_application(backend="neuron", device="cpu") is a TorchNeuronHunyuanVideoApplication
def test_generate_matches_diffusers_pipeline(tiny_pipeline):           # latents, 3 steps, seed 42, vs HunyuanVideoPipeline(output_type="latent"), fp32 cpu
def test_generate_tp2_gloo_matches_tp1(tiny_pipeline): ...
def test_generate_sends_guidance_times_1000(tiny_pipeline):            # record bundles at the adapter: guidance == 6000.0
def test_dit_adapter_rejects_other_shape(tiny_pipeline):               # ValueError mentions 'hidden_states' and both shapes
def test_text_encoder_matches_diffusers_encode_prompt(tiny_pipeline): ...
def test_refuses_cfg_parallel_and_device_text_encoder(tiny_pipeline): ...
def test_teacache_cadence_skips_and_probe_mode_runs(tiny_pipeline): ...   # cadence 2 halves DiT calls; adaptive with a calibration uses teacache_delta
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** (module docstring in the house style of `difflet/models/wan/neuron_application.py`).
- [ ] **Step 4: Run these tests plus `tests/unit/registry tests/unit/models/hunyuan_video tests/unit/backends/test_neuron_model_application.py`.**
- [ ] **Step 5: Commit** — `feat(neuron): HunyuanVideo end-to-end application (host Llama/CLIP encode, orchestrator, TeaCache) and its registry entry`.

---

### Task 6: The VAE decoder application

**Files:**
- Create: `difflet/backends/neuron/hunyuan_video/vae.py`
- Test: `tests/unit/backends/test_neuron_hv1_vae.py`

**Interfaces:**
- Produces: `NeuronHunyuanVideoVaeDecoderApplication(*, model_path, height=320, width=512, num_frames=61, tile_size=256, tile_frames=16, dtype=torch.bfloat16, device="neuron")` with `load(skip_warmup=False)`, `decode(latents) -> [B,3,F,H,W]` in [−1, 1], `latent_shape`, `scaling_factor`, `phase_seconds` — HV1.5's `vae.py` with `AutoencoderKLHunyuanVideo`; tiling sets diffusers' `tile_sample_min_height/width = tile_size`, `tile_sample_stride_height/width = tile_size × 3 // 4`, `tile_sample_min_num_frames = tile_frames`, `tile_sample_stride_num_frames = tile_frames × 3 // 4` (diffusers' defaults are 256/192/16/12); the encoder is dropped after loading.

- [ ] **Step 1: Write the failing tests** — on the tiny VAE from Task 5's toy: our decode equals diffusers' tiled decode with the same tile settings (allclose fp32); `latent_shape` at 320×512×61 is `(1, 16, 16, 40, 64)`; encoder is None after load.
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(neuron): HunyuanVideo VAE decoder application (diffusers' autoencoder, eager, tiled in space and time)`.

---

### Task 7: The CLI on the neuron backend

**Files:**
- Modify: `difflet/cli/orchestrators/hunyuan_video.py` (`compile`/`generate` :81/:93 branch on `torchrun_backend()`; new `_compile_torchrun`, `_generate_torchrun`, `_stage_generate_neuron`, `_stage_vae_neuron`; `_run_stage_internal` :113 dispatches `generate`/`vae` on neuron; `_stage_cache_inputs` :354 wraps with `torchrun_stage_keys`)
- Test: `tests/unit/cli/test_orchestrator_hunyuan_neuron.py`; modify `tests/unit/cli/test_neuron_stage_cache_keys.py` (add HunyuanVideo)

**Interfaces:**
- Consumes: `run_generate_stage`, `run_vae_stage`, `refuse_dit_overflow`, `reject_unsupported_torchrun_options`, `torchrun_stage_keys`, `root_compile_cache_at_cache_dir`, `torchrun_parallel` (`difflet/cli/orchestrators/torchrun.py`); `hbm.dit_fits` (Task 1); the application (Task 5); the VAE (Task 6).
- Produces: on neuron, `difflet generate --model hunyuan_video` runs two launches — `generate` on `tp × cp` cores (writes `latents.pt`), then `vae` on one core (writes the mp4) — following `difflet/cli/orchestrators/ltx_2.py:450-647` stage for stage. The AoT path (`clip`, `llama`, `generate`) is untouched. `--host-vae` decodes with diffusers on the host instead of the vae launch, if the LTX-2 orchestrator offers it (mirror its flag set; add nothing it lacks).

- [ ] **Step 1: Write the failing tests** (monkeypatch `runner.run_stage` and `torchrun_backend`, as `tests/unit/cli/test_orchestrator_ltx2.py` does): stage chain is `["generate", "vae"]` with core counts `[tp*cp, 1]`; `--sp`, `--cp-mode`, `--exec-mode` and the TeaCache flags are forwarded to the generate stage; tp1 exits 2 through `refuse_dit_overflow` before any stage runs; `--cfg-parallel` exits non-zero with the guidance-distilled message; the AoT path's stage chain is unchanged (`["clip", "llama", "generate"]`); the cache keys of the generate stage include `backend` and `exec_mode`.
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Run `tests/unit/cli` to verify they pass.**
- [ ] **Step 5: Commit** — `feat(cli): HunyuanVideo on the neuron backend: torchrun generate stage (host encode), one-core vae stage`.

---

### Task 8: On-device validation and the evidence report

**Files:**
- Create: `tests/manual/check_neuron_hv1_dit.py` (flags as `check_neuron_hv15_dit.py`: `--model-dir`, `--make-oracle PATH | --oracle PATH|none`, `--iters`, `--exec-mode`, `--tp-degree`, `--cp-mode`, `--sp`, `--frames`, `--valid-text`)
- Create: `tests/manual/check_neuron_hv1_e2e.py` (`--make-ref` / `--check`, `--tp-degree`; diffusers `HunyuanVideoPipeline`, 3 steps, seed 42, latents compared)
- Create: `benchmark/trn2/hunyuan_video_neuron.md`
- Modify: `.claude/skills/difflet-device-verify/references/feature-catalog.md` (HunyuanVideo's neuron row)

**Interfaces:**
- Consumes: everything above. The oracle is fp32 diffusers on the host CPU at 320×512, 5 frames, 256-token text with `--valid-text` valid (default 20): ~52 GB host RAM.

- [ ] **Step 1: Make the oracles** (host, fp32): `check_neuron_hv1_dit.py --make-oracle $JOB/tmp/hv1_dit_oracle.pt --valid-text 20` and again with `--valid-text 3`; `check_neuron_hv1_e2e.py --make-ref`.
- [ ] **Step 2: DiT parity on device**, each passing the 1.15× gate, in this order (stop and debug at the first failure, one commit per bug): TP2 eager; TP2 compile; TP4 compile; TP2+SP; TP2×CP2 gather_kv; TP2×CP2 ulysses; TP2×CP2 ring; TP2 at `--valid-text 3`.
- [ ] **Step 3: End-to-end parity** — `check_neuron_hv1_e2e.py --check --tp-degree 2` passes the gate.
- [ ] **Step 4: CLI runs** — `difflet generate --model hunyuan_video --backend neuron` at TP2, TP4 and TP2×CP2 ulysses, 320×512×61, 50 steps: a playable mp4 each; record per-step latency, compile and load seconds, and peak HBM in `benchmark/trn2/hunyuan_video_neuron.md` beside the AoT numbers from `benchmark/trn2/` for the same shape; then 720×1280×129 at the fastest layout that fits.
- [ ] **Step 5: Run the full unit suite** — `pytest tests/unit -q -x -n 8`.
- [ ] **Step 6: Commit** — `docs(benchmark): HunyuanVideo on the neuron backend: parity per layout and native vs AoT on trn2`.

Use the `difflet-device-verify` skill for Steps 2–4 (its evidence-doc format, disk and supervision rules, one-bug-one-commit protocol).

---

## Status and remaining work (2026-10-25)

**Done.** Tasks 1–8 are committed on `feat/hunyuan-neuron` (pushed; HEAD `02e5cf6`), including amendments A1–A9, the audit's doc corrections, and three fixes found on the device: decode in the one-core vae launch by default (`d6ffb9a`), the NKI trace cache for HunyuanVideo launches (`f4d9ac1`), and the bounded-attention head split above attention_cte's tested size (`15b503f`, `ce9d212`). Evidence: `docs/verification/hunyuan-video-neuron-evidence.md`, `benchmark/trn2/hunyuan_video_neuron.md`, `benchmark/trn2/native_vs_aot_tp4_2026-10-25.md`; raw logs under `/home/ubuntu/bench-2026-10-25/` (the parity logs under the job tmp dir are not durable).

Headline at TP4, 320×512×61, 50 steps: DiT step 808 ms native vs 814 ms AoT; peak HBM 13.0 vs 15.5 GiB per core; warm e2e about 222 s native vs 133 s AoT (the gap is the VAE). **Update 2026-10-26 (F1/F2 done):** native warm e2e 126.7 s (n=3, TP4 full method at `eadddec`) vs AoT 132.9 s; decode 16.0 s; see F1 below. Qwen-Image TP4 native is 14% faster per step than AoT (21% with `--sp`); Wan2.1 TP4 native is 11% slower than two-core AoT.

### Decisions for the plan owner

- [ ] **D1 Full-depth parity gate.** Proposed: at full depth, gate on the mean ratio over at least 5 timesteps (≤ 1.15×) with a 1.25× cap on any single sample; keep the single-sample 1.15× gate for truncated depth and end to end. Under it the compiled DiT passes (mean 0.979×, max 1.190×). The t=900 / 20-valid-token excess (1.19×) is neuronx-cc's default `--enable-mixed-precision-accumulation` keeping the dual block's mid-block residual `h1` unrounded (evidence doc, "Compiler-level cause"); fixes were measured and not adopted (`--disable-mixed-precision-accumulation`: +39–61% at TP4 5×320×512; explicit `h1` rounding: +13.7% at TP2). Until decided, full-depth TP2/TP4 stay FAIL in the evidence doc.
- [ ] **D2 Pull request.** Open a draft PR from `feat/hunyuan-neuron`, presumably into `feat/ltx2-neuron-rebased` (its base), or hold.

### Follow-up work (not started; ordered by expected value)

Each item: list a concrete plan and get approval before running anything on the device.

- [x] **F1 Close the warm e2e gap to AoT (222 s vs 133 s).** Nearly all of it is the decode: AoT's compiled bf16 VAE decodes in 17 s; native's eager fp32 vae launch takes about 118 s (launch, load, 74 s decode, 22 s frame write). Options: compile the decoder per block, use bf16 once F2 is fixed, speed up the frame write. Estimate: half a day to a day. **F1a done 2026-10-26 (`1c10dfc`, opt-in, CLI unchanged):** `exec_mode="compile"` with fixed-size tiles and a valid-extent tile decoder compiled region by region; bf16 decodes in 16.0 s (22.3 s first in a process), 56.07 dB, cold compile 135 s / 26 NEFFs; fp32 49.7 s (61.4 s first), 139.3 dB (`benchmark/trn2/hunyuan_video_neuron.md`, "F1a: a compiled decoder"). The diffusers-tiling control did not complete (neuronx-cc out of host memory). **F1b done 2026-10-26 (`55ecd5d`, `55761f5`, `ecd1b2d`, `74fc979`):** the 21.9 s "frame write" was the eager decode finishing on the device (the write itself is under 1 s; now one uint8 pass, same mp4 bytes); option A, the decode on rank 0's core in the generate launch with the compiled bf16 decoder, fits at TP4 (rank 0: 14.74 / 17.95 GiB), TP2 (19.24 / 23.50 GiB) and TP2xCP2 (19.24 / 22.85 GiB) and passes the default-change rule, so it is the CLI default (`--device-vae`, `--host-vae`, `--vae-dtype fp32 --vae-exec-mode eager` keep the earlier paths). Warm e2e at TP4 117.8-130.1 s (AoT 132.9 s), TP2 157.3-164.3 s, TP2xCP2 127.6-138.9 s; decode 16.0 s (`benchmark/trn2/hunyuan_video_neuron.md`, "F1b"). **Step 4 done 2026-10-26 (`eadddec`, no flag; `benchmark/trn2/hunyuan_video_neuron.md`, "TP4: native vs AoT" and "VAE: compiled decoder and in-process decode (F1/F2)"):** TP4 full method: warm e2e **126.7 s** (129.2, 130.6, 120.3; was 239.1 s; AoT 132.9 s, *reused*), cold e2e 457.8 s (was 570.4; AoT 617.3), compile cold 274.5 s (was 2877.2), step 808.2 ms, decode 16.0 s, rank 0's core 14.74 / 17.95 GiB, same latents as before. Targets: warm e2e about 150 s or less at TP4, **met**; decode 25 s or less, **met**. TP2 160.0 s (above the 150 s figure; denoise 73 s), TP2xCP2 gather_kv 126.0 s. Decoded frames vs the fp32 host decode of the same latents: 56.07-56.09 dB at TP4, TP2 and TP2xCP2 (gate 50 dB). The earlier default (`--device-vae --vae-dtype fp32 --vae-exec-mode eager`) still runs: 213.1 s, the same mp4 as before.
- [x] **F2 Eager bf16 VAE precision on the device.** 20.8 dB against fp32 with visible banding, while bf16 on the host is 55.9 dB (`benchmark/trn2/hunyuan_video_neuron.md`, VAE dtype). **Done 2026-10-26:** the device's bf16 GroupNorm (its error grows with the group size); `2e04372` computes it in fp32. Device bf16 full decode now 55.9 dB (rel-L1 3.8e-3, as on the host), fp32 bit-identical to before (`benchmark/trn2/hunyuan_video_neuron.md`, "F2: bf16 VAE precision on the device").
- [ ] **F3 The compile-mode vae launch lingers 927 s before exiting** (generate-mode launches exit in 3–4 s); it adds about 15 min to a cold compile. Estimate: 1–2 h. *Since `74fc979` the default decodes in the generate launch with the compiled decoder, so only the opt-in eager `--device-vae --vae-exec-mode eager` path still pays this.*
- [ ] **F4 720×1280×129.** Blocked: after the head split the attention kernel compiles, but neuronx-cc compiling the block graph is OOM-killed at 109.8 GiB of the host's 124.8 GiB. Untried: `-O1`, attention in its own graph, ring CP, fewer frames, a host with more RAM. Estimate: half a day, no guarantee.
- [ ] **F5 Adaptive TeaCache for HunyuanVideo on neuron.** No 320×512×61 calibration exists on the host; only the probe-free cadence mode was measured (cadence 2: 20/50 steps skipped, denoise 40.4 → 24.3 s, cos 0.99984). The probe uses the absolute-L2 metric (A3). Estimate: 2–3 h.
- [ ] **F6 Profile the native vs AoT DiT step gaps that are not explained yet.** What is known (`neuron_feature_matrix.md:54`, `neuron_final.md:246`, `native_vs_aot_tp4_2026-10-25.md` open issue 4): Qwen-Image native is 14% faster mainly because NxD's `RowParallelLinear` all-reduces in fp32 by default (forcing bf16 took AoT 417 → 360 ms per forward); LTX-2 native is 16% faster, partly because stock AoT ran attention on one physical core (two-core attention recovered 39 of the 111 ms per-forward gap), the rest attributed to the fp32 all-reduce, the older compiler (2.26) and unexplained per-block work. Open questions:
  - [x] **F6a Wan2.1:** native trails two-core AoT by 11.3% per step (1063.6 vs 955.8 ms). Wan's AoT modeling already all-reduces in bf16, and two-core attention took AoT 1151 → 957 ms. Whether native's per-block `torch.compile` attention uses both physical cores of the logical core is not profiled; that is the main suspect. Profile one native Wan block on the device (neuron-profile, or per-op timings) against the two-core AoT block. **User decision 2026-10-26: making native attention use both physical cores (two-core, as AoT's `attention_cte[2]`) may change code**; for the four-model native vs AoT comparison, both backends must run two-core attention at TP4. If native is not two-core today, implement it (with CPU tests and device parity) before that comparison's native runs. **Audit 2026-10-26 (read-only): native attention is already two-core, no code change needed:** the native NEFFs of HunyuanVideo, Qwen-Image and Wan2.1 split the bounded attention into `sg00 attention_cte_lnc0` and `sg01 attention_cte_lnc1` subgraphs, the same structure as AoT's `attention_cte[2]` (e.g. `/home/ubuntu/bench-2026-10-25/roots/wan_native_tp4/neuron/neff/17/78/17788144....neff`); `ops_impl/attention.py:168` launches with `grid=(get_logical_neuron_cores(),)` and every native compile carries `--lnc 2`. So two-core attention does not explain F6a's 11% gap; the cause is still open. LTX-2 native goes through the same path; confirm by scanning its NEFFs after its first compile. **Profiled 2026-10-28 (`benchmark/trn2/four_model_native_vs_aot_2026-10-26.md`, section "Wan2.1 native vs AoT DiT step: cause"; no code change):**
    - **The gap today** is 1064.9 vs 957.9 ms (+11.2%).
    - **It is the block's own device time.** The native block graph runs 12.31 ms against about 10.95 ms per block inside AoT's whole-model graph (neuron-explorer). Over 40 blocks that is 54 ms per forward, the whole 53.5 ms gap.
    - **Ruled out:** the compiler version (AoT's HLO recompiled with 2.28 runs 442.7 vs 438.1 ms), per-block graph launches, host dispatch, CFG batching and the all-reduce dtype.
    - **One compiler flag is worth 9 ms per forward.** Native compiles without AoT's `--tensorizer-options=--enable-ccop-compute-overlap`.
    - **The rest (about 40 ms per forward) has no measured cause.** *Inference:* the torch-native lowering. Measured: LayerNorm, GELU and RoPE are lowered differently from torch-xla, and the native PE idles 4.6 vs 2.8 ms per block, but single-op ablations recover nothing.
    - **Both measured knobs together** (the flag plus a scatter-free RoPE) give 1038.1 ms per step (−2.5%), at 22.06 dB / 0.876 (−0.35 dB, one seed). They are not committed: both change every native model's graphs.
    - **Open:** the user's decision on proposals 1–3 in that section. The remaining ~40 ms per forward needs torch-neuronx or compiler work.
  - **F6b HunyuanVideo 1.0:** native and AoT are level (808.3 vs 813.9 ms), both with two-core attention. Why AoT's fp32 all-reduce does not show here as it does for Qwen-Image is not profiled: check whether HunyuanVideo's AoT path all-reduces in fp32, and what a bf16 all-reduce would give AoT, and whether native has headroom the same way.
  Estimate: half a day of device time for both.
- [ ] **F7 Six Flux neuron unit tests fail only in full-suite order** (they pass alone; two setup errors likewise). Unrelated to this branch's changes. Estimate: 1 h.
- [x] ~~**F8 Units in older trn2 docs.**~~ *Dropped by the user 2026-10-26: not needed.* `neuron_feature_matrix.md`, `neuron_final.md`, `ltx_2_neuron.md` still say "124 GB RAM" for this host (124.8 GiB). Cosmetic, 10 min.
- [x] ~~**F9 Update the weight-loading explainer page**~~ *Dropped by the user 2026-10-26: not needed.* (https://claude.ai/artifact/JcjwomFzRoC5ronErdoEEe, source `docs/reports/2026-10-26-weight-loading-explainer.html`) with the post-F1/F2 HunyuanVideo numbers: TP4 warm e2e 126.7 s vs AoT 132.9 s (native now faster), cold e2e 457.8 vs 617.3 s, decode 16.0 s in process; the page still says native's warm requests are slower because of the decoder. Estimate: 30 min, no device.
- [x] ~~**F10 Add Flux, HunyuanVideo 1.5 and Wan 2.2 to the native vs AoT weight-loading comparison.**~~ *Dropped by the user 2026-10-26: not needed.* Needs a scouting pass (support on both backends, snapshots: HunyuanVideo 1.5 is not on this host, Wan 2.2's two 14B experts and HBM, comparable layouts: HunyuanVideo 1.5 native defaults to TP2 x CFG2), then device measurements with the `native_vs_aot_tp4_2026-10-25.md` method, approved before running.

- [ ] **F11 Make HunyuanVideo 1.0's AdaLN modulation layers column-parallel.** *Why:* the modulation linears (`norm1.linear`, `norm1_context.linear` in the 20 dual blocks, `norm.linear` in the 40 single blocks) are diffusers' `AdaLayerNormZero` / `AdaLayerNormZeroSingle` with a plain `nn.Linear` (`modeling_hunyuan_video.py:581-582, 684`), so neither backend splits them: every core holds all 6.80 GB of them, and NxD's `shard_checkpoint` writes them whole into every AoT rank file (`neuronx_distributed/trace/functions.py:871-890`, `trace.py:752-753, 796-822`), which is why the four AoT DiT files hold 46.54 GB for a 25.64 GB checkpoint. Making them `ColumnParallelLinear(gather_output=True)` (the modulation input is a per-sample `[B, 3072]` vector, so the all-gather is tiny) is what HunyuanVideo 1.0's own token refiner (`modeling_hunyuan_video.py:148-152`) and HunyuanVideo 1.5's native port (`hunyuan_video15/transformer.py:414-415`) already do. Expected (computed, not measured): about 5.1 GB less HBM per core on both backends (headroom for larger shapes, TP2 and the in-process decoder: R2 found only 0.5 GiB spare at TP2), AoT rank files about 6.5 GB instead of 11.6 GB (about 20 GB less on disk, about 160 s less cold read at ~130 MB/s). Gates: CPU gloo parity unchanged against diffusers at TP2/TP4/SP/CP; device parity and step time unchanged; recheck the AoT TeaCache probe, which loads `transformer_blocks.0.norm1.*` by name (`trainium/hunyuan_video/teacache_probe.py:36`). Estimate: half a day plus device checks.
- [x] **F12 Check the LTX-2 snapshot index issue.** **Done 2026-10-26 (`05614a2`, review: no defects):** `build_weight_map(model_dir, keys=None)` picks the complete index holding the most requested keys; on the real snapshot all 626 Gemma keys resolve from `model.safetensors.index.json`. *Why:* the local LTX-2 snapshot was re-downloaded on 2026-10-24 with both text-encoder indexes complete, and `build_weight_map` picks the first complete index in name order (`tpu/core/checkpoint.py:50-56`). That is now `diffusion_pytorch_model.safetensors.index.json`, which holds none of Gemma's `language_model.model.*` keys, and both the native loader and the weight store go through it (`neuron/core/checkpoint.py:206-209`, `weight_store.py:182-184`). So a native device-Gemma load (the native default) may fail to find its weights; LTX-2 has not run on this snapshot yet, and any new LTX-2 measurement depends on it. First a CPU check of which index is chosen and whether the Gemma keys resolve; if it fails, make the index choice prefer the index that matches the module's keys, with a unit test. Estimate: 1 h (CPU).
- [x] **F13 Run a bf16 umT5 accuracy check for Wan2.1 native.** **Done 2026-10-27/28 (phase 3, `benchmark/trn2/four_model_native_vs_aot_2026-10-26.md`, section "Phase 3: bf16 VAE and umT5"):** bf16 vs fp32 at TP4 on the device, cosine 0.99948 for the campaign prompt and 0.99995 for the empty negative prompt (gate 0.995); the 50-step e2e scored 22.41 dB / 0.880 against the CPU fp32 reference vs 22.66 / 0.885 in fp32 (−0.25 dB, −0.005 SSIM; gate 0.5 dB / 0.01). Cold e2e A 533.9 → 454.7 s (umT5 load 172.6 → 85.8 s, store hit 2.84 GB per rank), peak HBM 15.15 → 11.83 GiB. bf16 is now the default (upstream `59f70ee`; `--text-encoder-dtype fp32` keeps the old precision). *Why:* native keeps umT5 in fp32 because of a design comment, never measured (`neuron/wan/text_encoder.py:9-13`), so the weight store sees no gain and every cold start re-reads the 22.72 GB fp32 checkpoint (172.5 s). That is the main reason native Wan's cold e2e trails AoT (542.7 vs 447.5 s), whose umT5 is all-bf16 (11.36 GB of shards). Check bf16 against fp32 per prompt, including the empty negative prompt, with `tests/numerical/neuron_parity.py` (cosine ≥ 0.995), then a 50-step e2e against today's fp32 device baseline. If it passes, make bf16 the default (the store would then write an 11.36 GB entry; projected cold umT5 load about 86 s, cold e2e about 456 s, per-rank weights 5.68 → 2.84 GB). Estimate: 2–3 h of device time.
- [ ] **F14 Fix two documentation errors about umT5.** *Why:* wrong facts mislead later decisions. (a) The AoT bf16 umT5 is 11.36 GB (4 × 2,840,786,528 B on disk), not the "10.84 GB" in `benchmark/trn2/native_vs_aot_tp4_2026-10-25.md` (lines 33 and 221) and `docs/reports/2026-10-26-weight-loading-native-vs-aot.md` (lines 156, 178, 214). (b) The comment in `neuron/wan/text_encoder.py:9-11` says transformers keeps `wo` in fp32 for bf16 loads; in the installed transformers 4.57.6 `_keep_in_fp32_modules` applies only to fp16 loads (`modeling_utils.py:4981-4987`), and UMT5 has no `_keep_in_fp32_modules_strict`, so a bf16 load casts `wo` to bf16 too. That premise is part of the reason umT5 was kept in fp32 (F13). Doc and comment only, no behaviour change. Estimate: 20 min.
  - [ ] **F14a** (open): the "10.84 GB" figures in the two docs above are still uncorrected.
  - [x] **F14b** **Done (phase 3):** the module docstring of `difflet/backends/neuron/wan/text_encoder.py` now says transformers' `_keep_in_fp32_modules` keeps `wo` in fp32 for fp16 loads only, so a bf16 load casts `wo` to bf16 too.

### Review findings on the F1/F2 change (2026-10-26, not fixed yet)

Found by the whole-change review of `5c39e5a..5f415e0`; none was refuted; the unit suites show no new failure. All four are code and unit-test changes, CPU only, about 1 h together.

- [ ] **R1 (major; device needed to confirm the failure) `--vae-dtype fp32` / `--vae-exec-mode eager` without `--device-vae` puts an fp32 or eager decoder on rank 0's DiT core,** a combination never run (the eager fp32 decoder peaks around 21 GiB by neuron-monitor). The code comment and this plan's F1 item say those flags alone keep the old path; they do not. Proposed fix: refuse them unless `--device-vae` (or `--host-vae` where allowed) is given, with a message naming the right flags; correct the comment and this plan.
- [ ] **R2 (major; device needed to confirm the failure) The in-process decode is the default at every shape and layout, but its HBM fit was checked only at 320×512×61,** leaving 0.5 GiB free on rank 0's core at TP2, and no pre-launch HBM check counts the decoder (`hbm.dit_fits` and the application's meta-bytes check count only the DiT). Proposed fix: when the decode is in process, add the decoder (weights plus its measured run-time overhead) to the pre-launch check and refuse with advice to use `--device-vae` when it does not fit.
- [ ] **R3 (minor) `--vae-in-process`, `--vae-dtype`, `--vae-exec-mode` are registered for every model** (`difflet/cli/main.py:176-205`) but read only by HunyuanVideo's neuron orchestrator; other models and HunyuanVideo's AoT path ignore them silently. Proposed fix: refuse them outside HunyuanVideo on the neuron backend.
- [ ] **R4 (minor) Stale docstring:** `_stage_vae_neuron` (`difflet/cli/orchestrators/hunyuan_video.py`) still says the decoder is "fp32 and eager by default"; `--device-vae` now runs the compiled bf16 decoder. Proposed fix: update the docstring.

### Approved 2026-10-26: F1 + F2 execution plan

Run as one sequential workflow of subagents (one device job at a time; raw logs under `/home/ubuntu/bench-2026-10-26/`; one commit per device bug with a CPU test where possible).

1. **F2 — bf16 VAE precision on the device.** Per-module bisection on one tile (device bf16 vs host bf16 vs host fp32, forward hooks); find the first op whose error jumps (suspects GroupNorm, Conv3d, SiLU); fix in the style of `core/workarounds.py`'s fp32 LayerNorm (an fp32 workaround or a targeted upcast). Pass: device bf16 full decode ≥ 50 dB PSNR vs the fp32 host decode; the fp32 path unchanged (rel-L1 ≤ 1e-6).
2. **F1a — a compiled decoder.** Preferred: Difflet's backend-neutral `models/hunyuan_video/vae/modeling_vae.py` decoder with the AoT path's host-side fixed-size tiling (edge tiles padded, one graph), compiled with `torch.compile(backend="neuron")`. Control: `torch.compile` of diffusers' decoder inside diffusers' tiling loop. Measure decode seconds (fp32 and bf16), cold compile, accuracy vs the fp32 host decode.
3. **F1b — frame write and the separate vae launch.** Split the 22 s write (device→host copy, post-processing, mp4 encode) and cut it. Option A (approved): decode inside the generate launch on rank 0's core after the denoise (implement `enable_vae_decoder`, adjust the CLI stage chain) **if HBM fits** beside the DiT; otherwise keep the one-core vae launch and record why.
4. **Measure and document.** (Done 2026-10-26; outcome under F1 above.) TP4 full method (cold compile, cold e2e, n=3 warm, step, HBM, frames checked) at the new default; update the native vs AoT table and the evidence doc. Target: warm e2e ≈ 150 s or less (from 222.5 s), decode ≤ 25 s; record actuals if missed.
5. **Review.** Whole-change review plus adversarial verification of each finding; unit suite vs baseline.

**Default-change rule (approved):** a new path becomes the default only if (a) accuracy holds (bf16 ≥ 50 dB vs the fp32 host decode; a compiled fp32 decoder ≥ 60 dB) and (b) it runs at TP2, TP4 and TP2×CP2; the old path stays reachable by a flag. Otherwise it ships as an opt-in flag.
