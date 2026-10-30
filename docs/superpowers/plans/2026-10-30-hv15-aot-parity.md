# HunyuanVideo 1.5 AOT 对齐 PyTorch Native —— Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 最后一个 Task 是 on-device campaign，按 `.claude/skills/difflet-device-verify/SKILL.md` 执行。

**Goal:** 让 `difflet compile/generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` 在 Trainium AOT（NxD 静态图）后端上端到端出视频，并具备 native（torch-neuronx）后端已有的全部能力：真 CFG、TP、SP、CP（ulysses / ring / gather_kv）、CFG-parallel、DP 与 JSONL 批处理、TeaCache（固定步长 / online-delta / 自适应）。

**Architecture:** 去噪循环、guider 解析、host 文本编码、视频写出这些与后端无关的部分从 native 抽成共享代码，AOT 和 native 共用一份（不再各写一套 loop）。AOT 只负责 DiT 的静态图：仍然包 diffusers 的 `HunyuanVideo15Transformer3DModel`（现有 `backbone15.py` 的做法），在其上就地替换并行线性层和注意力 processor。所有与 rank 相关的切分都走 `difflet.ops` 的 SPMD 原语（`SPMDRank` + `scatter_to_process_group_spmd`），这和 HV1.0 / Wan 的 AOT 写法一致；native 那种 Python int `narrow` 在 AOT 静态图里会把 rank 0 固化进 NEFF。VAE 用现有的 `vae15.py`（编译好的 decoder tile），放在单独的 1 核 stage 里跑。

**Tech Stack:** Python 3.10–3.12, PyTorch, diffusers (`HunyuanVideo15Transformer3DModel`, `HunyuanVideo15Pipeline`, `AutoencoderKLHunyuanVideo15`), neuronx-distributed (NxD ModelBuilder, `ColumnParallelLinear` / `RowParallelLinear`), NKI `attention_cte`, pytest。

**Spec:** 本计划依据的是 2026-10-30 对 main（`b089a4a`）做的 native vs AOT 对比（就是本会话里的差距清单）。两边都没有的功能（serving、多 shape bucketing、meanflow `timestep_r`、i2v）不在本计划范围内。

## 功能总览：补哪些、各自有什么用

| # | 功能 | 有什么用 | Task |
|---|---|---|---|
| 1 | Registry / 能力元数据改正（真 CFG、支持 CP / SP） | 现在 `is_distilled=True` 是错的。它让 `main._validate_cfg_parallel` 在**两个后端**上都拒绝 `--cfg-parallel`，`difflet plan` 也报 "scaffold"。改正后 CLI 和 planner 才能放行后面所有功能 | 1 |
| 2 | 共享的 HV1.5 去噪循环 + 真 CFG（读取 `guider_config.json`，scale 6.0，负向 prompt） | 1.5 的 checkpoint 每一步要跑两遍 CFG。AOT 现在挂的是 HV1.0 的 loop（蒸馏 guidance embedding），出不了正确的视频。共享一份 loop 能保证两个后端数值一致，维护也只需一处 | 2 |
| 3 | AOT DiT 张量并行（Megatron 列/行切分，覆盖 54 个 block 和 token refiner） | 8.3B 的 DiT 单核放不下 121 帧的激活；现在 AOT 每个 rank 都放一整份权重。TP 是其他所有并行方式的基础 | 3 |
| 4 | 默认使用 key-bounded flash attention（文本 padding 用 key 数表示，O(L) 内存） | 现在 AOT 默认的对称 mask 会出 NaN，修复方案藏在环境变量后面，而且用的是 O(L²) 的稠密 mask，长视频直接爆内存。改完后默认路径就是正确且省内存的 | 3 |
| 5 | CFG-parallel（两个 cfg 组各算一个分支，最后 all-gather） | native 的默认布局就是 TP2×CFG2：同样 4 核，每步省掉一次串行 DiT 前向 | 4 |
| 6 | SP（Megatron 序列并行，在 TP 组内切图像序列） | 降低每核的激活内存，用于更长的视频或更高的分辨率 | 5 |
| 7 | CP：ulysses / ring / gather_kv | 把图像 token 切到多个核上，降低单请求延迟、扩展到更长序列；三种模式各有 shape 和编译器约束，需要全部可选 | 6 |
| 8 | AOT 版 1.5 应用（把 DiT 适配到共享 loop）+ 自适应 TeaCache probe | 把第 2–7 项接起来；TeaCache 跳步约省 40% 去噪时间（HV1.0 native 实测 cadence 2：40.4 → 24.3 s） | 7 |
| 9 | CLI 的 AOT 分支：host encode → AOT transformer → AOT VAE → MP4；DP 与 `--requests` 批处理 | 用户入口。现在 Trainium 上直接报 `NotImplementedError` | 8 |
| 10 | 元数据和矩阵翻转：planner、`verify_cli`、feature catalog、README | 设备验证通过后对外宣布支持，并让回归矩阵覆盖这些 cell | 9 |
| 11 | On-device 验证 campaign | 证明以上功能在 trn2 上真实可用，并给出与 native 的精度、速度对比 | 10 |

**明确不做（及原因）**
- **AOT 的 fp32→bf16 DiT 副本**：AOT 在 compile 时由 NxD 预切分写出每个 rank 的权重，冷加载已经读的是切好的分片，native 的 bf16 副本是为了弥补它没有预切分。
- **token refiner 放到设备上**：AOT 的 flash kernel 在少于 128 个有效 key 时出 NaN（`backbone15.py` 注释，cclog 86）。host refiner 是精确的，并按 timestep 做了缓存。如果 Task 10 的 profile 显示 refiner 占单步时间超过 5%，再单独立项。

## Global Constraints

- native 的数值不能变：Task 2 重构前后，native TP2×CFG2 同 seed、同 embeds 的 latents 要 **byte-identical**（Task 10 会在设备上复核）。
- 不改 HV1.0 的代码路径和 NEFF cache key；不改其他模型的 cache key（`tests/unit/pipeline/test_tp_reduce_cache_table.py` 必须保持绿）。
- 不往仓库写入任何 AWS 实例 ID、主机名、账号信息（只能放在私有 artifact 页面）。
- `docs/` 被 gitignore 了但按惯例要入库：新增文档用 `git add -f`。
- CPU CI：`pytest tests/unit -q` 全绿；CPU 测试一律走 `cpu` backend（`DIFFLET_BACKEND=cpu`）。
- 每个 bug 一个 commit（device-verify skill 的规则），commit message 写清症状、根因和修法。
- 设备上要求的 shape 约束统一在**编译前** fail fast，报错里给出可行的 shape 或并行度建议。

## Review Focus

1. **空负向 prompt / 极短 prompt**：uncond 分支的 prompt 是 `""`，有效文本 token 可能只有个位数甚至 0。bounded attention 必须在 `valid_text=0` 时输出有限值，而且等价于只 attend 图像 key。→ Task 3 的 `test_bounded_processor_zero_valid_text_is_finite`。
2. **默认 shape 不满足并行约束**：480×848×121 是 31×30×53 = 49,290 个图像 token，`%4 = 2`（tp4 SP 不行），每 rank 24,645 也不是 128 的倍数（ring 不行）。必须在编译前报错并给出建议 shape，不能编译 20 分钟后才在 neuronx-cc 里崩。→ Task 5、Task 6 的 `test_*_rejects_*_before_compile`。
3. **JSONL 中单个请求的 `guidance_scale` 把 CFG 打开或关掉**：图的 batch 已经由布局固定，应当拒绝并说明原因（native 已有这个行为）。→ Task 8 的 `test_aot_request_guidance_toggle_rejected`。
4. **过期的编译产物**：改 `cp_mode`、`sp`、`cfg`、tp 必须换 cache 目录；改 TeaCache 的 cadence 或 speedup **不能**导致重编译（adaptive 需要的 probe 是例外，它是额外的 component）。→ Task 8 的 `test_aot_stage_cache_key_fields`。
5. **不带布局参数的 `difflet generate`**：Trainium 上的默认布局要和 native 一样是 TP2×CFG2，不能是 registry 旧的 tp4。→ Task 8 的 `test_aot_default_layout_is_tp2_cfg2`。

---

### Task 1: 改正 HV1.5 的 registry 能力元数据

**Files:**
- Modify: `difflet/registry.py`（`_register_builtin_hunyuan_video_15`，约 L346–373）
- Test: `tests/unit/planner/test_model_capabilities.py:20`、`tests/unit/registry/test_hunyuan_video_registration.py`、`tests/unit/cli/test_cli_cfg_parallel.py`

**Interfaces:**
- Produces: `ModelCapabilities(num_attention_heads=16, is_distilled=False, supports_cp=True, supports_sp=True, cp_modes=frozenset({"ulysses", "ring", "gather_kv"}))`；`default_parallel=DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True)`。

- [ ] **Step 1: 改测试**
  - `test_model_capabilities.py` 中 hv15 一行改为 `(16, False, True, True)`。
  - 在 `test_cli_cfg_parallel.py` 新增 `test_hv15_cfg_parallel_not_rejected_as_distilled`：对 hv15 调用 `main._validate_cfg_parallel(args)`（`args.cfg_parallel=True`、`tp_degree=2`），断言不抛异常。
  - 在 `test_hunyuan_video_registration.py` 新增 `test_hv15_default_parallel_is_tp2_cfg2`：断言 `tp_degree == 2` 且 `cfg_parallel_enabled is True`。
- [ ] **Step 2: 运行，确认失败**：`pytest tests/unit/planner/test_model_capabilities.py tests/unit/cli/test_cli_cfg_parallel.py tests/unit/registry/test_hunyuan_video_registration.py -q`，预期上面 3 处 FAIL。
- [ ] **Step 3: 修改 registry**：删掉 "CP and SP are both deferred" 那段注释，改为一句话说明：checkpoint 的 guider 每步跑两遍真 CFG（scale 6.0）。
- [ ] **Step 4: 跑全量**：`pytest tests/unit -q`。`test_verify_cli.py` 里由 registry 派生的集合（`DISTILLED`、`CP_UNSUPPORTED`）会跟着变，相关断言（`test_distilled_set_matches_documented_matrix`、`test_cp_unsupported_models`、`_EXPECTED_SKIPS` 中的 hv15 行、`test_plan_cells_counts`）在这一步一并更新为新值。hv15 在 `EXPECTED_FAIL_CELLS` 里仍然是 XFAIL，Task 9 才翻转。
- [ ] **Step 5: Commit**：`fix(registry): HunyuanVideo 1.5 runs true CFG and supports CP/SP`

### Task 2: 把 HV1.5 去噪循环抽成共享代码

**Files:**
- Create: `difflet/models/hunyuan_video/pipeline15.py`
- Modify: `difflet/backends/neuron/hunyuan_video15/pipeline.py`（改成薄封装，类名和构造参数不变）
- Modify: `difflet/backends/neuron/hunyuan_video15/transformer.py`（新增 `make_teacache_probe()`）
- Test: `tests/unit/backends/test_neuron_hv15_pipeline.py`（现有用例保持不变且通过）、`tests/unit/models/hunyuan_video/test_pipeline15.py`（新建）

**Interfaces:**
- Produces（`pipeline15.py`）：
  - `load_guidance(model_path: str, guidance_scale: float | None = None) -> float | None`：从 native 原样搬过来；native 模块里 re-export 这个名字。
  - `class HunyuanVideo15DiT(typing.Protocol)`：
    - 属性 `cfg: int`、`model_config: dict`、`image_tokens: int`；
    - `__call__(*, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, encoder_hidden_states_2, encoder_attention_mask_2, image_embeds) -> tuple[torch.Tensor]`；
    - `make_teacache_probe() -> TeaCacheProbe`；
    - `load(skip_warmup: bool = False) -> None`。
  - `class TeaCacheProbe(typing.Protocol)`：`reset() -> None`、`teacache_delta(latent_input, timestep) -> float | torch.Tensor`。
  - `class HunyuanVideo15Denoiser`：
    - `__init__(self, dit: HunyuanVideo15DiT, *, model_path, height, width, num_frames, dtype=torch.bfloat16, guidance_scale=None, teacache_cadence=None, teacache_online_delta_alpha=None, teacache_speedup=None, teacache_calibration_path=None)`；
    - `load(skip_warmup=False)`：加载 scheduler，再调 `dit.load`；
    - `__call__(*, prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2, negative_prompt_embeds=None, negative_prompt_embeds_mask=None, negative_prompt_embeds_2=None, negative_prompt_embeds_mask_2=None, num_inference_steps=50, generator=None, latents=None, callback=None) -> torch.Tensor`；
    - 可读写属性 `guidance_scale`（JSONL 按请求覆盖时要用）。
  - `VAE_SPATIAL, VAE_TEMPORAL = 16, 4` 移到这里；native transformer 改为从这里 import。
  - native 侧：`NeuronHunyuanVideo15TransformerApplication.make_teacache_probe()` 返回 `NeuronTeaCacheProbe(self.teacache_mod_input)`。
  - `TorchNeuronHunyuanVideo15Pipeline.__init__` 构造 dit 后创建 `self._denoiser = HunyuanVideo15Denoiser(self.dit, ...)`。`dit`、`guidance_scale`、`scheduler`、`teacache` 这些属性保持可访问，因为 CLI 和测试会读写。

- [ ] **Step 1: 写失败测试**：在 `test_pipeline15.py` 写 `test_denoiser_drives_any_dit_protocol`。用一个只返回 `hidden_states[:, :out_channels] * 0.1` 的假 DiT（`cfg=1`），配 `test_neuron_hv15_pipeline.py` 的 `snapshot` fixture（把 fixture 挪到 `tests/unit/models/hunyuan_video/conftest.py` 共享），跑 2 步。断言：输出 shape 为 `(1, C, (F-1)//4+1, H//16, W//16)`；CFG 开启时每步调用假 DiT 2 次；`cfg=2` 时每步调用 1 次，输入 batch 为 2。
- [ ] **Step 2: 运行，确认失败**（`ModuleNotFoundError: difflet.models.hunyuan_video.pipeline15`）。
- [ ] **Step 3: 实现**：把 native `__init__` 里 TeaCache 的构造逻辑和 `__call__` 的循环原样搬进 `HunyuanVideo15Denoiser`。唯一改动：`NeuronTeaCacheProbe(self.dit.teacache_mod_input)` 改为 `self.dit.make_teacache_probe()`。
- [ ] **Step 4: 运行**：`pytest tests/unit/models/hunyuan_video/test_pipeline15.py tests/unit/backends/test_neuron_hv15_pipeline.py tests/unit/cli/test_orchestrator_hv15_neuron.py -q`，全部 PASS，现有 native 测试不改一行。
- [ ] **Step 5: Commit**：`refactor(hv15): share the denoising loop between backends`

### Task 3: AOT DiT 的 TP 与默认 bounded attention

**Files:**
- Create: `difflet/backends/trainium/hunyuan_video/parallel15.py`
- Modify: `difflet/backends/trainium/hunyuan_video/backbone15.py`（`HunyuanVideo15BackboneInferenceConfig.add_derived_config`、`_HunyuanVideo15TraceModule.__init__`、`NeuronHunyuanVideo15BackboneApplication.convert_hf_to_neuron_state_dict`）
- Modify: `difflet/models/hunyuan_video/application.py`（`create_hunyuan_video15_backbone_config` 加并行参数；`_host_refiner_on` 对 1.5 恒为 True）
- Test: `tests/unit/backends/trainium/test_hv15_parallel.py`（新建）、`tests/unit/models/hunyuan_video/test_hunyuan_application.py`

**Interfaces:**
- Produces（`parallel15.py`，只 import `difflet.ops`，不直接 import 任何 backend 的 `ops_impl`）：
  - `install_tensor_parallel(model, tp: int, sp: bool = False) -> None`：切分布局与 native `_install_tensor_parallel` 完全一致（属性名不变，NxD 的切分器按名字切 checkpoint）。SP 下 row 层用 `reduce_output=False`，bias 的重复累加用 HV1.0 的 `_sp_unbias` 做法修正（从 `modeling_hunyuan_video.py` import，不要复制）。
  - `class JointBoundedProcessor(sp: bool = False)`：等价于 native 的 `_JointBoundedProcessor`。bounds 在图内由 `key_valid_len = mask.sum(1) + N` 经 `_keypad_bounds` 的写法生成（只用 expand / reshape；**不能**在图内调用 `mask_to_contiguous_bounds`，见 commit `cd54d0f`），然后调用 `difflet.ops.attention(..., bound_min, bound_max, tp_q=True, tp_k=True, tp_out=False)`。
  - `_joint_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb)` 和 `_joint_out(attn, img_out, txt_out, dtype)` 与 native 签名相同。
  - `HunyuanVideo15BackboneInferenceConfig` 新增默认值：`cfg_parallel_enabled=False`、`context_parallel_enabled=False`、`cp_mode="gather_kv"`、`sp_enabled=False`。
  - `create_hunyuan_video15_backbone_config(..., cfg_parallel_enabled=False, cp_degree=1, cp_mode="gather_kv", sp_enabled=False)`。
  - `_HunyuanVideo15TraceModule`：1.5 的生产路径恒定启用 static-reorder forward、`JointBoundedProcessor` 和 host refiner。删除 `DIFFLET_HUNYUAN15_KEY_MASK_ATTENTION` / `DIFFLET_HUNYUAN15_HOST_REFINER` 两个开关和 `_HunyuanVideo15KeyMaskAttnProcessor`；其他诊断开关（`NO_ATTENTION_MASK`、`NKI_ATTENTION`、`ZERO_BLOCK_*`）保持不变。`tp > 1` 时调用 `install_tensor_parallel`。
  - `convert_hf_to_neuron_state_dict`：cfg、cp、sp 任一开启时写入 `state_dict["transformer.global_rank.rank"] = torch.arange(world_size, dtype=torch.int32)`（键名跟随 `transformer.` 前缀，并与 `SPMDRank` 实际挂载的位置一致）。

- [ ] **Step 1: 写失败测试**（tiny config：2 层，4 头，head_dim 8，refiner 1 层；CPU backend，world 1）
  - `test_tp_install_keeps_parameter_names`：install 前后 `state_dict().keys()` 相同；`to_q` 是 `difflet.ops.ColumnParallelLinear`，`to_out[0]` 是 `RowParallelLinear`。
  - `test_bounded_processor_matches_diffusers_on_valid_tokens`：同一组随机权重，mask 只有前 3 个文本 token 有效，比较 diffusers 默认 processor 与 `JointBoundedProcessor`。图像 token 和有效文本 token 的输出 `torch.allclose(atol=1e-5)`（fp32）。
  - `test_bounded_processor_zero_valid_text_is_finite`：mask 全 0 时输出全部有限，且图像 token 输出等于只对图像 key 做 attention 的结果。
  - `test_state_dict_adds_global_rank_when_sharding`：`cfg_parallel_enabled=True`、world 4 时，`transformer.global_rank.rank` 等于 `arange(4)`；全部关闭时没有这个键。
  - `test_trace_module_uses_bounded_processor_without_env`：未设置任何 `DIFFLET_HUNYUAN15_*` 环境变量时，每个 block 的 processor 都是 `JointBoundedProcessor`。
- [ ] **Step 2: 运行，确认失败**：`pytest tests/unit/backends/trainium/test_hv15_parallel.py -q`
- [ ] **Step 3: 实现上面的 Interfaces**
- [ ] **Step 4: 运行**：新测试加 `tests/unit/models/hunyuan_video/ -q` 全 PASS。`scripts/` 里引用已删除环境变量的脚本不需要改（设置一个不存在的环境变量是无害的），但要用 `grep -rn KEY_MASK_ATTENTION scripts/` 确认没有脚本 import `_HunyuanVideo15KeyMaskAttnProcessor`。
- [ ] **Step 5: Commit**：`feat(trainium/hv15): tensor-parallel DiT with key-bounded attention by default`

### Task 4: AOT 的 CFG-parallel

**Files:**
- Modify: `difflet/backends/trainium/hunyuan_video/backbone15.py`（trace module 的 forward 和 `ModelWrapperHunyuanVideo15Backbone.input_generator`）
- Modify: `difflet/models/hunyuan_video/entry.py`（删除 L72–77 的 "guidance-distilled" 拒绝）
- Modify: `difflet/models/hunyuan_video/application.py`（`cfg_parallel_enabled` 时 `batch_size=2`）
- Test: `tests/unit/models/hunyuan_video/test_hunyuan_entry.py`、`tests/unit/backends/trainium/test_hv15_parallel.py`

**Interfaces:**
- Consumes：Task 3 的 config 字段和 `global_rank`。
- Produces：开启 CFG 时 trace module 的 forward 接受 batch 2（`[uncond, cond]`）。做法同 Wan（`difflet/models/wan/modeling_wan.py` L828–850 与 L949–952）：`cfg_rank = get_cfg_rank_spmd(self.global_rank.get_rank())`，7 个输入各自 `scatter_to_process_group_spmd(x, partition_dim=0, rank=cfg_rank, process_group=get_cfg_group())`，输出再 `gather_from_tensor_model_parallel_region_with_dim(out, gather_dim=0, process_group=cfg_group)`。CFG 与 CP 同时开启时，只在 `difflet.pipeline.parallel_config.cfg_cp_composable()` 为真时允许，否则报错。

- [ ] **Step 1: 测试**
  - 把 `test_create_application15_rejects_cfg_parallel` 改名为 `test_create_application15_accepts_cfg_parallel`，断言 `application.batch_size == 2`。
  - 新增 `test_cfg_input_generator_traces_batch2`：开启 cfg 时 `input_generator()` 每个张量的第 0 维为 2。
  - 新增 `test_cfg_with_cp_follows_composable_rule`：monkeypatch `cfg_cp_composable` 返回 False，断言构造时抛 `NotImplementedError`。
- [ ] **Step 2: 运行，确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 运行 `pytest tests/unit/models/hunyuan_video tests/unit/backends/trainium/test_hv15_parallel.py -q`，PASS**
- [ ] **Step 5: Commit**：`feat(trainium/hv15): CFG-parallel DiT`

### Task 5: AOT 的 SP

**Files:**
- Modify: `difflet/backends/trainium/hunyuan_video/parallel15.py`、`backbone15.py`
- Test: `tests/unit/backends/trainium/test_hv15_parallel.py`

**Interfaces:**
- Produces：
  - `install_sequence_parallel(model, num_image_tokens: int, tp: int) -> None`：给 block 0 加 pre-hook，用 `scatter_to_process_group_spmd(x, partition_dim=1, rank=self.global_rank.get_rank(), process_group=None)` 把图像流切到本 rank（**不能**用 nxd 的 `scatter_to_sequence_parallel_region`，它在 SPMD trace 下会把 rank 固化成常量，见 `modeling_hunyuan_video.py` L1274–1283 的注释）。给最后一个 block 加 post-hook，调 `gather_from_sequence_parallel_region`。
  - `JointBoundedProcessor(sp=True)` 先对图像流 all-gather；图像 MLP 先 gather 再 reduce-scatter（native 的 `_sp_mlp_forward` 在 `difflet.ops` 上的版本）。
  - 校验（编译前）：`sp` 要求 `tp > 1`、`cp == 1`、`num_image_tokens % tp == 0`；报错文本要给出图像 token 数和可行的 frames / tp 建议。

- [ ] **Step 1: 测试**
  - `test_sp_rejects_indivisible_tokens_before_compile`：480×848×121、tp 4、sp → `ValueError`，消息里包含 `49290`。
  - `test_sp_rejects_cp`
  - `test_sp_world1_matches_dense`：tiny config、world 1 下 SP 开与关的输出 `allclose(atol=1e-5)`（scatter 和 gather 在 world 1 时退化为恒等，用来验证代码路径能跑通）。
- [ ] **Step 2–4:** 确认失败 → 实现 → 运行通过
- [ ] **Step 5: Commit**：`feat(trainium/hv15): Megatron sequence parallelism`

### Task 6: AOT 的 CP（ulysses / ring / gather_kv）

**Files:**
- Modify: `difflet/backends/trainium/hunyuan_video/parallel15.py`、`backbone15.py`
- Modify: `difflet/models/hunyuan_video/entry.py`（删除 L70–71 的 "CP is deferred"）
- Modify: `difflet/models/hunyuan_video/application.py`（`world_size = tp * cp * (2 if cfg else 1)`）
- Test: `tests/unit/backends/trainium/test_hv15_parallel.py`、`tests/unit/models/hunyuan_video/test_hunyuan_entry.py`、`tests/unit/models/hunyuan_video/test_hunyuan_application.py`（`test_backbone15_world_size_is_process_world` 里 "1.5 has no CP yet" 的注释和断言随之更新）

**Interfaces:**
- Produces：
  - `install_context_parallel(model, num_image_tokens: int, cp: int, mode: str) -> None`：每个 block 的 pre-hook 用 `scatter_to_process_group_spmd(..., rank=get_cp_rank_spmd(global_rank), process_group=get_cp_group())` 切图像流和 rotary 的对应行，最后一个 block 之后沿 CP 组 all-gather。
  - `JointUlyssesProcessor(num_image_tokens)` → `difflet.ops.joint_ulysses_attention(q, eq, k, v, ek, ev, scale=..., key_valid_len=mask.sum(1) + N)`。
  - `JointRingProcessor()` → `difflet.ops.joint_ring_attention(cat([q, eq]), k, v, ek, ev, scale=..., text_valid_len=mask.sum(1))`。
  - `JointGatherKvProcessor(num_image_tokens)`：图像 K/V 沿 CP 组 all-gather。**与 native 不同**：AOT 的 bounded kernel 要求 `q_len == kv_len`（`trainium/ops_impl/attention.py` 中 `joint_ulysses_attention` 的 docstring），所以这里改用 HV1.0 的 gather_kv 做法：key-only additive mask 加 `difflet.ops.attention(..., attention_mask=...)`，见 `modeling_hunyuan_video.py` L534–551。
  - 编译前校验：
    - ulysses：`(16 // tp) % cp == 0` 且 `N % cp == 0`；
    - ring：`(N // cp) % 128 == 0`（feature catalog 中 ring 的规则），报错里给出满足条件的 shape，例如 320×512×61 → 每 rank 5,120；
    - 所有模式：`N % cp == 0`。

- [ ] **Step 1: 测试**
  - `test_cp_processors_world1_match_bounded`：world 1 时三个 processor 在有效 token 上的输出都与 `JointBoundedProcessor` `allclose(atol=1e-4)`。
  - `test_ring_rejects_default_shape_before_compile`：480×848×121、cp 2、ring → `ValueError`，消息里包含 `24645` 和 `128`。
  - `test_ulysses_rejects_heads_indivisible`：tp 4、cp 8 → `ValueError`。
  - 把 `test_create_application15_rejects_context_parallel` 改为 `test_create_application15_accepts_context_parallel`，三种 mode 参数化。
- [ ] **Step 2–4:** 确认失败 → 实现 → 运行通过
- [ ] **Step 5: Commit**：`feat(trainium/hv15): context parallelism (ulysses, ring, gather_kv)`

### Task 7: 把 AOT 版 1.5 应用适配到共享 loop，并接上 TeaCache probe

**Files:**
- Create: `difflet/models/hunyuan_video/aot15.py`
- Modify: `difflet/models/hunyuan_video/application.py`（`model_version == "1.5"` 时不再构造 1.0 的 `HunyuanVideoOrchestrator`，`self.pipeline = None`；直接用 kwargs 调 `__call__` 的旧行为不变）
- Modify: `difflet/backends/trainium/hunyuan_video/teacache_probe15.py`（probe 继承 trace module，开启 TP 时一起切分，与 backbone 共享权重）
- Test: `tests/unit/models/hunyuan_video/test_aot15.py`（新建）

**Interfaces:**
- Consumes：Task 2 的 `HunyuanVideo15DiT` / `TeaCacheProbe` 协议；Task 3–6 的 application。
- Produces：
  - `class AotHunyuanVideo15DiT`（实现 `HunyuanVideo15DiT` 协议）：`__init__(self, app: NeuronHunyuanVideoApplication, compiled_model_path: str, *, cfg: int)`。
    - `__call__` 组装 `HunyuanVideo15DiTInputBundle`（`timestep_r` 填 0），调 `app.forward_dit`，返回 `(noise_pred.cpu(),)`；
    - `model_config` 取 `app.transformer_config`；
    - `image_tokens` 取 `app.image_seq_len`；
    - `load(skip_warmup)` 调 `app.load(compiled_model_path, skip_warmup=...)`；
    - `make_teacache_probe()`：需要 adaptive 时返回包着 `NeuronHunyuanVideo15TeacacheProbeFusedApplication` 的对象，`teacache_delta` 返回 fused probe 的 rel-L1 标量，`reset()` 把 `prev_mod` 清零；否则抛 `RuntimeError("adaptive TeaCache needs the teacache_probe component compiled")`。
  - `build_aot_denoiser(*, model_path, compiled_model_path, parallel, height, width, num_frames, guidance_scale=None, teacache_*=...) -> HunyuanVideo15Denoiser`：工厂函数，供 CLI 调用。

- [ ] **Step 1: 测试**（用 fake app 记录 `forward_dit` 收到的 bundle）
  - `test_aot_dit_bundle_fields`：bundle 的 8 个字段齐全，`timestep_r` 为 0，dtype 为 bf16。
  - `test_aot_denoiser_cfg2_one_call_per_step`：`cfg=2` 时每步只调一次 `forward_dit`，hidden_states 的 batch 为 2。
  - `test_aot_probe_required_for_adaptive`：没有 probe component 却传了 `teacache_speedup` → `RuntimeError`。
  - `test_application15_has_no_v10_pipeline`：1.5 的 `app.pipeline is None`。
- [ ] **Step 2–4:** 确认失败 → 实现 → 运行通过
- [ ] **Step 5: Commit**：`feat(trainium/hv15): drive the AOT DiT from the shared loop, with the TeaCache probe`

### Task 8: CLI 的 AOT 分支（encode → transformer → vae）+ DP / JSONL

**Files:**
- Modify: `difflet/cli/orchestrators/hunyuan_video_15.py`
- Test: `tests/unit/cli/test_cli_hunyuan_video_15_orchestrator.py`、`tests/unit/cli/test_orchestrator_hv15_neuron.py`（native 用例保持不变）

**Interfaces:**
- Consumes：Task 7 的 `build_aot_denoiser`；`vae15.NeuronHunyuanVideo15VAEDecoderApplication`；`difflet.cli.orchestrators.base.hashed_stage_dir` / `write_stage_manifest` / `has_valid_stage_manifest`；`difflet.cli.dp.stage_loop`；`difflet.cli.orchestrators.torchrun.replica_request_scope`。
- Produces：
  - `_run` 不再对非 neuron backend 报错。共用的部分：`_layout`（默认 TP2×CFG2）、`encode` stage（host，原样共用）、`_stage_cli_args`。backend 为 `trainium` 时：
    - `transformer` stage：`runner.run_stage(..., num_cores=tp * (2 if cfg else 1) * cp, virtual_core_size=2, ...)`（与 HV1.0 的 `_VIRTUAL_CORE_SIZE=2` 一致）。compile 模式下调用 `app.compile(compiled_dir)` 并写 manifest；generate 模式下用 `build_aot_denoiser` 跑每个 claim 到的请求，rank 0 写 `latents.pt`。
    - `vae` stage：1 核，`vae15` 的 decoder，自己独立的 compiled 目录（world 1，**不能**和 DiT 同进程）。
  - `_aot_stage_cache_inputs(args, component: str) -> dict`：
    - transformer 的字段包括 `model`、`revision`、`tp`、`cp`、`cp_mode`、`sp`、`cfg`、`dtype`、`height/width/num_frames`、`text_seq_len=1000`、`text_seq_len_2=256`、`toolchain`、`teacache_probe: bool`；
    - vae 的字段包括 `model`、`dtype`、`shape`、`tile`、`toolchain`；
    - **不能**包含 cadence、speedup、steps、seed、prompt。
  - `_write_output(video: torch.Tensor, out: Path) -> None`：从现有 `_stage_vae` 中抽出 `.pt` / MP4（24 fps，diffusers 后处理）的写出逻辑，两个后端共用。
  - JSONL 中按请求覆盖 `guidance_scale` 的规则与 native 相同：如果会打开或关掉 CFG，就报错。

- [ ] **Step 1: 测试**（mock `runner.run_stage` 和 application）
  - `test_aot_generate_runs_three_stages`：stage 顺序为 `encode → transformer → vae`，核数为 `1, 4, 1`（默认 TP2×CFG2）。
  - `test_aot_default_layout_is_tp2_cfg2`
  - `test_aot_stage_cache_key_fields`：改 `cp_mode`、`sp`、`cfg`、`tp` 任一项，目录都变；改 `teacache_cadence` 目录不变；加 `teacache_speedup`（adaptive）后 `teacache_probe=True`，目录改变。
  - `test_aot_request_guidance_toggle_rejected`：一个请求的 `guidance_scale=1.0` 会关掉 CFG → `ValueError`。
  - `test_aot_requests_write_per_request_latents`：两个请求分别写出 `latents_req0000.pt` 和 `latents_req0001.pt`。
  - 删除或改写断言 "Trainium 报 NotImplementedError" 的旧用例。
- [ ] **Step 2–4:** 确认失败 → 实现 → `pytest tests/unit/cli -q` 全部 PASS
- [ ] **Step 5: Commit**：`feat(cli/hv15): end-to-end AOT generate with DP and JSONL batches`

### Task 9: 元数据与验证矩阵翻转（Task 10 中对应的 cell PASS 之后才合入）

**Files:**
- Modify: `difflet/planner/feasibility.py`（删除 `KNOWN_BAD[("hunyuan_video_15", None)]`）、`difflet/planner/model_profile.py`（`unverified` 去掉；`WeightFootprint` 填 DiT 实际字节数，取自 Task 10 的 load 日志）
- Modify: `scripts/verify_cli.py`（`MODELS["hunyuan_video_15"]`：`shape_flags=("--height","320","--width","512","--num-frames","61")`、`staged=True`；从 `EXPECTED_FAIL_CELLS` 里删掉 hv15；凡是 Task 10 结论为 BLOCKED 或 LIMIT 的 cell 加回去并写明原因）
- Modify: `.claude/skills/difflet-device-verify/references/feature-catalog.md`（hv15 那一行，加一个新 section："Trainium AOT backend: HunyuanVideo 1.5"）
- Modify: `README.md`（Trainium 的两张表各加一行 HunyuanVideo 1.5；Supported models 表加一行）
- Test: `tests/unit/cli/test_verify_cli.py`（`test_staged_flags`、`test_expected_fail_cells`、`_EXPECTED_SKIPS`、`test_plan_cells_counts`、`test_hunyuan_video_15_compile_cmd_minimal` → 改为断言命令里包含 `--height 320`）、`tests/unit/planner/test_feasibility.py:169`、`tests/unit/planner/test_model_profile.py`

- [ ] **Step 1: 按 Task 10 的结论改测试** → **Step 2: 确认失败** → **Step 3: 改代码和文档** → **Step 4: `pytest tests/unit -q` 全绿**
- [ ] **Step 5: Commit**（矩阵翻转单独一个 commit，按 skill 规则）：`chore(matrix): HunyuanVideo 1.5 on Trainium AOT verified`

### Task 10: On-device 验证 campaign

按 `difflet-device-verify` skill 执行。分支 `verify/hv15-aot-<date>`；证据放在 `artifacts/verification-<date>/hv15-aot/`；报告写到 `docs/verification/hv15-aot-evidence.md`（`git add -f`）。每完成一行就 checkpoint、commit、push。不要把实例 ID 写进仓库。

**Phase 0 —— 主机准备**
- [ ] `scripts/env_check.sh`。当前机器的 repo 下没有 `.venv`，需要先跑 `./scripts/setup_env.sh`，并且 AOT 和 native 两个环境都要有（native 回归 cell 要用）。
- [ ] 下载 `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`；用 `du` 预估磁盘。
- [ ] **native 基线**：在 Task 2 之前的 commit（`b089a4a`）上跑 native TP2×CFG2，320×512×61、50 步、seed 42，保存 `latents.pt` 的 sha256。之后的回归对比和精度对比都以它为参照。

**Phase 1 —— 正确性门槛（3 步，320×512×61，seed 42，embeds 与 native 基线完全一致）**

| Cell | 内容 | 通过标准 |
|---|---|---|
| native-regress | Task 2 之后的 native TP2×CFG2 | latents 与 Phase 0 基线 **byte-identical** |
| aot-tp2 | AOT TP2，CFG 串行两遍 | 与 native TP2 的 latents 余弦 ≥ 0.999，无 NaN / Inf |
| aot-tp2cfg | AOT TP2×CFG2（默认布局） | 与 aot-tp2 余弦 ≥ 0.999 |
| aot-tp4 | AOT TP4 | 与 aot-tp2 余弦 ≥ 0.999 |
| aot-tp4sp | AOT TP4 + SP | 与 aot-tp4 余弦 ≥ 0.999 |
| aot-tp2cp2 ×3 | gather_kv / ring / ulysses | 各自与 aot-tp2 余弦 ≥ 0.999。gather_kv 如果复现 HV1.0 的 `NCC_INLA001`/`NCC_IBIR243`，按 skill 第 5 节记为 BLOCKED，加 fail-fast 保护，并记录后续跟进 |

报告中每个 cell 都附上 rel-L1、余弦、最大绝对误差，以及与 diffusers CPU bf16 参照的距离（9 帧，方法同 `neuron-backend-results.md`）。

**Phase 2 —— 并行矩阵（`scripts/run_cells.sh "hunyuan_video_15" "tp4 tp2cp2 tp2cp2ring tp2cp2ulysses tp2cfg tp4sp dp2tp2"`，50 步）**
- [ ] 每个 cell 记录 compile 秒数、generate 秒数（staged 墙钟时间）、单步中位数、每核峰值 HBM（neuron-monitor），并输出 MP4 和中间帧 PNG。
- [ ] dp2tp2：两个请求的 latents 与单副本同 seed 的运行 byte-identical（`scripts/verify_dp_correctness.py`，视频模型比较 latents）。如果和 HV1.0 一样在加载 VAE 时 HBM 不够，记为 LIMIT，并写明可行的 shape。

**Phase 3 —— 默认 shape 的头条 cell**
- [ ] AOT TP2×CFG2，480×848×121，50 步：e2e 时间、单步时间、MP4，与 native 同布局对比（native 单步 12.6 s，数据来自 `neuron-backend-results.md`，同机重测为准）。
- [ ] ring 和 tp4 SP 在默认 shape 下应当 **在编译前** 报错（Review Focus #2）。记录报错文本和耗时（应在秒级）。

**Phase 4 —— TeaCache**
- [ ] 固定步长：`scripts/teacache_cadence_ab.sh hunyuan_video_15`，baseline 对比 `--teacache-cadence 2`。日志里要有 `[teacache] stats: ... skipped_steps 20`（50 步时），且 latents 余弦 ≥ 0.99。在 TP2×CFG2 下跑，以证明 CFG 两遍一起跳过。
- [ ] online-delta：`--teacache-online-delta <alpha>`，记录跳步数和余弦。
- [ ] 自适应：先用 `scripts/calibrate_teacache.py` 生成 hv15 的 calibration（如果这个脚本不支持 hv15，用一个单独的 commit 补上 hook），再跑 `--teacache-speedup`。报告里写信号拟合度（Pearson / R²）。`teacache_probe15.py` 的注释预测 HV1.5 的信号偏弱，控制器因此几乎不跳步属于有效结论，不算失败。

**Phase 5 —— JSONL 批处理**
- [ ] `--requests` 带两个 prompt（其中一个带负向 prompt、一个自定义 seed）：输出两个 MP4，各自与单独运行时的 latents byte-identical。

**Phase 6 —— 报告**
- [ ] 证据文档：每个 cell 的结果、时间、证据路径、复现命令；bug ledger。
- [ ] 支持矩阵（PASS / LIMIT / BLOCKED / N/A / NOT MEASURED），同时提供"模型 × 功能"表和"按功能分块"两种形式。
- [ ] 回到 Task 9，按结论翻转矩阵。
