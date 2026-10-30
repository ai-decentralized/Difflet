# HunyuanVideo 1.5 AOT 对齐 PyTorch Native —— Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development（执行方式已由用户选定）to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Task 0 和 Task 10 在设备上执行，按 `.claude/skills/difflet-device-verify/SKILL.md` 进行。

**Goal:** 让 `difflet compile/generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` 在 Trainium AOT（NxD 静态图）上端到端生成视频，并具备 native 后端已有的能力：真 CFG、TP、SP、CP（ulysses、gather_kv；ring 见决策点）、CFG-parallel、DP 与 JSONL 批处理，以及 TeaCache 的固定步长、online-delta 和自适应三种模式。

**Architecture:**
- **native 代码一行不动。** AOT 写自己的 HV1.5 去噪循环（`difflet/models/hunyuan_video/pipeline15.py`），以 diffusers 的 `HunyuanVideo15Pipeline` 为语义参照，单元测试直接对比 diffusers。
- **DiT 静态图**仍然包 diffusers 的 `HunyuanVideo15Transformer3DModel`（沿用 `backbone15.py` 的方式），在上面就地换成 NxD 的并行线性层和我们自己的 attention processor。
- **按 rank 切分**全部走 `difflet.ops` 的 SPMD 原语（`SPMDRank` + `scatter_to_process_group_spmd`），与 HV1.0 / Wan 的 AOT 写法一致。
- **文本 token 重排**（有效 token 排在前面）用的置换由 host 计算，作为图的输入传进去；图内只做静态形状的 `gather`。
- **token refiner** 留在 host 上运行（AOT 的 flash kernel 在少于 128 个有效 key 时会出 NaN）。
- **VAE** 用现有的 `vae15.py` 编译好的 decoder，在独立的 1 核 stage 里运行。

**Tech Stack:** Python 3.10–3.12、PyTorch、diffusers（`HunyuanVideo15Transformer3DModel`、`HunyuanVideo15Pipeline`、`AutoencoderKLHunyuanVideo15`、`ClassifierFreeGuidance`）、neuronx-distributed（ModelBuilder、`ColumnParallelLinear` / `RowParallelLinear`）、NKI `attention_cte`、pytest。

**Spec:** 依据 2026-10-30 对 main（`b089a4a`）做的 native vs AOT 差距清单。两边都没有的功能（serving、多 shape bucketing、meanflow `timestep_r`、i2v）不在本计划范围内。本版已经过一轮代码核对（5 个审查角度 × 对抗式复核，36 条确认问题全部吸收），以及一轮 Fable 5.1 独立 review（15 条全部吸收）。

## 功能总览：补哪些、各自有什么用

| # | 功能 | 有什么用 | Task |
|---|---|---|---|
| 1 | Registry 元数据改正（真 CFG，支持 CP / SP，默认 TP2×CFG2）以及 CLI 布局一致性 | 现在 `is_distilled=True` 是错的，`--cfg-parallel` 在**两个后端**上都被拒绝；DP router 算出的每副本核数也必须和 stage 实际用的核数一致 | 1 |
| 2 | AOT 自己的去噪循环 + 真 CFG（guider 6.0、负向 prompt） | 1.5 每步都跑两遍 CFG；AOT 现在挂的是 HV1.0 的 loop，出不了正确的视频 | 2 |
| 3 | DiT 张量并行 | 现在每个 rank 都装着一整份 8.3B 权重；TP 是其他所有并行方式的基础 | 3 |
| 4 | 默认使用 key-bounded flash attention（文本 padding 用"有效 key 个数"表达，O(L) 内存） | AOT 现在默认路径出 NaN，修复藏在环境变量后面，而且是 O(L²) 的稠密 mask | 3 |
| 5 | CFG-parallel | native 的默认布局：同样 4 核，每步少跑一次串行前向 | 4 |
| 6 | SP | 降低每核的激活内存 | 5 |
| 7 | CP：ulysses（必做）、gather_kv（视 Task 0 spike 结果） | 把图像 token 切到多个核上，降低单请求延迟、支持更长序列 | 6 |
| 8 | CP：ring（**需要你决定**，见下面的决策点） | 同上，另一种 CP 实现 | 6a（可选） |
| 9 | AOT 应用接入循环 + TeaCache probe | 把上面这些串起来；TeaCache 大约省 40% 去噪时间 | 7 |
| 10 | HV1.5 TeaCache calibration 采集 | 自适应 TeaCache 需要 calibration；现有脚本只支持 HV1.0 | 7b |
| 11 | CLI 的 AOT 分支 + DP / JSONL | 用户入口；现在在 Trainium 上直接 `NotImplementedError` | 8 |
| 12 | 元数据和验证矩阵翻转 | 设备验证通过后对外宣布支持 | 9 |
| 13 | 设备 spike（Task 0）和 on-device campaign（Task 10） | 尽早排除编译器风险，最后在 trn2 上证明可用，并与 native 对比精度和速度 | 0、10 |

## 决策点（需要你在 review 时拍板）

**ring CP 怎么处理？** Trainium 的 `joint_ring_attention` 只要收到文本 padding 长度（`text_valid_len`）就抛 `NotImplementedError`（`difflet/backends/trainium/ops_impl/attention.py:278-282`）。HV1.5 的文本永远带 padding（uncond 分支的 prompt 是 `""`，ByT5 那一路通常整段都是 padding），feature catalog 也已经把 HV1.0 的 AOT ring 标成 N/A。有两个选项：
- **(A，推荐)** AOT 上不支持 hv15 ring：编译前就报错，提示改用 ulysses，矩阵里记为 N/A（与 HV1.0 一致）。零风险。
- **(B)** 加 Task 6a：给 Trainium 的 `joint_ring_attention` 的文本 partial 加上 key mask。这会改一个 HV1.0 也在用的共享 op，需要单独做设备 spike，并让 HV1.0 的 ring cell 重新跑一遍回归。

## 明确不做（及原因）
- **AOT 的 fp32→bf16 DiT 副本**：NxD 编译时已经把每个 rank 的权重预切分好写到磁盘，冷加载读的是切分后的分片。
- **把 token refiner 放到设备上**：AOT 的 flash kernel 在少于 128 个有效 key 时会出 NaN（cclog 86）；host refiner 结果精确，并且按 timestep 缓存。如果 Task 10 的 profile 显示它占单步时间超过 5%，再另行立项。

## Global Constraints

- **native 代码不动**：`git diff main -- difflet/backends/neuron/ tests/unit/backends/test_neuron_hv15_pipeline.py` 必须为空。`tests/unit/cli/test_orchestrator_hv15_neuron.py` 只允许一处改动：删掉 `test_aot_backend_is_not_implemented`（L83），对应用例挪到 AOT 的测试文件里。`difflet/cli/orchestrators/hunyuan_video_15.py` 中现有的 `_stage_encode`、`_stage_transformer`、`_stage_vae`、`_stage_cli_args`、`_layout` 的函数体不改，只允许新增 AOT 方法和分发语句。每个 Task 的 reviewer 都要检查这一条。
- **AOT 代码不能 import `difflet.backends.neuron`**。这里的 AOT 代码包括 `difflet/models/hunyuan_video/`、`difflet/backends/trainium/hunyuan_video/`，以及 orchestrator 里新增的 AOT 方法。注意 `difflet.cli.orchestrators.torchrun.replica_request_scope` 内部会 import native，AOT 不能使用它。
- 图内所有整数归约（mask 求和、计数）都先在 **fp32** 里算，再转成 int32。原因是 Trainium LNC2 会把长整数归约编译错，见 `modeling_hunyuan_video.py:1475-1481`。
- 不改 HV1.0 的代码路径和 NEFF cache key，也不改其他模型的 cache key（`tests/unit/pipeline/test_tp_reduce_cache_table.py` 必须保持绿）。
- 仓库里不写任何 AWS 实例 ID、主机名或账号信息。
- `docs/` 在 gitignore 里，但按惯例要入库，新文档用 `git add -f`。
- CPU CI：`pytest tests/unit -q` 全部通过；CPU 测试一律在 `DIFFLET_BACKEND=cpu` 下运行。
- 每个 bug 单独一个 commit；shape 约束一律在**编译前** fail fast，报错里要给出可行的 shape 或并行度建议。

## Review Focus

1. **文本有效 token 不是前缀**：t2v 时 image 段（729 个 token）整段无效，ByT5 段通常整段是 padding，而有效的 mllm token 排在后面。重排必须把有效 token 移到最前面，否则 bounded attention 会 attend 到错误的 key，并且**不会报 NaN**。对应测试：Task 3 `test_trace_module_matches_diffusers_non_prefix_mask`。
2. **空负向 prompt**：有效文本可能只有个位数甚至 0 个 token，bounded attention 的输出必须是有限值。对应测试：Task 3 `test_bounded_processor_zero_valid_text_is_finite`。
3. **joint 序列长度不是 128 的倍数**：N + 1985 永远不是 128 的倍数，必须在图内 pad，否则会在编译阶段失败。对应测试：Task 3 `test_bounded_processor_padding_is_exact`，以及 Task 0 的 spike。
4. **布局与核数一致**：在 DP、`--cfg-parallel`、`--mode latency` 或完全不给布局参数的情况下，router 分配给每个副本的核数必须等于 transformer stage 实际使用的核数。对应测试：Task 1 `test_replica_cores_match_stage_layout`。
5. **多请求之间的状态串扰**：JSONL 里连续的请求不能复用上一个请求的 refiner 缓存、TeaCache 统计和 probe 状态。对应测试：Task 2 `test_teacache_state_resets_per_call`，Task 7 `test_refiner_cache_reset_per_request`。

---

### Task 0: 设备 spike —— attention 形状与编译器风险（在 Task 3 之前）

**为什么先做：** 有两个风险只有在设备上才能暴露，而且会影响 Task 3 和 Task 6 的设计：(a) bounded kernel 在 pad 到 128 之后、长度为 N+1985 时能否编译；(b) gather_kv 那种"分片的 q 对全量的 k"的 bounded attention 会不会复现 HV1.0 的 `NCC_INLA001` / `NCC_IBIR243`。

**Files:**
- Create: `scripts/hv15_aot_attention_spike.py`（参数：`--mode bounded|ulysses|gather_kv`、`--height/--width/--num-frames`、`--tp`、`--cp`）。只构建**一个** `HunyuanVideo15TransformerBlock` 的 NxD trace，用随机权重，编译后跑一次 forward，打印 compile 秒数、是否出现 NaN，以及与 CPU fp32 参考结果的 rel-L1。

- [ ] **Step 1:** 在设备主机上执行 Task 10 的 Phase 0，准备好环境。
- [ ] **Step 2:** 跑三组：`bounded` tp2 320×512×61；`ulysses` tp2 cp2 320×512×61；`gather_kv` tp2 cp2 320×512×61。
- [ ] **Step 3:** 结果写入 `docs/verification/hv15-aot-evidence.md` 的 "Spikes" 一节，并据此决定：
  - bounded 失败 → 先修 Task 3 的 pad 设计，再继续往下做；
  - gather_kv 复现 `NCC_INLA001` → Task 6 只做 ulysses，gather_kv 做成编译前 fail-fast 并记为 BLOCKED，同时写明后续跟进项（更换 toolchain，或改走 masked SDPA 但会付出 O(L²) 的代价）。
- [ ] **Step 4: Commit** `chore(spike): hv15 AOT attention shapes on device`

### Task 1: Registry 元数据 + CLI 布局一致性 + 测试账本

**Files:**
- Modify: `difflet/registry.py`（`_register_builtin_hunyuan_video_15`）
- Modify: `difflet/cli/main.py`（`_replica_cores` / `_replica_tp`：当 tp、cfg、cp 都没指定时，使用 `entry.default_parallel` 的 cfg 设置；router 把解析后的 `--tp-degree` / `--cfg-parallel` 显式转发给 worker）
- Modify: `scripts/verify_cli.py`：
  - `MODELS["hunyuan_video_15"]` 改为 `shape_flags=("--height","320","--width","512","--num-frames","61")`、`generate_flags=("--steps","50")`、`staged=True`。这一步放在这里而不是 Task 9，原因是 Task 10 的 Phase 2 要用到它；
  - hv15 新出现的可运行 cell（`tp2cp2`、`tp2cp2ulysses`、`tp2cfg`、`tp4sp`）暂时放进 `EXPECTED_FAIL_CELLS`，原因写 "pending AOT port"，到 Task 9 再翻转。
- `_replica_cores` 的默认 cfg 规则**只**在走 registry 默认值的分支里生效，也就是平台默认 tp 为 None 的情况；CUDA 的语义保持不变（`tests/unit/cli/test_main_dp.py:115-127` 必须保持绿）。
- Test: 见 Step 1

**Interfaces:**
- Produces：
  - `ModelCapabilities(num_attention_heads=16, is_distilled=False, supports_cp=True, supports_sp=True, cp_modes=frozenset({"ulysses", "gather_kv"}))`。若决策点选 B，再加上 `"ring"`。
  - `default_parallel=DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True)`。

- [ ] **Step 1: 改测试和新增测试**
  - `tests/unit/planner/test_model_capabilities.py:20` 中 hv15 一行改为 `(16, False, True, True)`。
  - `tests/unit/registry/test_hunyuan_video_registration.py:22` 的 `test_hunyuan_video_15_registry_defaults_are_production_480p` 改为断言 `tp_degree == 2` 且 `cfg_parallel_enabled is True`。
  - `tests/unit/cli/test_cli_cfg_parallel.py`：从 `test_cfg_parallel_rejected_for_distilled_model` 的参数化里删掉 hv15，新增 `test_hv15_cfg_parallel_allowed`。
  - `tests/unit/cli/test_cli_sp.py`：从 `test_sp_rejected_for_unsupported_models` 里删掉 hv15。
  - `tests/unit/cli/test_modes.py`：从 `test_distilled_never_gets_cfg2` 里删掉 HYV15，新增断言 `model_class(hv15) == "true_cfg"`，并且 `latency` 模式对应 `ModeConfig(1, True, 1)`。
  - 新增 `tests/unit/cli/test_cli_main_extra.py::test_replica_cores_match_stage_layout`，参数化四种输入：不带参数、只带 `--cfg-parallel`、`--dp 2 --tp-degree 2`、`--mode latency`。断言 `_replica_cores(args) == tp * (2 if cfg else 1) * cp`，其中三者由 hv15 orchestrator 的 `_layout(args)` 计算。
  - `tests/unit/cli/test_verify_cli.py`：`DISTILLED`、`CP_UNSUPPORTED`、`SP_SUPPORTED` 三个集合，`_EXPECTED_SKIPS` 中 hv15 的几行，`test_expected_fail_cells`、`test_plan_cells_counts`、`test_staged_flags`，以及 `test_hunyuan_video_15_compile_cmd_minimal`（改为断言命令里包含 `--height 320`），全部按新值更新。
  - `test_replica_cores_match_stage_layout` 的说明里要写明：它测的是 Trainium、没有平台默认值的那条路径。
- [ ] **Step 2: 运行，确认上述测试失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 运行 `pytest tests/unit -q`，全部通过**
- [ ] **Step 5: Commit** `fix(registry): HunyuanVideo 1.5 runs true CFG and supports CP/SP; keep DP replica cores in step with the stage layout`

### Task 2: AOT 自己的 HV1.5 去噪循环（不依赖 native 代码）

**Files:**
- Create: `difflet/models/hunyuan_video/pipeline15.py`
- Create: `tests/unit/models/hunyuan_video/conftest.py`
- Test: `tests/unit/models/hunyuan_video/test_pipeline15.py`

**Interfaces:**
- Produces（`pipeline15.py` 只 import `torch`、`numpy`、`diffusers`、`difflet.pipeline.teacache`）：
  - `VAE_SPATIAL, VAE_TEMPORAL = 16, 4`
  - `load_hv15_guidance(model_path: str, guidance_scale: float | None = None) -> float | None`：读取 `guider/guider_config.json`。
    - `_class_name` 不是 `ClassifierFreeGuidance` → `NotImplementedError`；
    - `enabled=false` → 返回 `None`；
    - 只要 `guidance_rescale≠0.0`、`use_original_formulation≠False`、`start≠0.0`、`stop≠1.0` 中任一成立 → `NotImplementedError`，报错里列出这些字段；
    - 传入的 `guidance_scale` 覆盖配置值；最终 scale ≤ 1 时返回 `None`。
  - `cfg_on(scale) -> bool`：`scale is not None and float(scale) > 1.0`。
  - `class HunyuanVideo15DiT(typing.Protocol)`：
    - 属性：`cfg: int`、`model_config: dict`（一个普通 dict，至少包含 `in_channels`、`out_channels`、`image_embed_dim`）、`image_tokens: int`；
    - `load(skip_warmup: bool = False) -> None`；
    - `begin_request() -> None`：每个请求开始时清掉逐请求的缓存；
    - `__call__(*, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, encoder_hidden_states_2, encoder_attention_mask_2, image_embeds) -> torch.Tensor`：返回 CPU 上的 noise_pred，batch 与输入相同；
    - `make_teacache_probe() -> TeaCacheProbe`。
  - `class TeaCacheProbe(typing.Protocol)`：`reset() -> None`；`teacache_delta(latent_input, timestep) -> float`，其中 `latent_input` 就是 DiT 本步收到的 `hidden_states`。
  - `class HunyuanVideo15Denoiser`：
    - `__init__(self, dit, *, model_path, height, width, num_frames, dtype=torch.bfloat16, guidance_scale=None, teacache_cadence=None, teacache_online_delta_alpha=None, teacache_speedup=None, teacache_calibration_path=None)`；
    - `load(skip_warmup=False)`；
    - `__call__(*, prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2, negative_prompt_embeds=None, negative_prompt_embeds_mask=None, negative_prompt_embeds_2=None, negative_prompt_embeds_mask_2=None, num_inference_steps=50, generator=None, latents=None, callback=None) -> torch.Tensor`；
    - 可写属性 `guidance_scale`。
- **循环语义**：以 diffusers `HunyuanVideo15Pipeline.__call__` 的 t2v 路径为准，reviewer 拿 diffusers 源码逐条核对。
  - scheduler 按 `model_index.json` 从 `scheduler/` 加载。
  - `sigmas = np.linspace(1.0, 0.0, steps + 1)[:-1]`，再调用 diffusers pipeline 模块里的 `retrieve_timesteps(scheduler, steps, "cpu", sigmas=sigmas)`。
  - 噪声：`randn_tensor((1, out_channels, (F-1)//4+1, H//16, W//16), generator=generator, dtype=dtype)`。
  - 所有 embeds 和 mask 统一转成 `dtype`，与 diffusers 一致。
  - 每步：`latent_input = cat([latents, zeros_like(latents), zeros(1, 1, ...)], dim=1)`，`timestep = t.expand(B).to(latent_input.dtype)`，`image_embeds` 是零张量 `[B, image_tokens, image_embed_dim]`。`scheduler.step` 收到的是未做类型转换的 `t`。
  - CFG：`noise = uncond + s * (cond - uncond)`。`dit.cfg == 2` 时只调用一次，**每个**按样本区分的输入（latent_input、timestep、四个文本张量、image_embeds）都按 `[uncond, cond]` 拼成 batch 2。否则分两次调用，先 cond 再 uncond，每次 batch 1。
  - 每次 `__call__` 开头依次调用 `dit.begin_request()`；若有 controller，再调 `ctrl.reset()`、`sync_probe_free_num_steps(ctrl, len(timesteps))`；若有 probe，再调 `probe.reset()`。
- **TeaCache**：
  - 只要传了 `teacache_speedup` 或 `teacache_calibration_path` 就走自适应模式，此时与 cadence / online-delta 模式互斥，同时给出则抛 `ValueError`。
  - 自适应模式用 `load_teacache_calibration_or_raise(path, model="hunyuan_video_15", shape_label=f"{H}x{W}x{F}")` 加载 calibration。若 `speedup > calibration.target_speedup + 1e-6`，抛 `ValueError`，消息里包含 "target speedup"。
  - probe-free 模式用 `build_probe_free_controller(model="hunyuan_video_15", shape_label=..., cadence=..., online_delta_alpha=...)`。
  - 跳步决策基于 guided prediction；一旦跳过某一步，该步的两次 CFG 前向都不做。自适应模式下，只要 `ctrl.needs_signal()` 为真，就先测 probe 再做决策。
  - 循环结束时打印 `[teacache] stats: {...}`。

- [ ] **Step 1: 写失败测试**
  - conftest 里的 `hv15_snapshot` 用 tiny 模型：
    - `HunyuanVideo15Transformer3DModel(in_channels=9, out_channels=4, num_attention_heads=4, attention_head_dim=8, num_layers=2, num_refiner_layers=1, mlp_ratio=4.0, patch_size=1, patch_size_t=1, qk_norm="rms_norm", text_embed_dim=32, text_embed_2_dim=24, image_embed_dim=16, rope_axes_dim=(2,2,4), rope_theta=256.0, target_size=640, task_type="t2v", use_meanflow=False)`；
    - `AutoencoderKLHunyuanVideo15(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8,16,16,16,16), layers_per_block=1, spatial_compression_ratio=16, temporal_compression_ratio=4, downsample_match_channel=True, upsample_match_channel=True, scaling_factor=0.9)`；
    - `FlowMatchEulerDiscreteScheduler(shift=7.0)`；
    - guider 配置为 CFG 6.0，另加 `model_index.json`。
  - 测试内的 `_DiffusersDiT` 适配器按协议在 CPU 上直接调用 diffusers transformer。
  - `test_loop_matches_diffusers_with_cfg`：denoiser 和 DiT 都用 fp32，32×48×5，3 步，seed 0，正向 prompt 有 6 个有效 token，负向 2 个；与 `diffusers.HunyuanVideo15Pipeline(text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None, transformer=..., vae=..., scheduler=..., guider=ClassifierFreeGuidance(guidance_scale=6.0))(..., output_type="latent")` 的结果比较，`assert_close(atol=1e-4, rtol=1e-4)`。
  - `test_stacked_cfg_matches_two_pass`：`cfg=2` 时每步只调用 1 次，传进去的**所有**张量 dim0 都是 2；结果与两次调用版本 `assert_close(atol=1e-5)`。
  - `test_bf16_dtypes_reach_dit`：默认 bf16 时，DiT 收到的 `timestep` 和 `hidden_states` 是 bf16，返回的 latents 也是 bf16。
  - `test_guider_unsupported_settings_refused`：`guidance_rescale=0.5` → `NotImplementedError`，消息里包含 `guidance_rescale`。
  - `test_guidance_disabled_is_single_pass`
  - `test_teacache_cadence_skips_both_passes`：cadence 2，16 步。断言 `skipped_steps > 0`、`skipped + full == 16`、DiT 调用次数 `== full_steps * 2`。
  - `test_teacache_state_resets_per_call`：连续调用两次，两次的 stats 完全相同；`begin_request` 被调用 2 次。
  - `test_adaptive_refuses_speedup_above_calibration`：先写出 `TeaCacheCalibration(model="hunyuan_video_15", shape_label="32x48x5", num_steps=16, poly_coef=(1.0, 0.0), threshold=0.1, target_speedup=1.5).to_dict()`，再请求 speedup 2.0，预期抛 `ValueError`。
  - `test_pipeline15_does_not_import_native`：用 `ast` 扫描 import，确认没有引入 native 模块。
- [ ] **Step 2: 运行，确认失败**（`ModuleNotFoundError`）
- [ ] **Step 3: 实现 `pipeline15.py`**
- [ ] **Step 4: 运行通过**；`git diff main --stat -- difflet/backends/neuron/` 输出为空
- [ ] **Step 5: Commit** `feat(hv15): AOT denoising loop with true CFG and TeaCache`

### Task 3: AOT DiT 的 TP、文本重排、默认 bounded attention

**Files:**
- Create: `difflet/backends/trainium/hunyuan_video/parallel15.py`
- Modify: `difflet/backends/trainium/hunyuan_video/backbone15.py`（config 默认值、`_hv15_static_reorder_forward`、`_HunyuanVideo15TraceModule`、`ModelWrapperHunyuanVideo15Backbone.input_generator`、`convert_hf_to_neuron_state_dict`）
- Modify: `difflet/models/hunyuan_video/application.py`（`HunyuanVideo15DiTInputBundle` 新增 `text_order` 字段；`validate_hunyuan_video15_dit_inputs`；`create_hunyuan_video15_backbone_config` 加并行参数；1.5 版本的 `_host_refiner_on` 恒为 True）
- Modify: `difflet/backends/cpu/ops_impl/collectives.py`（`gather_from_tensor_model_parallel_region_with_dim` 增加可选参数 `process_group=None`，让 CPU 测试能跑通 AOT 的调用方式）
- Test: `tests/unit/backends/trainium/test_hv15_parallel.py`（新建）、`tests/unit/models/hunyuan_video/test_hunyuan_application.py`（`test_dit_input_contract_15_shapes`、`test_call_15_with_direct_kwargs_builds_bundle` 按新增的输入更新）

**Interfaces:**
- Produces（`parallel15.py` 只 import `difflet.ops`，以及 HV1.0 的 `_sp_unbias`）：
  - `hv15_text_order(mask_image: Tensor, mask_byt5: Tensor, mask_mllm: Tensor) -> LongTensor[B, 729+256+1000]`：**在 host 上运行**。先按 `backbone15` 的段顺序 `[image_3, byt5, mllm]` 把三段 mask 拼起来，再做稳定排序，把有效 token 排在前面（`torch.argsort((~m).to(torch.int8), dim=1, stable=True)`）。`mask_image` 的取值规则与静态 forward 一致（t2v 时全 0），见 `backbone15.py` 约 L630-640。
  - `_hv15_static_reorder_forward` 多接收一个 `text_order` 输入。拼好的文本流和 mask 都用 `torch.gather(..., 1, text_order[..., None].expand(...))` 重排。`text_valid = mask.float().sum(1).to(torch.int32)`。
  - **静态 forward 绑定在实例上**，即 `self.transformer.forward = types.MethodType(_hv15_static_reorder_forward, self.transformer)`。**不要**像现在 `backbone15.py:716` 那样改写 diffusers 类本身的 `forward`；否则同一个 pytest 进程里后面构造的所有 diffusers 模型（包括测试用的参照模型）都会悄悄跑我们的实现。
  - `install_tensor_parallel(model, sp: bool = False) -> None`：每个 rank 的头数取运行时的值 `difflet.ops.get_tensor_model_parallel_size()`（CPU 上是 1，设备上是真实 tp），与 HV1.0 的 `modeling_hunyuan_video.py:344-350` 一致；`num_attention_heads % config.tp_degree == 0` 在 `validate_config` 里检查并 fail fast。**只作用于 `transformer_blocks`**。token refiner 在 host 上运行，不进 NEFF，不切。切分布局：
    - q/k/v 和 add_q/k/v 用列并行，`to_out[0]` 和 `to_add_out` 用行并行；
    - MLP 先列后行；
    - `norm1.linear` 和 `norm1_context.linear` 用列并行，`gather_output=True`；
    - `attn.heads //= get_tensor_model_parallel_size()`。

    线性层用 `difflet.ops.ColumnParallelLinear` / `RowParallelLinear`，参数写法参照 `modeling_hunyuan_video.py` 中 HV1.0 的构造方式（NxD 的签名）；SP 下 bias 被重复累加的问题用 `_sp_unbias` 修正。
  - `bounded_joint_attention(q, k, v, key_valid_len, *, scale) -> Tensor`：输入 `[B,H,S,d]`。在图内用静态 `F.pad` 把 **q、k、v 三者**都补到 128 的倍数；有效区间按 `[0, key_valid_len)` 生成，生成方式只用 expand / reshape，与 `_keypad_bounds` 相同，pad 出来的 query 行区间设为 `[0, 1)`；然后调用 `difflet.ops.attention(..., bound_min=..., bound_max=..., tp_q=True, tp_k=True, tp_out=False)`，最后把 pad 部分切掉。
  - `class JointBoundedProcessor(sp: bool = False)`：做投影、QK norm 和 rotary（与 diffusers 的 `HunyuanVideo15AttnProcessor2_0` 一致），然后调用 `bounded_joint_attention`，`key_valid_len = N + text_valid`；SP 时先对图像流做 all-gather。
  - `_HunyuanVideo15TraceModule.__init__`：只要 cfg、cp、sp 任一开启，就先调用 `init_parallel_mesh(config)`，并在**根上**创建 `self.global_rank = SPMDRank(world_size=get_world_group().size())`；随后设置 processor 和 TP。
  - `convert_hf_to_neuron_state_dict`：**仅当** `getattr(config, "cfg_parallel_enabled" | "context_parallel_enabled" | "sp_enabled", False)` 中有一项为真时，在加 `transformer.` 前缀**之后**写入 `state_dict["global_rank.rank"] = arange(world_size, int32)`（不带前缀）。写法与 HV1.0 的 `backbone.py:259-268` 相同。不加条件会让现有测试 `test_hunyuan_video_15_checkpoint_keys_are_prefixed_for_trace_wrapper` 和 probe 的 key 测试失败。
  - config 新增默认值：`cfg_parallel_enabled=False`、`context_parallel_enabled=False`、`cp_mode="ulysses"`、`sp_enabled=False`。
  - `create_hunyuan_video15_backbone_config(..., cfg_parallel_enabled=False, cp_degree=1, cp_mode="ulysses", sp_enabled=False)`。
  - 删除 `DIFFLET_HUNYUAN15_KEY_MASK_ATTENTION` 和 `DIFFLET_HUNYUAN15_HOST_REFINER` 两个开关，以及 `_HunyuanVideo15KeyMaskAttnProcessor`；其余诊断开关保持不变。

- [ ] **Step 1: 写失败测试**（tiny config，CPU，world 1）
  - `test_text_order_puts_valid_first`：mask 为 `byt5=[1,1,0,0]`、`mllm=[1,0,0]`，image 段全 0。断言结果的前 3 个位置是原来这 3 个有效 token 的位置，而且保持原有先后顺序。
  - `test_trace_module_matches_diffusers_non_prefix_mask`：**整个** trace module（静态 forward + `JointBoundedProcessor` + host refiner）与一个**新构造的** diffusers 模型（同一份权重，不复用 `trace_module.transformer`）对比。输入 mask 在 byt5 和 mllm 两段里都有有效 token，中间夹着 padding。只比较图像流的输出，fp32 下 `allclose(atol=1e-4)`。
    - 不比较 padding 位置上的文本 query 行：diffusers 用对称 mask，那些行是 torch ≥ 2.5 的 safe softmax 置零的；我们的实现在这些行上的值不同，但它们作为 key 是被 mask 掉的。测试依赖写明 `torch>=2.5`。
  - `test_trace_module_does_not_patch_diffusers_class`：构造 trace module 之后，`HunyuanVideo15Transformer3DModel.forward` 仍然是原始实现。
  - `test_bounded_processor_zero_valid_text_is_finite`
  - `test_bounded_processor_padding_is_exact`：在 S = 12225（不是 128 的倍数）时，pad 版本与不 pad 的参考结果在有效行上 `allclose(atol=1e-5)`。
  - `test_tp_install_keeps_parameter_names`：install 前后 `state_dict().keys()` 不变；refiner 的线性层保持为 `nn.Linear`。
  - `test_global_rank_key_matches_module_state`：在 CPU 上，`SPMDRank` 只是一个普通对象；在 NxD 上，它是 shape 为 `[1]` 的 `nn.Parameter`。因此测试里先 monkeypatch `parallel15` / `backbone15` 中的 `SPMDRank`，换成一个持有 `rank` 参数的两行 `nn.Module`，然后断言：`"global_rank.rank" in trace_module.state_dict()`，并且等于 `set(converted) - transformer_keys`。另外断言：cfg、cp、sp 全关时，转换结果里没有这个键。
  - 需要同步更新的现有用例：
    - `tests/unit/registry/test_hunyuan_video_registration.py:598-660` 的 `test_hunyuan_video_15_trace_module_accepts_fixed_tuple_for_t2v`：输入从 8 个变成 9 个（多了 `text_order`）；
    - 同一文件 `:664-680` 的 `test_hunyuan_video_15_checkpoint_keys_are_prefixed_for_trace_wrapper`：确认条件写键后它保持绿。
  - `scripts/hv15_keymask_cpu_check.py` 会 import 已删除的 `_HunyuanVideo15KeyMaskAttnProcessor`，并且改写的是类的 forward：删掉这个脚本，它的用途已由上面的 CPU 测试覆盖。
  - `test_mesh_initialized_when_sharding`：开启 cfg 时会调用 `init_parallel_mesh`（用 monkeypatch 计数）。
  - `test_trace_module_uses_bounded_processor_without_env`
- [ ] **Step 2: 确认失败** → **Step 3: 实现** → **Step 4: `pytest tests/unit/backends/trainium tests/unit/models/hunyuan_video tests/unit/registry -q` 全部通过**；`grep -rn KeyMaskAttnProcessor scripts/ difflet/` 无结果
- [ ] **Step 5: Commit** `feat(trainium/hv15): tensor-parallel DiT, static valid-first text order, padded key-bounded attention`

### Task 4: AOT 的 CFG-parallel

**Files:** `backbone15.py`（forward、`input_generator`）、`difflet/models/hunyuan_video/entry.py`（删掉 L72-77 的拒绝逻辑）、`application.py`（cfg 时 `batch_size=2`）。测试：`test_hunyuan_entry.py`、`test_hv15_parallel.py`

**Interfaces:**
- 参照 Wan 的写法（`difflet/models/wan/modeling_wan.py` L828-850 / L949-952）：
  - `cfg_rank = get_cfg_rank_spmd(self.global_rank.get_rank())`；
  - **9 个**输入（含 `text_order` 和两个 int64 mask）都做 `scatter_to_process_group_spmd(x, partition_dim=0, rank=cfg_rank, process_group=get_cfg_group())`；
  - 输出用 `gather_from_tensor_model_parallel_region_with_dim(out, gather_dim=0, process_group=cfg_group)` 收回。
- CFG 与 CP 同时开启的条件是 `cfg_cp_composable()` 返回真，否则抛 `NotImplementedError`。

- [ ] **Step 1: 测试**
  - `test_create_application15_rejects_cfg_parallel` 改名为 `..._accepts_cfg_parallel`，断言 batch 为 2；
  - `test_cfg_input_generator_traces_batch2`：断言全部 9 个输入的 dim0 都是 2；
  - `test_cfg_with_cp_follows_composable_rule`。
- [ ] **Step 2–4:** 确认失败 → 实现 → 运行通过
- [ ] **Step 5: Commit** `feat(trainium/hv15): CFG-parallel DiT`

### Task 5: AOT 的 SP

**Files:** `parallel15.py`、`backbone15.py`；测试 `test_hv15_parallel.py`

**Interfaces:**
- 开启 SP 时，在根上新增 `self.tp_rank_util = SPMDRank(world_size=tp)`，转换器写入 `state_dict["tp_rank_util.rank"] = arange(tp)`，参照 `difflet/models/wan/checkpoint/backbone.py:76-81`。之所以不能直接用 global_rank：在 SP+CFG 下 world = 2·tp，cfg 分支 1 的那些 rank 拿到的 global rank 是 tp..2tp-1，会切到越界的位置。
- 在 `_hv15_static_reorder_forward` 里、block 循环**之前**只做**一次** scatter：`scatter_to_process_group_spmd(hidden_states, partition_dim=1, rank=self.tp_rank_util.get_rank(), process_group=None)`。block 循环之后做一次 `gather_from_sequence_parallel_region`。不用逐 block 的 hook。
- `JointBoundedProcessor(sp=True)` 先 gather 图像流；图像流的 MLP 做 gather 和 reduce-scatter，bias 用 `_sp_unbias` 修正。
- 编译前检查：`tp > 1`、`cp == 1`、`N % tp == 0`。报错里给出 N 的值和可行的帧数。

- [ ] **Step 1: 测试**
  - `test_sp_rejects_indivisible_tokens_before_compile`：480×848×121 配 tp4 → 报错，消息里含 `49290`；
  - `test_sp_rejects_cp`；
  - `test_sp_with_cfg_writes_tp_rank_util`：monkeypatch 方式与 Task 3 的 `global_rank` 测试相同；
  - `test_sp_world1_matches_dense`（atol 1e-5）。
- [ ] **Step 2–4:** 确认失败 → 实现 → 运行通过
- [ ] **Step 5: Commit** `feat(trainium/hv15): Megatron sequence parallelism`

### Task 6: AOT 的 CP（ulysses；gather_kv 视 Task 0 结果而定；ring 编译前拒绝）

**Files:** `parallel15.py`、`backbone15.py`、`entry.py`（删掉 L70-71）、`application.py`（`world_size = tp * cp * (2 if cfg else 1)`）。测试：`test_hv15_parallel.py`、`test_hunyuan_entry.py`、`test_hunyuan_application.py`（`test_backbone15_world_size_is_process_world` 中 "1.5 has no CP yet" 那段要更新）、`tests/unit/registry/test_hunyuan_video_registration.py:814`（`test_hunyuan_video_15_rejects_cp_until_transformer_port` 改为 `..._accepts_cp`）

**Interfaces:**
- 在 `_hv15_static_reorder_forward` 里、block 循环之前只做**一次** CP scatter：
  - `hidden_states` 沿 dim 1 切；
  - rotary 的 `(cos, sin)` 两个张量沿 N 所在的那一维切，rank 取 `get_cp_rank_spmd(global_rank)`。
  - 循环之后沿 CP 组做一次 all-gather。
- `JointUlyssesProcessor(num_image_tokens)` → `difflet.ops.joint_ulysses_attention(q, eq, k, v, ek, ev, scale=..., key_valid_len=N + text_valid)`。joint 序列长度同样要先补到 128 的倍数，复用 `bounded_joint_attention` 的 pad 规则；如果 op 内部不做 pad，就把 pad 加在调用之前。
- `JointGatherKvProcessor(num_image_tokens)`：**仅当 Task 0 的 gather_kv spike 通过时才实现**。预期它不会通过：`trainium/ops_impl/attention.py:476-479` 已经记录，neuronx-cc 拒绝这种"分片的 q 对全量的 k"的 bounded 形状。所以默认按 BLOCKED 处理，spike 只用来在当前 toolchain 上确认一次。做法是 HV1.0 的路线：all-gather 图像 K/V，分片的 q 对全量的 k 做 bounded attention，并 pad 到 128。spike 失败时，`cp_mode="gather_kv"` 在编译前报错，并引用 spike 的证据。**不要**改走 additive-mask 的稠密 SDPA，那是 O(L²) 内存。
- `cp_mode="ring"`：决策点选 A 时，编译前报错并提示改用 ulysses。
- 编译前检查：ulysses 要求 `(16 // tp) % cp == 0` 且 `N % cp == 0`；所有模式都要求 `N % cp == 0`。

- [ ] **Step 1: 测试**
  - `test_cp_processors_world1_match_bounded`（atol 1e-4）；
  - `test_ulysses_bound_uses_full_image_tokens`：
    - 背景：CP scatter 之后，processor 看到的图像 token 只有 N/cp 个；如果 N 是从张量形状推出来的，bound 会悄悄漏掉一半的图像 key。world 1 时两种写法结果一样，测不出来。
    - 做法：构造 processor 时传 `num_image_tokens = 2 * shard`，monkeypatch 掉 CPU 的 `joint_ulysses_attention` 以抓取 `key_valid_len`，断言它等于 `2*shard + text_valid`；
  - `test_ulysses_rejects_heads_indivisible`（tp4 cp8）；
  - `test_ring_rejected_before_compile`（决策点选 A 时）；
  - `test_cp_scatter_happens_once`：数一下 forward 里 scatter 的调用次数，应为 1 次；
  - `test_create_application15_accepts_context_parallel`。
- [ ] **Step 2–4:** 确认失败 → 实现 → 运行通过
- [ ] **Step 5: Commit** `feat(trainium/hv15): context parallelism (ulysses, gather_kv per spike)`

### Task 6a（可选，仅当决策点选 B）：Trainium `joint_ring_attention` 支持文本 key mask

**Files:** `difflet/backends/trainium/ops_impl/attention.py`（文本 partial 加上 `text_valid_len` 的 bound）、`parallel15.py`（`JointRingProcessor`）。测试：trainium ops 的 CPU 等价测试；设备上先跑一次 spike，并重跑 HV1.0 的 `tp2cp2ring` 回归 cell。

### Task 7: AOT 应用接入循环 + TeaCache probe

**Files:**
- Create: `difflet/models/hunyuan_video/aot15.py`
- Modify: `application.py`：
  - 1.5 不再构造 HV1.0 的 `HunyuanVideoOrchestrator`，改为 `self.pipeline = None`；
  - 新增 `reset_refiner_cache()`；
  - `forward_dit` 在调用前用 `hv15_text_order` 生成 `text_order`。
- Modify: `difflet/backends/trainium/hunyuan_video/teacache_probe15.py` 和 `application.py` 中构造 probe 的那段（约 L561-564）：probe 的 trace module 复用 Task 3 的 TP 安装，与 backbone 共享权重。probe 用 `copy.deepcopy(config)` 再设 `neuron_config.batch_size = 1` 得到的配置来构造（world 和 tp 不变）。原因：`teacache_probe15.py:128` 和 `:152-157` 都直接读 `config.neuron_config.batch_size`，而 Task 4 在 CFG 下会把它设为 2。
- Modify: `scripts/profile_denoise_loop.py`（`_run_hunyuan_video15` 里引用了 `app.pipeline._scheduler_step`，改成使用 `HunyuanVideo15Denoiser` 的 scheduler）
- Test: `tests/unit/models/hunyuan_video/test_aot15.py`

**Interfaces:**
- Consumes: Task 2 的协议；Task 3–6 的 application。
- Produces:
  - `class AotHunyuanVideo15DiT`，构造函数 `__init__(self, app, compiled_model_path: str, *, cfg: int)`：
    - `model_config`：一个**普通 dict**，由 `json.load(transformer/config.json)` 读出（`app.transformer_config` 是一个 hook 闭包，不是配置，不能用它）；
    - `image_tokens = app.image_seq_len`；
    - `begin_request()` 调用 `app.reset_refiner_cache()`；
    - `__call__` 组装 `HunyuanVideo15DiTInputBundle`：两个 mask 用 `.round().to(torch.int64)`，其余张量（含 `timestep_r = zeros(B)`）用 `app.dtype`，然后调 `app.forward_dit`，返回 CPU 上的张量；
    - `load(skip_warmup)` 调用 `app.load(compiled_model_path, skip_warmup=...)`；
    - `make_teacache_probe()`：只在 app 以 `teacache_fused=True` 构造时可用，否则抛 `RuntimeError("adaptive TeaCache needs the teacache_probe component compiled")`。
  - probe 包装对象的 `teacache_delta(latent_input, timestep)`：
    - 只取 `latent_input[:1]`（完整的 in_channels，即 latents + cond + mask，`app.dtype`）和 `timestep[:1]`，补上 `timestep_r = zeros(1)`；
    - 返回 `float(rel_l1)`；
    - `reset()` 把 `prev_mod` 清零。

    只取第一行没有问题：block-0 的调制只依赖 latent 和 t，cond 与 uncond 两行完全相同。
  - `build_aot_denoiser(*, model_path, compiled_model_path, parallel, height, width, num_frames, guidance_scale=None, teacache_cadence=None, teacache_online_delta_alpha=None, teacache_speedup=None, teacache_calibration_path=None) -> HunyuanVideo15Denoiser`：其中 app 构造时传 `teacache_fused=bool(teacache_speedup or teacache_calibration_path)`。

- [ ] **Step 1: 测试**
  - `test_aot_dit_bundle_dtypes`：两个 mask 是 int64，其余是 bf16；并且对拼好的 bundle 跑一遍真实的 `validate_hunyuan_video15_dit_inputs`，输入的 mask 是 bf16，模拟 `encode_prompt` 的实际输出。
  - `test_aot_dit_model_config_is_dict`：在 tiny snapshot 上构造真实的 `NeuronHunyuanVideoApplication`（CPU）。
  - `test_aot_denoiser_cfg2_one_call_per_step`：bundle 里每个张量的 dim0 都是 2。
  - `test_refiner_cache_reset_per_request`：两个请求负向 prompt 相同，正向 prompt 不同但有效长度相同；断言第二个请求重新计算了 refiner。
  - `test_probe_receives_full_channel_batch1_input`
  - `test_probe_example_inputs_are_batch1_under_cfg`：backbone 配置为 batch 2 时，断言 probe 的 `input_generator()[0][0].shape[0] == 1`，且 `prev_mod.shape[0] == 1`。
  - `test_aot_probe_required_for_adaptive`
  - `test_application15_has_no_v10_pipeline`
- [ ] **Step 2–4:** 确认失败 → 实现 → 运行通过；`grep -rn "app.pipeline" scripts/ | grep -i 15` 没有残留
- [ ] **Step 5: Commit** `feat(trainium/hv15): drive the AOT DiT from the AOT loop, with the TeaCache probe`

### Task 7b: HV1.5 TeaCache calibration 采集

**Files:** `scripts/calibrate_teacache.py`（新增 `--model hunyuan_video_15` 分支）；测试 `tests/unit/scripts/test_calibrate_hv15.py`

**Interfaces:**
- 采集器用 `build_aot_denoiser` 加 `make_teacache_probe()`，每步记录 `(probe.teacache_delta, guided noise_pred 的 rel-L1 变化)` 两个量。x 轴信号**必须**就是 probe 输出的 rel-L1，与运行时完全一致；不能沿用 HV1.0 采集器的 L2 范数。
- 输出的 calibration 带 `model="hunyuan_video_15"`、`shape_label="320x512x61"`、`num_steps=50`。
- 输入：encode stage 产出的 `embeds.pt`，格式与 Task 8 相同。用 `callback` 跑完整的去噪轨迹，每一步记录一对数据。不用 `hunyuan15_cache_dit_inputs.py`，它只缓存了第 0 步的数据。

- [ ] **Step 1: 测试**：`test_collector_uses_runtime_probe_signal`（采集器调用的 probe 方法与运行时调用的是同一个）、`test_calibration_model_key`
- [ ] **Step 2–4:** 确认失败 → 实现 → 运行通过
- [ ] **Step 5: Commit** `feat(teacache): HunyuanVideo 1.5 calibration collector on the rel-L1 probe signal`

### Task 8: CLI 的 AOT 分支（encode → aot_transformer → aot_vae）+ DP / JSONL

**Files:**
- Modify: `difflet/cli/orchestrators/hunyuan_video_15.py`
- Modify: `tests/unit/cli/test_orchestrator_hv15_neuron.py`：**只**删掉 `test_aot_backend_is_not_implemented`，这是 Global Constraints 中唯一允许的例外
- Test: `tests/unit/cli/test_cli_hunyuan_video_15_orchestrator.py`。这个文件 L22-42 的 `test_compile_raises_not_implemented_with_model_id_flag` / `test_generate_raises_...` 断言 AOT 会抛 `NotImplementedError`，改写成断言会分发到 `_run_aot`。

**Interfaces:**
- 对 native 只改两处分发：
  - `_run` 开头原本的 `raise` 改成 `if torchrun_backend() != "neuron": return self._run_aot(mode)`；
  - `_run_stage_internal` 新增 `aot_transformer` 和 `aot_vae` 两个分支。
- `_run_aot(mode)`：
  - 用 `_layout` 和 `_stage_cli_args`，只调用不修改；
  - generate 模式先跑 `encode` stage（native 现成的 `_stage_encode`，跑在 host 上）；
  - 然后 `runner.run_stage("aot_transformer", num_cores=tp * (2 if cfg else 1) * cp, virtual_core_size=2)`；
  - 再 `runner.run_stage("aot_vae", num_cores=1, virtual_core_size=2)`；
  - 失败时保留 work dir。
- `_stage_aot_transformer(args)`：在一个进程里完成，由 NxD 驱动所有核，**没有** host 侧的 rank。
  - **compile 模式**：构造 app，`app.compile(dir)`，写 manifest。
  - **generate 模式**：
    1. `build_aot_denoiser` 加 `load(skip_warmup=True)` **只做一次**；
    2. 自适应模式下，如果 probe 还没编译，就 `app.compile(dir, select=["teacache_probe"])` 补编译到同一个目录（HV1.0 的做法）；
    3. 用 `for req in stage_loop.claimed_requests(args)` 遍历请求，每个请求包在 `stage_loop.request_scope(args, req, final=False)` 里；
    4. 每个请求先设置 `denoiser.guidance_scale`，seed 用 `effective(req, args, "seed", 42)`，steps 用 `effective(req, args, "steps", 50)`，然后写出 `work_file(args, req, "latents.pt")`。
  - 如果请求的 `cfg_on(scale) != cfg_on(stage 默认值)`，抛 `ValueError`。在批处理模式下，这个错误会让该请求被 `request_scope` 标记为 failed。
  - **不要**用 `replica_request_scope`，它会 import native。
- `_stage_aot_vae(args)`：用 `vae15` 的 decoder，world 1，编译目录独立。解码写法：`video = app.decode(latents.to(bf16) / scaling_factor, return_dict=False)[0].float().cpu()`（`vae15` 本身既不除 `scaling_factor`，默认返回的也是 `SimpleNamespace`），然后调用 `_aot_write_output`。
- `_aot_write_output(video, out: Path)`：`.pt` 后缀直接保存帧；其他后缀走 `HunyuanVideo15ImageProcessor(vae_scale_factor=16).postprocess_video`，再 `export_to_video(fps=24)`。这里为了不动 native，接受约 10 行的重复代码。
- `_aot_stage_cache_inputs(args, component) -> dict`：
  - transformer 组件的 key 包含 `model`、`revision`、`tp`、`cp`、`cp_mode`、`sp`、`cfg`、`dtype`、`height`、`width`、`num_frames`、`text_seq_len=1000`、`text_seq_len_2=256`、`image_seq_len=729`、`toolchain`；
  - vae 组件的 key 包含 `model`、`dtype`、`shape`、`tile`、`toolchain`；
  - 以下内容**都不进 key**：TeaCache 相关参数（probe 是同一目录里补编译的组件）、steps、seed、prompt。

- [ ] **Step 1: 测试**（mock 掉 `runner.run_stage` 和 application）
  - `test_aot_generate_runs_three_stages`：stage 顺序与核数为 `encode(1) → aot_transformer(4) → aot_vae(1)`。
  - `test_aot_default_layout_is_tp2_cfg2`
  - `test_aot_backend_dispatches_to_run_aot`：替代从 native 测试文件里删掉的那个用例。
  - `test_native_branch_unchanged`：`DIFFLET_BACKEND=neuron` 时 stage 顺序仍为 `encode → transformer → vae`。
  - `test_aot_stage_cache_key_fields`：`cp_mode`、`sp`、`cfg`、`tp` 任一改变，编译目录就变；`--teacache-cadence` 或 `--teacache-speedup` 改变时目录不变。
  - `test_aot_lazy_probe_compile`：自适应模式下 probe 缺失时，会以 `select=["teacache_probe"]` 补编译。
  - `test_aot_request_guidance_toggle_marks_failed`：批处理模式下断言 `claims.is_failed(...)`，并检查 `.failed` 文件里的消息。
  - `test_aot_guidance_one_is_single_pass`：`--tp-degree 2 --guidance-scale 1.0` 能正常跑，且只跑单次前向。
  - `test_aot_requests_load_once`：两个请求时 `load` 只调用 1 次，两个 `latents_req000{0,1}.pt` 都写出。
  - `test_aot_vae_divides_scaling_factor`：app 在 load 之前不能调用（`vae15.py:204-207`），所以测试里给它塞一个桩：`app.models[0].model = HunyuanVideo15VAEDecoderModel(cfg).eval()`，用 tiny VAE 的权重。然后用 `_stage_aot_vae` 里的同一条解码调用，与 diffusers 的 `vae.decode(latents / scaling_factor)` 对比，`allclose`。
  - `test_aot_stages_do_not_import_native`：mock 掉 `_run_stage_internal("aot_transformer")` 和 `("aot_vae")` 后执行，`sys.modules` 里不能出现 `difflet.backends.neuron*`。
- [ ] **Step 2–4:** 确认失败 → 实现 → `pytest tests/unit/cli -q` 全部通过
- [ ] **Step 5: Commit** `feat(cli/hv15): end-to-end AOT generate with DP and JSONL batches`

### Task 9: 元数据与验证矩阵翻转（等 Task 10 中对应的 cell 有了结论再合入）

**Files:**
- `difflet/planner/feasibility.py`：删掉 `KNOWN_BAD[("hunyuan_video_15", None)]`。
- `difflet/planner/model_profile.py`：
  - hv15 的 `text_seq_len` 改为图里实际的 joint 文本长度 1985（1000 + 256 + 729）；
  - 去掉 `unverified`；
  - `WeightFootprint.total_bytes` 用 `component_weight_bytes(<480p snapshot>)` 的结果求和，与其他模型的口径一致，不从 load 日志里取。
- `scripts/verify_cli.py`（`MODELS` 中 hv15 的 shape 等字段已在 Task 1 改好）：
  - `EXPECTED_FAIL_CELLS` 中的 hv15 条目按 Task 10 的结论翻转，结论为 BLOCKED 或 LIMIT 的 cell 保留并写明原因；
  - 决策点选 A 时，把 hv15 加入 ring 不支持的集合。
- `.claude/skills/difflet-device-verify/references/feature-catalog.md`：更新 hv15 行，新增 "Trainium AOT backend: HunyuanVideo 1.5" 一节。
- `README.md`：Trainium 的两张表和 Supported models 表各加一行 HunyuanVideo 1.5。
- 测试：
  - `tests/unit/cli/test_verify_cli.py`：`test_expected_fail_cells`、`_EXPECTED_SKIPS`、`test_plan_cells_counts`；
  - `tests/unit/planner/test_feasibility.py:169`；
  - `tests/unit/planner/test_model_profile.py`：新增对 `text_seq_len` 和 `total_bytes` 的断言。

- [ ] **Step 1–4:** 改测试 → 确认失败 → 实现 → `pytest tests/unit -q` 全部通过
- [ ] **Step 5: Commit**（矩阵翻转单独一个 commit）：`chore(matrix): HunyuanVideo 1.5 on Trainium AOT verified`

### Task 10: On-device 验证 campaign

- 分支：`verify/hv15-aot-<date>`。
- 证据目录：`artifacts/verification-<date>/hv15-aot/`。
- 报告：`docs/verification/hv15-aot-evidence.md`。
- 每完成一行就 checkpoint、commit、push。
- 实例 ID 不写进仓库。
- 所有 skill 脚本都在 `.claude/skills/difflet-device-verify/scripts/` 下，运行时需要设置 `REPO=$PWD`。

**Phase 0 —— 主机准备与参照数据**
- [ ] 执行 `REPO=$PWD bash .claude/skills/difflet-device-verify/scripts/env_check.sh`。当前机器的 repo 下没有 `.venv`，AOT 环境要先用 `./scripts/setup_env.sh` 建好；native 环境用已有的 `/home/ubuntu/venvs/difflet-neuron`（diffusers 0.38.0、torch 2.9.1），只用来生成参照数据。
- [ ] 下载 `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`，并用 `du` 预估磁盘占用。
- [ ] native 参照：TP2×CFG2，320×512×61，seed 42，分别跑 3 步和 50 步；保存 `embeds.pt`、`latents.pt` 及 sha256。AOT 的 cell 用**同一份 `embeds.pt`**。
- [ ] 执行 Task 0 的 spike（如果之前还没跑）。

**Phase 1 —— 正确性门槛**（3 步，320×512×61，seed 42，使用同一份 embeds）

| Cell | 内容 | 通过标准 |
|---|---|---|
| aot-tp2 | TP2，CFG 两遍串行 | 与 native TP2×CFG2 的 latents 余弦 ≥ 0.999，且没有 NaN / Inf |
| aot-tp2cfg | TP2×CFG2（默认布局） | 与 aot-tp2 余弦 ≥ 0.999 |
| aot-tp4 | TP4 | 与 aot-tp2 余弦 ≥ 0.999 |
| aot-tp4sp | TP4 + SP | 与 aot-tp4 余弦 ≥ 0.999 |
| aot-tp2cp2-ulysses | TP2 × CP2 ulysses | 与 aot-tp2 余弦 ≥ 0.999 |
| aot-tp2cp2-gather_kv | 仅当 Task 0 spike 通过时运行 | 与 aot-tp2 余弦 ≥ 0.999；否则记为 BLOCKED |
| ring | 决策点 A：记为 N/A，并验证编译前报错在几秒内出现；决策点 B：同上面的门槛 | — |
| aot-vae | `vae15` 的 tiled decode（tile 大小 16 latent，blend 25%），9 帧，320×512；与 diffusers 的 CPU decode 对比（`scripts/hunyuan15_vae_parity.py`） | PSNR ≥ 40 dB，无接缝；记录最大绝对误差。最终交付物是 MP4，这一行专门防止 tile 拼接的 bug 混过 latents 的检查 |

注：aot-tp2（串行 CFG）每步要跑两次 host refiner（cond 和 uncond 各一次），所以会比 TP2×CFG2 慢，这是预期内的；在证据文档里注明。

每个 cell 都要记录 rel-L1、余弦、最大绝对误差。另外在 9 帧上与 diffusers 的 CPU bf16 参照比较距离，方法同 `neuron-backend-results.md`。

**Phase 2 —— 并行矩阵**（50 步）
- [ ] 执行 `REPO=$PWD bash .claude/skills/difflet-device-verify/scripts/run_cells.sh "hunyuan_video_15" "tp4 tp2cp2 tp2cp2ulysses tp2cfg tp4sp dp2tp2"`。`verify_cli` 里**没有** `tp2cp2ring` 这个配置，`tp2cp2` 对应的是 gather_kv。
- [ ] 每个 cell 记录：compile 秒数、generate 秒数（staged 的墙钟时间）、单步耗时中位数、每核峰值 HBM（neuron-monitor），以及 MP4 和中间帧 PNG。
- [ ] DP 逐比特一致性检查：`PYTHONPATH=$PWD python scripts/verify_dp_correctness.py --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --dp 2 --tp-degree 2 --steps 3 -- --height 320 --width 512 --num-frames 61`。脚本固定发 4 个请求，每个请求的 latents 都要与单副本运行逐比特一致。如果和 HV1.0 一样在加载 VAE 时 HBM 不够，就记为 LIMIT 并写明原因。

**Phase 3 —— 默认 shape**
- [ ] TP2×CFG2，480×848×121，50 步：记录端到端时间、单步时间，并输出 MP4。与 native 同布局、同一台机器的重测结果对比（`neuron-backend-results.md` 里 native 的单步时间是 12.6 s，仅作参考）。
- [ ] tp4 SP（以及决策点选 A 时的 ring）在默认 shape 下必须在**编译前、几秒之内**报错；记录报错文本。

**Phase 4 —— TeaCache**
- [ ] 固定步长，TP4 串行 CFG 版本：`REPO=$PWD SHAPE_FLAGS="--height 320 --width 512 --num-frames 61" bash .claude/skills/difflet-device-verify/scripts/teacache_cadence_ab.sh hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v 50 mp4`。
- [ ] 固定步长，TP2×CFG2 版本：手动执行两条 `difflet generate --tp-degree 2 --cfg-parallel` 命令，一条不带 TeaCache，一条带 `--teacache-cadence 2`。日志中必须出现 `skipped_steps 20`；latents 余弦 ≥ 0.99。这一项证明 CFG 的两遍前向会被一起跳过。
- [ ] online-delta：`--teacache-online-delta <alpha>`，记录跳步数和余弦。
- [ ] 自适应：用 Task 7b 的采集器生成 `hunyuan_video_15 / 320x512x61 / 50` 的 calibration，再跑 `--teacache-speedup`。报告里写出信号的拟合度（Pearson / R²）。HV1.5 的信号预计偏弱，控制器几乎不跳步也是有效结论，不算失败。

**Phase 5 —— JSONL 批处理**
- [ ] 用 `--requests` 发两个 prompt（一个带负向 prompt，一个自定义 seed）：输出两个 MP4，每个都与单独运行时的 latents 逐比特一致；日志显示 DiT 只加载了一次。

**Phase 6 —— 报告**
- [ ] 证据文档：每个 cell 记录结果、耗时、证据路径和复现命令，另附 bug ledger。
- [ ] 支持矩阵（PASS / LIMIT / BLOCKED / N/A / NOT MEASURED），给出"模型 × 功能"表和"按功能分块"两种形式。
- [ ] 回到 Task 9，按结论翻转矩阵。
