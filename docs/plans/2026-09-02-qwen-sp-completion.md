# 2026-09-02 Qwen-Image SP 收尾 session

> 目标：把 9a00746 落地的 Qwen-Image Megatron 式序列并行（双流纯 SP）做完 ——
> 设备 parity 门禁、单测补齐、stale 断言修复、tp4/tp4sp 四阶段测量 —— 并把
> 这台 AL2023 trn2.3xlarge 原生工具链闭合记录在案（8/27 被迫用 Ubuntu 容器
> 的 glibc 问题本次在宿主机上直接解决）。
>
> 分支：`session/parallel-phase-sweep-20260826` → 本次工作推送至
> `session/qwen-sp-completion-20260902`。

## 1. 环境：AL2023 原生 Neuron 工具链闭合（本次最大的基础设施产出）

8/27 的 sweep 报告记录了 "AL2023 glibc 2.34 不兼容 Neuron 栈 → 特权容器方案"。
本次在 AL2023 宿主机上原生解决了全部三个障碍，**不再需要容器**：

| 组件 | 版本 | 备注 |
|---|---|---|
| OS / 内核 | AL2023, 6.18.44-99.149 | 裸 AL2023 AMI（非 DLAMI） |
| 驱动 | aws-neuronx-dkms 2.30.2 | 对 6.18 内核 DKMS 编译一次通过 |
| runtime | aws-neuronx-runtime-lib **2.34.10** + collectives **2.34.10** | 必须用 2.34 线，见障碍 3 |
| 工具 | aws-neuronx-tools 2.32.28 | neuron-ls 正常报 4 logical / LNC=2 |
| venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`（Python 3.12） | 与参考镜像同名同位 |
| pip 栈 | torch 2.9.1 / torch-neuronx 2.9.0.2.15.32035 / torch-xla 2.9.0 / libneuronxla 2.2.17544 / neuronx-cc 2.27.5334 + **islpy==2026.1** / nki 0.6.0 / nxd 0.19.28492 / nxdi 0.10.18399 / transformers 4.57.6 / diffusers 0.38.0 | Neuron pip index |

三个障碍与修法：

1. **`_XLAC.so` 需要 GLIBC_2.35**（AL2023 只有 2.34）。全 venv 扫描后确认缺失符号
   仅 `hypot/hypotf` 两枚（verneed 记录挂在 libm.so.6 上，LD_PRELOAD 骗不过
   verneed 校验）。修法：直接把 `_XLAC.so` 的 verneed vernaux 从 `GLIBC_2.35`
   改指 `GLIBC_2.2.5`（系统 libm 实际提供的版本，`hypot@2.2.5` 真实存在），
   原文件备份为 `_XLAC.cpython-312-...so.orig-glibc235`。torch-xla 导入与设备
   计算自此全部正常。
2. **neuronx-cc 内部错误 `NCC_ISMP902 is_subset()`**：neuronx-cc 2.27 的
   `islpy~=2026.1` 约束下 pip 装了 2026.2.1（新版 islpy 改了 API 签名），显式
   降级 `islpy==2026.1` 解决。另注意：`/var/tmp/neuron-compile-cache` 会**缓存
   编译失败结果并原样回放**（错误时间戳都不变），排障时先清缓存再用
   `NEURON_CC_CACHE=0` 复跑。
3. **驱动只接受整设备或单核分配，且必须配 runtime-lib ≥2.34**：本机 dkms 的
   规则是 `NEURON_RT_NUM_CORES ∈ {1, 整设备}`（内核报 "must request one core,
   or the whole device"）。关键坑：runtime-lib 2.30.51 把 `NUM_CORES=4 +
   VIRTUAL_CORE_SIZE=2` 换算成"部分多核请求"被驱动拒绝（NRT/TorchScript 权重
   加载路径必挂）；**runtime-lib 2.34.10 的换算正确**（4 逻辑核 = 整设备），
   tp4 全链路（加载/前向/staged stage）实测通过。XLA/libneuronxla 路径不受
   runtime 版本影响（两种 runtime 下裸 XLA 计算都正常）。结论：runtime 2.34
   不再需要"配套驱动重启"——dkms 2.30.2 即可。影响面：tp2 等不满核配置仍
   不可用（驱动规则），但 qwen sweep 的 tp4/tp4sp 与 VAE（1 核）全部合法。

## 2. 代码与测试补齐（qwen SP 的"实现已落地、验证与配套未跟上"部分）

- `scripts/qwen_sp_parity_smoke.{py,sh}`（新增）：Qwen 密集 vs SP 设备 parity
  门禁，对齐 wan/flux/hunyuan 的模式；默认 tp=4（整设备，见 §1.3）。
- `tests/unit/models/qwen_image/test_qwen_sp.py`（新增，10 例）：fork 与 diffusers
  父类 state_dict 键位全等（权重加载契约）、tp=1 下 SP-on 与 dense 逐位相等、
  `_sp_unbias`（无偏置/tp=1 恒等 + tp=2 补偿量）、`_shard_sequence` 不整除拒绝、
  `zero_cond_t`+SP 拒绝、registry `supports_sp` 钉死。
- **修复 SP 加载缺权重键（本次在设备上首次跑通 tp4sp 时暴露的真 bug）**：
  SP 入口 scatter 依赖的 `SPMDRank` 缓冲（`sp_rank_util.rank`）在
  `convert_hf_to_neuron_state_dict` 中没有按 nxd 配方物化
  （`ckpt['spmd_rank.rank'] = torch.arange(0, tp, int32)`），加载报
  "Missing weight tensor with key sp_rank_util.rank"；CP 路径早有同款处理
  （`global_rank.rank`），SP 漏了。旧机器从未编译/加载过 tp4sp，所以此前不可见。
- **修复 sweep step worker**（`scripts/parallel_phase_sweep.py` `worker_qwen`）：
  原实现抄 flux 的 `DiffletPipeline.from_pretrained`（哈希缓存路径），而 qwen 是
  staged 模型 —— 产物在 `qwen_image_dit_tp4...` 命名目录下，两者永远对不上
  （旧 tp4.json `errors.generate` 的 "transformer/model.pt does not exist"）。
  改为 wan worker 模式：进程内驱动 CLI 自己的 text+generate stage，包裹
  `NeuronQwenImageTransformerApplication.__call__` 计逐步间隔。
- **修复整设备 stage 显式 `NEURON_RT_NUM_CORES` 被驱拒**（`difflet/cli/runner.py`
  与 sweep `_worker_env`）：整设备 stage 不再显式设该变量（见 §1 障碍 3），
  判定基于继承的可见核集合（`resolve_available_neuron_core_ids`），不满核时
  行为不变。
- **修复 `--prune` 死代码**（8/26 session §4.1 的既定方案）：`_phases_for` 返回
  真实 JSON 键（`e2e_cold`/`e2e_warm`），完成即 `state["done"]`。
- **修复 benchmark harness 并行轴字段**（planner-benchmark-collection 文档的
  "采集前必须修"）：`BenchConfig` 增 `sp`/`cfg_parallel`/`cp_mode` 一等字段 +
  `parallel_flags()`/`parallel_dict()`；trainium adapter、bench.py、
  step_latency/step_realloop、report.py 全链路透传 —— tp4sp 行不再被误记成 tp4。
- **修复 serving 编译身份指纹漏 sp**：`difflet/common/orchestrators/qwen_image.py`
  `_compile_identity` 补 `sp_enabled`（仅 generate 阶段），否则两个只差 `--sp`
  的 serving profile 会共享身份指纹但编译目录不同（hunyuan 早已带此字段）。
- **stale 修正**（9a00746 改了 registry 但下游没跟上，8 个预存失败测试）：
  `test_cli_sp.py`（Qwen 从拒绝列表移入支持列表 + staged 目录/参数转发参数化）、
  `test_model_capabilities.py`（EXPECTED 矩阵 + `_validate_sp` 用例）、
  `test_feasibility.py`（无 SP 模型示例换 ltx_2）、`test_model_registry.py`
  （qwen serving SP 改为接受断言）、`test_verify_cli.py`（SP_SUPPORTED 集合 /
  skip 表 / plan 计数 11→10 skip）、`test_pipeline.py`（teacache 缓存键契约为
  `teacache_probe_enabled`）；CLI 错误提示语补 Qwen；README/DEVELOPER/
  planner 矩阵/verify-matrix spec/megatron plan 文档同步。

## 3. 验证结果

| 门禁 | 结果 |
|---|---|
| 单元测试（CPU，全量） | **2166 passed / 32 skipped / 0 failed**（含新增 10 例 qwen SP + runner 整设备契约 2 例） |
| 设备 parity（dense vs SP, tp=4, 2 层 256², trn2.3xlarge） | **cosine 0.999991 ✅**（门槛 ≥0.999；max_abs 0.0625 / mean_abs 0.0093 / rmse 0.0133，bf16 精度量级） |
| 端到端 tp4/tp4sp 四阶段 sweep | 本机全量重测（公平性协议：同机同工具链冷编译双配置），见 `artifacts/parallel_phase_sweep/qwen/` |

## 4. 测量（本机重测，旧 8/28 数据隔离为交叉参考）

旧 `tp4.json`（它机、含 stale 错误记录）移至
`artifacts/parallel_phase_sweep/qwen/crossref-0828/`。本机（ip-172-31-38-211，
AL2023 原生 + §1 工具链）tp4 与 tp4sp 均从冷编译起测，四阶段齐测。

<!-- SWEEP_RESULTS -->

## 5. 后续（不阻塞本次）

- planner 测量播种桥（`seed_planner_measurements.py` 目前只吃
  `benchmark/<device>/*.json`，sweep 产物在 `artifacts/` 下，wan/qwen SP 行
  仍未入 `difflet/planner/data/measurements.json`）。
- 本机驱动不接受部分多核分配 —— tp2 类不满核配置的补测需要换驱动/换机。
- hunyuan tp4sp 仍未测量（planner-benchmark-collection 的第二配置表）。
