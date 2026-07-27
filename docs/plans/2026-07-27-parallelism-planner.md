# Difflet 自动并行策略 Planner — 设计与实施计划

Date: 2026-07-27

Status: 设计已确认，P0–P3 待实施。

参考：`difflet/pipeline/parallel_config.py`、`difflet/pipeline/parallel_mesh.py`、
[DEVELOPER.md](../../DEVELOPER.md) 的 Parallelism 与 Runtime protocol 两节、
`scripts/verify_cli.py` 的并行矩阵，以及 `benchmark/trn2/*.json` 的实测数据。

## Context

Difflet 目前有 5 类并行策略（TP / CP / CFG / SP / DP），组合规则散落在 6 个地方：
`DiffletParallelConfig.__post_init__` 的硬校验、`cli/main.py` 的模型能力集合、
`cli/modes.py` 的静态预设表、`serving/options.py` 的 serving 限制、各模型
`entry.py` 的 guard、以及 `scripts/verify_cli.py` 的 skip 集合。用户必须自己算
`world_size = dp × cfg × cp × tp`、自己记住哪些组合互斥、自己知道哪个模型支持
哪个特性。

更糟的是**硬件是假的**：`difflet/common/neuron_cores.py:8` 把
`DEFAULT_NEURON_CORE_IDS = (0,1,2,3)` 写死，`cli/modes.py` 的预设表也硬编码
4 核（`dp=4` / `cp=4`）。在非 `trn2.3xlarge` 的机器上这些都是错的，而
`_validate_dp` 在既没有 `--total-cores` 也没有 `NEURON_RT_NUM_CORES` 时**完全
不做容量检查**。

目标：新增一个 planner，自动探测当前硬件 + 目标模型 + 目标 shape，输出一个
**可行且接近最优**的组合并行配置，并能解释为什么。

**已确认的范围决定**：
- planner **只规划并行度**（tp / cp / cp_mode / cfg / sp / dp）。TeaCache 等有损
  加速开关只作为排斥约束参与，不由 planner 建议档位。
- **第一版只支持单 Neuron device**（trn2.3xlarge 这类），**第二版扩展到多设备**
  （trn2.48xlarge，16 device / 64 核）。多设备的设计接口在第一版就预留好，见 Part 6。
- 本次实施做到 **P3**（成本模型 + 测量库，`difflet plan` 出可排序的建议），
  P4/P5 留作后续。

---

## Part 1 — 现有并行策略清单

### 统一抽象

`difflet/pipeline/parallel_config.py:15` 的 `DiffletParallelConfig` 是唯一入口，
`difflet/pipeline/parallel_mesh.py:29` 的 `MeshSpec(dp, cfg, cp, tp)` 是设备网格：

```
world_size = dp × cfg × cp × tp        # SP 不出现在这个乘积里
rank       = tp + T·(cp + C·(cfg + G·dp))     # tp 最内、dp 最外
```

### 五个轴

| 策略 | 切什么 | 配置 | 是否吃 world_size |
|---|---|---|---|
| **TP** 张量并行 | 层内权重/激活、attention head | `tp_degree` | ✅ ×tp |
| **CP** 上下文并行 | latent token 序列（`cp` 轴） | `cp_degree` + `cp_mode` | ✅ ×cp |
| **CFG** 并行 | uncond/cond 两条 CFG 分支（`cfg` 轴，固定为 2） | `cfg_parallel_enabled` | ✅ ×2 |
| **SP** Megatron 序列并行 | norm/modulation/residual 沿序列切，**复用 TP 组** | `sp_enabled` | ❌ 不变 |
| **DP** 数据并行 | 整副本，进程级路由 | `dp_degree` / `--dp N` | ✅ ×dp |

**CP 有三种 attention 策略**（`CP_MODES`）：
- `gather_kv`（默认）— all-gather 全量 K,V；通信 `O(S·H_local·d)`，简单、精确。
- `ring` — K,V 分片在环上轮转 + online softmax 合并；依赖实验性
  `nkilib.experimental.attention.ring_attention_fwd`，**TRN1 上会静默回退到
  gather_kv**（`modeling_flux.py:1377`），且不支持 `attention_mask` / `causal=True`。
- `ulysses` — 两次 all-to-all 把「序列分片」换成「head 分片」，中间是一次普通稠密
  attention。通信量比 gather_kv 少一个 `cp` 因子，不需要特殊 kernel，数学上与
  gather_kv 等价。代价是需要 `num_attention_heads % (tp × cp) == 0`。

**DP 有两个完全不同的东西**（容易混淆，planner 必须区分）：
- `dp_degree` — mesh 里预留的轴，`tests/unit/test_no_dp_parasites.py` 静态强制它
  **没有任何 per-layer/per-step 消费者**；目前是空壳。
- `--dp N` — 已落地的进程级副本路由（`difflet/cli/dp/`），router 给每个 worker 切
  一段 `NEURON_RT_VISIBLE_CORES`，每个 worker 是一个 `dp=1` 的完整子进程。

### 不是并行的东西（planner 不要碰）
- `pp_degree` / `ep_degree` 只存在于 NxDI 的 fork（`backends/trainium/core/config.py`），
  diffusion 路径恒为 1，`MeshSpec` 里根本没有这两个轴。
- Wan/Qwen/Hunyuan 的多 stage 子进程是**顺序**的组件阶段，不是流水并行。
- `CandidateConfig` 的 `N` 轴是 batch-like，明确不改 `world_size`、不引入 collective。
- Wan 2.2 双 transformer 是 timestep 边界上的 MoE 切换，不是专家并行。

---

## Part 2 — 互斥 / 不兼容矩阵

### A. 配置级硬互斥（`DiffletParallelConfig.__post_init__` 直接 `ValueError`）

| 组合 | 结论 | 原因 |
|---|---|---|
| `cp_degree > 1` + `cfg_parallel` | ❌ **互斥** | 两者都占用 data-parallel lane |
| `sp_enabled` + `cp_degree > 1` | ❌ **互斥** | SP 沿 TP 组切序列，CP 沿 DP 组切序列 |
| `cp_mode != gather_kv` + `cp_degree ≤ 1` | ❌ 非法 | ring/ulysses 必须 cp>1 |
| `sp_enabled` + `cfg_parallel` | ✅ 允许 | world = tp×2；但 CLI 层没有模型同时满足两边（SP 集合 ∩ true-CFG = 只有 Wan） |
| `dp` + 任意 | ✅ 允许 | 纯乘 world_size |
| TP 与其余各轴 | ✅ 正交 | — |

### B. 模型 × 策略支持矩阵

| 模型 | TP | CP gather | CP ring | CP ulysses | SP | CFG-par |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FLUX.1-dev | ✅ | ✅ | ✅¹ | ✅ | ✅ | ❌ 蒸馏 |
| Qwen-Image | ✅ | ✅ | ✅ | ✅ | ❌² | ❌ 蒸馏 |
| Wan 2.2 / 2.1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HunyuanVideo | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 蒸馏 |
| HunyuanVideo 1.5 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ 蒸馏 |
| LTX-2 | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |

¹ TRN1 上静默回退 gather_kv。² Qwen 的 forward 是 monkey-patch 上游 diffusers 的，
`SPMDRank` 不是活图输入，每个 rank 都读成 rank 0。

### C. 数值/整除约束

- `num_attention_heads % tp_degree == 0`（Flux/Hunyuan/Qwen = 24，Wan = 40）
- **`num_attention_heads % (tp × cp) == 0` — ulysses 专属。目前只有 attention 层的
  `_ulysses_check_heads` 在跑时才报，CLI/config 层完全没有校验**
- `world_size` 必须能被 `tp × cfg × dp` 整除（CP degree 是反推出来的）
- `scatter_tp_dim` 要求被切维度能被 tp 整除

### D. 跨特性（非并行）排斥

- Flux TeaCache ⊗ CFG-parallel（跨 CFG rank 的单次 skip 决策不成立）
- 任意 TeaCache flag ⊗ `--dp` / `--requests`
- LTX-2 CFG-parallel ⊗ STG / modality guidance；LTX-2 TP>1 ⊗ `perturbed_attn`
- ring / ulysses ⊗ `attention_mask`；ulysses ⊗ `causal=True`（注释指出是**静默错误**）
- HV-1.5 分段运行时 ⊗ TeaCache

### E. Serving 额外收紧（`difflet serve`）

CFG-parallel **完全禁止**；SP 仅 flux/wan/hunyuan_video；Qwen 强制 cp=1；
Wan/Hunyuan/LTX-2 的常驻 adapter 硬钉死 `tp=4, cp=1, dp=1, cfg=off`。

### F. 已知坏格（编译器 bug，不是配置问题）

`scripts/verify_cli.py:186` 的 XFAIL：`(hunyuan_video, tp2cp2)` 触发 neuronx-cc
`NCC_INLA001` / `NCC_IBIR243`；`(hunyuan_video, dp2tp2)` 是单副本 HBM 放不下。

---

## Part 3 — 业界对标：这块基本是空白

调研了 xDiT/xfuser、HF Diffusers、SGLang Diffusion、vLLM-Omni、TensorRT-LLM
VisualGen、DistriFusion、ParaAttention、FastVideo、LightX2V、raylight/ComfyUI、
OneDiff、AWS NxDI、DeepSpeed。**没有任何一个生产框架有 Alpa/Galvatron 意义上的自动并行
planner**（成本模型 + 联合策略空间搜索 + 硬件拓扑感知）。全部要求用户手动指定各
并行度，唯一约束就是 `product(degrees) == num_gpus`。

| 框架 | 策略 | 配置方式 | 自动规划 |
|---|---|---|---|
| xDiT / xfuser | USP(ulysses+ring)、PipeFusion、CFG、TP、DP、DistVAE | 手动指定各 degree | **无**（只有 `dit_parallel_size = world_size`） |
| HF Diffusers | Ring / Ulysses / Unified、Ulysses-Anything | `ContextParallelConfig(ring_degree, ulysses_degree)` | **无**；`device_map="auto"` 是容量放置不是并行 |
| SGLang Diffusion | SP(ulysses+ring)、TP、CFG、FSDP、分布式 VAE | `--sp-degree` 等 + `--performance-mode auto` | **部分** — auto 只根据剩余显存决定 FSDP 开关、可能打开 CFG 并行；各 degree 仍手动 |
| vLLM-Omni | TP/PP/DP/EP、CFG | engine args | **无**，roadmap 里也没有 |
| TensorRT-LLM VisualGen | CFG、Ulysses(+async A2A)、Ring、CP、Attn2D mesh、TP、parallel VAE | `VisualGenArgs.parallel_config` | **无** |
| ParaAttention | Ulysses / Ring / Unified | 用户自己 `init_device_mesh` | **无**（README 里是写死的 2×N/2 启发式） |
| FastVideo / LightX2V | SP(ulysses/ring)、CFG | `num_gpus` / JSON 里的 `seq_p_size` | **无** |
| AWS NxDI | TP、CP、CFG（**CP ⊕ CFG 互斥**，与 Difflet 一致） | `backbone_tp_degree` + 模式开关 | **部分** — `get_flux_parallelism_config()` 只是算术推导 world_size |
| DeepSpeed | TP(AutoTP)、PP、ZeRO-DP、Ulysses SP、Domino | ds_config 手填各 degree | **无** — Autotuner 只搜 ZeRO stage / micro-batch，并行度不在搜索空间；diffusion 支持已废弃 |

大家的替代方案是**「文档即 planner」**：xDiT 在 README 里列「2 卡用 cfg=2、4 卡用
cfg=2×pipefusion=2」，Diffusers 给 4×H100 的 benchmark 表让你自己看，SGLang 的文档
干脆写「presets are intentionally coarse… 不要指望自动调优细粒度参数，请自己
benchmark」。

### DeepSpeed：两个带 "auto" 的功能，但都不是选并行度

DeepSpeed 值得单独说，因为它是最容易被误认为「已经有 planner」的项目。实际上：

| 功能 | 自动的是什么 | 手动的是什么 |
|---|---|---|
| **Autotuner**（`"autotuning": {...}`） | ZeRO stage、micro-batch size、ZeRO bucket 参数 | **所有并行度（TP/PP/SP）完全不在搜索空间里** |
| **AutoTP** | **切分策略**（哪些层按行/按列切，不用手写 injection policy） | `tp_size` / `autotp_size` —— **度数本身** |
| **Ulysses** | 无 | `sequence_parallel_size`、chunk size |
| **MII / FastGen** | kernel 与优化选择 | `tensor_parallel`（默认取 `WORLD_SIZE`） |

关键细节：Autotuner 的配置里**确实有 `mp_size`**，但它是**输入常量而非搜索维度** ——
只用来算 `dp_size = num_gpus / mp_size` 好做显存估算。想比较 tp=2 与 tp=4，得手动跑
两遍 autotuner 再自己比。而且 autotuner 是**训练专用**、靠**真机试跑**（官方例子：
16×V100 上 13 次实验跑 27 分钟），推理侧完全没有对应物。

AutoTP 的 "automatic" 是指自动推导切分策略（从 HF 的 `base_model_tp_plan` 或
正则 `partition_config`），degree 仍来自 `WORLD_SIZE`。MII 文档里那句
"Based on the model architecture, model size, batch size, and available hardware
resources, MII automatically applies the appropriate set of system optimizations"
指的是 kernel 选择，不是并行度 —— `mii/config.py` 里没有任何代码去看模型参数量
或显存来决定 `tensor_parallel`。

另外两点与 Difflet 直接相关：
- **DeepSpeed 的约束和 Difflet 高度同构，且同样没有强制**：TP | heads、TP | hidden、
  SP | heads、SP | seq_len，AutoTP-training ⊗ ZeRO-3 互斥，
  Ulysses ⊗ Megatron TP/PP 互斥。这些全部散落在各个 tutorial 里写成「用户义务」，
  **没有任何兼容性矩阵**，只能靠 init 时的 assert 撞出来。Difflet 至少已经把互斥
  规则写进了 `DiffletParallelConfig.__post_init__`，比 DeepSpeed 强。
- **DeepSpeed 的 diffusion 支持实质上已废弃**：`module_inject/containers/vae.py`
  最后一次功能性改动是 2024-02，现行 MII（v0.2+）**彻底移除了 Stable Diffusion**，
  只剩 10 个文本生成架构。没有 SDXL / SD3 / Flux / 任何 DiT 的支持痕迹。
- DeepSpeed 本身仍然活跃（v0.19.3，2026-07-23；2025-01 已捐给 Linux Foundation，
  org 变成 `deepspeedai/DeepSpeed`），但 2025–2026 的投入全在显存 offload、
  optimizer 和编译器上，**autotuner 的范围四年没扩过**。

Galvatron 的论文直接把 DeepSpeed 当作「手动调参」的对照基线，宣称比
"Megatron and DeepSpeed frameworks that employ manual tuning" 高 1.26–1.47× 吞吐。
一句话概括差别：**DeepSpeed 的 autotuner 调的是一份配置，Alpa/Galvatron 规划的是
一种并行化方式**；DeepSpeed 用实测的地方它们用预测，而 DeepSpeed 的搜索空间恰好
排除了构成「并行方案」的那几个轴。

**研究界刚刚开始做**（都是 2026 年的，且都没放代码）：
- **AoiZora**（arXiv 2606.17566，2026-06）— 第一个真正的 DiT 推理自动并行 planner。
  两阶段：先用编译前 IR 剪掉弱候选，再只编译幸存者、用编译后 HLO + 拓扑感知通信
  模型排序物理放置。Wan 2.1 单步降噪 1.42×。**但只做了 TPU（v5e）**，规划耗时
  47–108s（v5e-4）到 250–376s（v5e-16），离线缓存。
- **GF-DiT**（arXiv 2606.13501，2026-07）— 服务侧动态调度，按 (模型, 任务类型,
  请求 shape, 并行配置) 索引的 profiling 成本模型，在线调整 SP degree。点名批评
  vLLM-Omni 和 SGLang Diffusion 的并行配置一旦分配就固定不变。
- SwiftFusion / CoCoDiff / db-SP / DDiT / TridentServe — 固定策略优化或服务调度，
  不是 planner。

**对 Difflet 的意义**：这是个真实的空白，而且 Difflet 有两个别人没有的有利条件 ——
（1）搜索空间小到可以穷举（4 核上个位数），不需要 AoiZora 那套两阶段剪枝；
（2）AOT 编译意味着每个配置的成本稳定可测，不像 GPU 那样 JIT 后还受运行时波动
影响。反过来最大的不利条件也来自 AOT：换配置 = 全量重编译，所以不能靠在线搜索。

两个可直接借鉴的点：
- **head 整除是所有人都撞的硬约束**（`ulysses_degree ≤ num_heads`），Diffusers 的
  "Ulysses Anything" 就是为绕开它而生。planner 必须把整除约束当一等公民 —— 这正是
  Difflet 目前唯一没在 Python 层强制的约束。
- HF 的 benchmark 显示**只要 ulysses 可行，ulysses > unified > ring**（吞吐）。
  可作为 cost model 里 CP mode 排序的先验，与 Ulysses 设计文档「通信量少一个 cp
  因子」的分析一致。

---

## Part 4 — Planner 设计

### 核心判断：搜索空间极小，穷举即可

`world_size = dp·cfg·cp·tp` 必须整除可用核数。4 核上合法组合只有个位数，
64 核也只有几十个。**不需要 Alpa/Galvatron 那套 ILP/动态规划**，
「枚举 → 过滤 → 打分」就够，而且完全可解释。

### 真正的难点是成本模型，不是搜索

每换一个并行配置 = `compile_cache` key 变化 = **完整 AOT 重编译**。实测
（`benchmark/trn2/flux_1_dev.json`）Flux 1024×1024 编译一次要 **1484 秒（约 25 分钟）**，
Qwen-Image 1316 秒。所以：

1. **默认走解析成本模型**（零成本、瞬时出结果）。
2. **实测优先**：命中测量库就用实测值覆盖预测值。种子数据来自已有的
   `benchmark/trn2/*.json` 和 `scripts/verify_cli.py` 的 `results.json`。
3. **可选 `--auto-tune`**（P5）才真的上机扫描，结果写回测量库。

### 模块划分（新包 `difflet/planner/`）

**`hardware.py` — `HardwareProfile`**

优先级：显式 `--total-cores` > `NEURON_RT_VISIBLE_CORES` / `NEURON_RT_NUM_CORES` >
`neuron-ls -j` > `get_platform_target()` 兜底。`neuron-ls -j` 已在本机验证可用：

```json
{"instance_type": "trn2.3xlarge", "neuron_device": 0, "nc_count": 4,
 "logical_neuroncore_config": 2, "memory_size": 103079215104,
 "neuroncore_ids": [0,1,2,3], "neuron_processes": []}
```

**`model_profile.py` — `ModelProfile`**
- **能力标志**（是否蒸馏、支持哪些 CP mode、是否支持 SP）→ 上移到
  `difflet/registry.py` 的 `ModelEntry`，成为单一事实源。
- **维度**从 HF `config.json` 读，走已有的
  `difflet/utils/diffusers_adapter.py::load_diffusers_config`，**不加载权重**。
- **序列长度**复用各模型 `InferenceConfig` 已有的推导属性。
- **参数字节**复用 `difflet/cli/dp/hbm_check.py::component_weight_bytes`。

**`feasibility.py`** — 枚举 + 按 §2 的 A–F 全部过滤。

**`cost_model.py`** — 解析打分。**`measurements.py`** — 实测库。
**`planner.py`** — 顶层 `plan(...) -> Plan`。

细节见下面每个阶段的说明。

---

## Part 5 — 实施阶段（本次做到 P3）

### P0 — 真实硬件探测

**做什么**：新增 `difflet/planner/hardware.py`，用 `neuron-ls -j` 真实读取硬件，
产出 `HardwareProfile`；把 `common/neuron_cores.py` 里写死的
`DEFAULT_NEURON_CORE_IDS = (0,1,2,3)` 换成探测结果；让 `_validate_dp` 在能探测到
核数时**强制**做容量检查（今天探测不到就完全不检查）。

```python
@dataclass(frozen=True)
class HardwareProfile:
    instance_type: str          # "trn2.3xlarge"
    platform_target: str        # "trn2"
    num_devices: int            # 1
    cores_per_device: int       # 4
    total_cores: int            # 4
    hbm_bytes_per_device: int   # 103_079_215_104
    lnc: int                    # 2
    busy_cores: tuple[int, ...] # 从 neuron_processes 推出，已被别的进程占用的核
```

**举例**：今天在一台 8 核机器上跑 `difflet generate --model-id ... --dp 2`，
既没设 `--total-cores` 也没设 `NEURON_RT_NUM_CORES` —— `_validate_dp` 里
`total is None` 直接跳过检查，程序一路跑到 runtime 才 SIGSEGV 或报
`global communicator` 错误。P0 之后，planner 探测到 `total_cores=8`，
`dp=2 × tp=8 = 16 > 8`，在**编译开始前**就报清楚的错。

反过来在一台 64 核的 trn2.48xlarge 上，今天
`resolve_available_neuron_core_ids` 会返回 `(0,1,2,3)`，用户即使想用 tp=8 也
拿不到核 —— 静默地只用了 1/16 的机器。

**验收**：本机 `HardwareProfile` 断言 `instance_type == "trn2.3xlarge"`、
`total_cores == 4`、`hbm_bytes_per_device == 103079215104`；`neuron-ls` 不存在时
回落到 `get_platform_target()` 且不崩。

---

### P1 — 模型能力元数据统一到 `ModelEntry`

**做什么**：把「这个模型支持哪些并行策略」这件事集中到
`difflet/registry.py::ModelEntry`，删掉散落的 4 份副本。

```python
@dataclass(frozen=True)
class ModelCapabilities:
    is_distilled: bool           # True => 拒绝 cfg_parallel
    supports_cp: bool
    cp_modes: frozenset[str]     # {"gather_kv","ring","ulysses"} 的子集
    supports_sp: bool
    num_attention_heads: int     # flux/hunyuan/qwen=24, wan=40
```

**举例**：今天「Flux 是蒸馏模型所以不能用 CFG-parallel」这个事实写在四个地方 ——
`cli/main.py:532` 的 `_DISTILLED_MODELS`、`cli/modes.py:16` 的 `MODEL_CLASS`、
`models/*/entry.py` 的 guard、`scripts/verify_cli.py:89` 的 `DISTILLED`。
最后一份需要 `tests/unit/cli/test_verify_cli.py` 专门写一个 drift-guard 测试来
防止它和 `cli/main.py` 漂移 —— 这个测试在 P1 之后就可以退休了。

新增一个模型时，今天要记得改 4 个地方；P1 之后只填一个 `ModelCapabilities`。

**验收**：`_DISTILLED_MODELS` / `_SP_SUPPORTED_MODELS` / `MODEL_CLASS` /
verify_cli 的 skip 集合全部改为从 registry 派生；drift-guard 测试删除；
`scripts/verify_cli.py` 全矩阵行为不变。

---

### P2 — 可行性枚举 + `difflet plan`（先不打分）

**做什么**：`difflet/planner/feasibility.py` 枚举所有
`(tp, cp, cp_mode, cfg, sp, dp)` 组合并按 §2 的 A–F 过滤；新增只读命令
`difflet plan`，打印合法配置和每个被拒配置的理由。**完全不碰设备、不编译**。

关键设计：**互斥规则不重新实现一遍**。枚举时直接构造
`DiffletParallelConfig` 并捕获 `ValueError`，这样 planner 永远不会和运行时的真实
约束漂移。额外补上目前无人执行的 ulysses `heads % (tp·cp) == 0`。

**举例 1 — Flux 在本机 4 核**（heads=24）。`world_size` 必须 = 4：

| 候选 | 结论 |
|---|---|
| `tp=4` | ✅ |
| `tp=4 --sp` | ✅ Flux 在 `_SP_SUPPORTED_MODELS` 里 |
| `tp=2 cp=2 gather_kv` | ✅ |
| `tp=2 cp=2 ring` | ✅ |
| `tp=2 cp=2 ulysses` | ✅ 24 % (2×2) == 0 |
| `tp=1 cp=4 ulysses` | ✅ 24 % 4 == 0 |
| `dp=2 tp=2` / `dp=4 tp=1` | ✅ |
| `tp=2 --cfg-parallel` | ❌ Flux 是蒸馏模型，没有第二条 CFG 分支 |
| `tp=2 cp=2 --sp` | ❌ SP 与 cp>1 互斥 |
| `tp=3` | ❌ 3 不整除 4，且 24 % 3 虽为 0 但核数不够 |

**举例 2 — ulysses 整除约束会真的咬人**：在 16 核上给 Flux 选
`tp=8 cp=2 ulysses`，`24 % (8×2) = 24 % 16 = 8 ≠ 0` → 非法，但同样的
`tp=8 cp=2 gather_kv` 是合法的。**今天这个错误要一直等到 attention 层
`_ulysses_check_heads` 在编译期才报**，P2 之后在枚举时就排除掉。

**举例 3 — HunyuanVideo 的 `tp2cp2`**：语义上完全合法，但 `verify_cli` 的
XFAIL 表记录它触发 neuronx-cc `NCC_INLA001` 崩溃。planner 把它放进 denylist，
标注「已知编译器缺陷」而不是「配置非法」—— 两者对用户是不同的信息。

**验收**：`tests/unit/planner/` 对每个 (模型 × 4 核) 断言可行集与 §2.B 矩阵逐格
一致；断言可行集 == `verify_cli.py` 实测矩阵中非 skip 的格子（含 XFAIL denylist）。

---

### P3 — 成本模型 + 测量库 → 排序输出

**做什么**：给 P2 的可行集打分排序。两个来源，实测优先。

**`measurements.py`** — key = `(instance_type, model_id, shape, steps, parallel)`，
value = `step_latency_ms / e2e_s / peak_mem / compile_seconds`。种子直接来自现有
benchmark JSON，例如 `benchmark/trn2/flux_1_dev.json`：

```
flux @ tp=4, 1024×1024, 28 steps → step_latency 268ms, e2e_warm 35.3s, compile 1484s
wan_2_2 @ tp=4, 480×832×9, 20 steps → step_latency 555ms, e2e_warm 57.4s
qwen_image @ tp=4, 1024×1024, 20 steps → step_latency 447ms, e2e_warm 62.7s
```

命中就标 `evidence=measured`，没命中标 `evidence=predicted`，**输出里如实标注**。

**`cost_model.py`** — 单步延迟 ≈ 计算 + 通信 + 固定开销：

- 计算随 `tp·cp` 近似线性下降，对过小的 per-core 分片加次线性惩罚
- attention 通信量（每 rank 每次 attention）：
  - `gather_kv`：`O(S · H/tp · d · (cp−1)/cp)` × 2（K 和 V）
  - `ring`：总量相当，但可与计算重叠 → 打折
  - `ulysses`：`O(S/cp · H/tp · d)` × 2 次 all-to-all → **比 gather_kv 少一个 cp 因子**
- TP all-reduce：每 block `O(S · hidden)` × 2
- SP：把 all-reduce 换成 reduce-scatter + all-gather，通信量相当，省激活内存
- CFG-parallel：per-rank batch 减半 → 接近 2× 加速，每步一次 gather（很便宜）
- DP：纯吞吐乘子，**不改单请求延迟**
- 目标函数：`latency` 最小化单请求延迟；`throughput` 最大化
  `dp / per-replica-latency`；`balanced` 加权

常数从 benchmark JSON 标定 + Neuron 峰值算力 / HBM 带宽 / collective 带宽。

**举例 — Flux 1024×1024（S_img=4096, S_txt=512, heads=24, head_dim=128）在 4 核上，
`--objective latency`**，planner 输出大致形如：

```
model=black-forest-labs/FLUX.1-dev  hw=trn2.3xlarge (1 device, 4 cores, 96GB)
shape=1024x1024  steps=28  objective=latency

  #  config                    est. e2e   evidence    note
  1  tp=4 --sp                   31.2s    predicted   SP 省激活内存；world 不变
  2  tp=2 cp=2 --cp-mode ulysses 33.0s    predicted   24%(2*2)=0 ✓ 通信量最低的 CP 模式
  3  tp=4                        35.3s    measured    benchmark/trn2/flux_1_dev.json ← 已编译
  4  tp=2 cp=2                   36.1s    predicted   gather_kv 通信量是 ulysses 的 2×
  5  tp=2 cp=2 --cp-mode ring    37.4s    predicted   实验性 nkilib kernel
     dp=4 tp=1                     —      —           吞吐配置，latency 目标下不适用

rejected:
  tp=2 --cfg-parallel     蒸馏模型，无第二条 CFG 分支
  tp=2 cp=2 --sp          SP 与 cp>1 互斥
```

**缓存亲和（Difflet 相对 GPU 框架的独有价值）**：因为换配置就要重编译，
planner 用 `compile_cache.has_valid_manifest()` 标出哪些候选**已经编译好**，
并提供 `--prefer-cached`：只有当预测收益超过重编译摊销成本时才建议换。
上面例子里第 3 名 `tp=4` 已有缓存，若预测第 1 名只快 12%，而重编译要 25 分钟，
那么对一次性任务应该建议留在 `tp=4`。GPU 框架是 JIT 的，没有这个问题也就没有
这个功能。

**验收**：
- `difflet plan --model-id black-forest-labs/FLUX.1-dev --objective latency`
  在本机给出上述形态的排序表，且 `tp=4` 一行标为 `measured` 并匹配 35.3s。
- `--objective throughput` 时 `dp=4 tp=1` 一类应排到前面。
- 成本模型对已有 benchmark 点的预测误差在合理范围内（先定 ±25% 的宽门槛，
  单测里对 3 个已知点断言）。

---

### P4 / P5（本次不做，记录方向）

- **P4** — `--auto` / `--objective` 挂到 `compile` / `generate` / `run` / `serve`
  的 `_add_parallel_flags`；显式并行 flag 永远覆盖 planner（沿用 `resolve_mode`
  的 per-field 覆盖语义）；`--mode` 保留为兼容别名，硬件可探测时委托给 planner，
  探测不到时回落现有 `_BASE` 表。
- **P5** — `--auto-tune` 实机扫描 top-K 候选并回写测量库。注意单次扫描的代价：
  Flux 每个候选约 25 分钟编译 + 生成，扫 5 个候选就是 2 小时以上。

---

## Part 6 — 多设备支持（第二版）

第一版**只支持单 Neuron device**（`num_devices == 1`，如 trn2.3xlarge 的 4 核）。
这不是偷懒，而是因为多设备的成本模型需要区分设备内 NeuronLink 与跨设备互联的
带宽/延迟，而我们**目前一条多设备实测数据都没有** —— 强行建模只会给出置信度很低
的排序。

**第一版就预留好的接口**（这样第二版是扩展而不是重写）：

1. `HardwareProfile` 从一开始就带 `num_devices` / `cores_per_device`，而不是只有
   一个扁平的 `total_cores`。单设备时 `num_devices == 1`。
2. `neuron-ls -j` 返回的是**每个 device 一条记录**的数组，含 `connected_to`
   （设备间拓扑）、`numa_node`、`cpu_affinity`。第一版只读第 0 条，但解析器按数组
   写，多设备时天然可用。
3. `cost_model` 的通信项写成 `comm_bytes / bandwidth(axis, placement)` 的形式，
   第一版 `bandwidth()` 返回常数，第二版按「该轴的 rank 组是否跨设备」返回不同值。
4. `feasibility` 预留一个 `placement` 概念：mesh 的哪个轴落在设备内、哪个跨设备。
   第一版恒为「全部设备内」。

**第二版要新增的**：
- 拓扑感知的 placement 打分 —— 同样是 `tp=8 cp=2`，把 tp 组放在单设备内
  （8 核不跨设备）还是跨 2 个设备，通信成本差很多。这正是 AoiZora 论文指出的、
  现有 auto-parallel 系统普遍忽略的点：只搜逻辑 mesh，不看物理互联布局。
- 搜索空间从个位数涨到几十个 —— 仍可穷举，但需要先按整除性剪枝再打分。
- 跨设备的 HBM 预算按 device 而非全局算（`hbm_check.py` 目前是单个 96GB 常数）。
- 需要在 trn2.48xlarge 上跑一轮 benchmark 播种测量库，否则成本模型无从标定。

---

## 验证（整体）

- **单测** `tests/unit/planner/` — 合成 `HardwareProfile` + `ModelProfile` 输入，
  断言枚举结果与 §2 矩阵逐格一致；断言 planner 产出的每个配置都能成功构造
  `DiffletParallelConfig`（即永不产出运行时会拒绝的配置）。
- **一致性测** — 对每个 (模型, 4 核)，planner 可行集 == `verify_cli.py` 实测矩阵
  中非 skip 的格子，XFAIL 格单独标注为 denylist 而非 infeasible。
- **端到端** — 本机跑
  `difflet plan --model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024 --objective latency`，
  确认输出排序合理、`tp=4` 行命中实测 35.3s、缓存标记正确。
- **回归** — `scripts/verify_cli.py` 全矩阵仍全绿。P0–P3 不改变任何既有默认行为
  （`difflet plan` 是纯新增的只读命令，`--auto` 要到 P4 才引入）。
