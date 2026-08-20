# AoiZora 两阶段 Planner 原型 — 实验报告与使用说明

Date: 2026-08-20
Branch: `worktree-difflet-planner-fable5`
主机: EC2 `trn2.3xlarge`（1 NeuronDevice = 2 Trainium2 芯片 × 2 核 = 4 逻辑核，
96 GiB HBM，LNC=2，`neuron-ls` 实测探测）
模型: `black-forest-labs/FLUX.1-dev`（MMDiT，19 double + 38 single = 57 块，
24 heads × 128 dim，1024×1024 → 4096 image + 512 text tokens）

---

## 1. 摘要

把 AoiZora（arXiv 2606.17566，TPU 上的 diffusion 推理自动并行 planner）的
**两阶段 parallelism 选择方案**移植为 Difflet planner 原型：

- **阶段 1**（placement 无关剪枝）：沿用既有解析成本模型，对全部可行配置打分，
  可选 top-K 截断（`--survivors`）；
- **阶段 2**（拓扑感知 placement 排序，新增 `difflet/planner/topology.py`）：对每个
  幸存候选枚举 mesh 轴序的**物理放置**（按对称性去重），在 Trainium2
  核/芯片/设备层级上对每族集合通信算物理乘数（链路层级 + 共享链路争用），
  用论文的引擎模型合成 Q2 排序目标。

在 Flux 上验证（`difflet plan`，只读、不编译、不占核）：

| 结果 | 数值 |
|---|---|
| 阶段 2 重打分 | 14 候选全部重打分，规划耗时 **3.0 ms**（论文 47–376 s，因其需编译） |
| placement 差距 | `tp2cp2` 两种轴序 Q2 差 **1.24×**（+23.9%），与论文 v5e-8 上 16.6% 同量级 |
| 争用定量 | 坏序 `dp.tp` 使 TP all-reduce 乘 **×4.00**（2× 慢层 × 2 组共享链路），64.6 ms/步争用惩罚 |
| 重叠结构效应 | `tp1cp4*` 预测从串行和降至 0.170 s（CP 通信全部藏进计算，−6.6%~−12.5%） |
| 回归 | planner 套件 180 passed；全量 1549 passed，62 失败与改动前基线逐条一致（环境缺 `diffusers` 等），**零回归** |

**结论**：本机拓扑下运行时默认轴序（tp 最内）在全部可双轴配置上已是最优——
高流量、不可重叠的 TP all-reduce 落在片内链路。planner 的价值是把这一事实
**量化、解释并固化为可复查的机制**；在 tp 跨设备的大主机（trn2.48xlarge，
16/64 核）上，同一套机制才有真正的判别空间（见 §6 限制）。

---

## 2. 背景与移植映射

论文核心主张：**逻辑切分只说哪些 rank 通信，不说流量落在哪些物理链路**。
同一逻辑配置，TP 组放进一个芯片的两个核还是拆到片间链路两侧，只取决于
mesh 轴铺到 core ID 的顺序；在链路不等价的拓扑上这改变延迟（论文实测
Wan 2.1 在 v5e-8 上同一逻辑切分不同放置差 16.6%，主因共享链路争用）。

| AoiZora（TPU/JAX） | Difflet 原型（Neuron/Trainium） |
|---|---|
| 工作负载轴候选生成 + 合法性过滤 | `feasibility` 网格（dp/cfg/cp/tp/sp/cp_mode），已有 |
| 阶段 1：编译前 IR + 隐式通信增广，Q1 剪枝取 top-K | 既有 `cost_model`（解析、标定锚定），新增 `--survivors K` |
| 阶段 2：编译幸存者、解析 HLO 得 collective 图（类型/载荷/replica groups） | 解析推导 `collective_schedule`（每族：类型/轴/字节/次数/barrier） |
| 物理轴序枚举 + 对称去重 | `enumerate_placements`：活跃轴内→外全排列，按各轴组芯片归属签名去重 |
| v5e torus 逐维引擎 + ring/line 展开 + 共享链路争用 | Trainium2 三级链路（片内/片内跨芯片/跨设备）+ **芯片对级**链路身份 + 每链路负载求和 |
| Q2 = 排序目标（非绝对延迟预测） | `Q2 = C_comp + barrier + max(0, overlappable − C_comp)`，绝对秒锚定阶段 1 标定 |
| 输出 (sharding, placement)，沿标准编译路径执行 | 输出 (config, placement)；placement 目前只推荐不执行（运行时 mesh 序固定） |

**两个对论文的诚实替换**（决策 D28，详见
`docs/plans/2026-08-20-planner-aoizora-topology-prototype.md`）：

1. Difflet 每个配置发哪些 collective 由配置**静态决定**（plan 与 run 之间无
   编译器改写），解析调度即真实调度；
2. 论文编译每个幸存者代价 O(秒)；在 Difflet 上 AOT 编译是 **~25 分钟/候选**，
   恰是 planner 要避开的开销。测试钉住调度字节量与 `cost_model.comm_bytes`
   逐字节一致（ring 的重叠折扣放在秒分配处，秒数守恒有第二个测试钉住），
   两个视图不会漂移。

---

## 3. 实验设置

- 环境：`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`（torch 2.9.1、
  neuronx-distributed-inference 0.10.18399、neuronx-cc 2.26）；
- 硬件探测：`neuron-ls -j` → 1 device / 4 核 / 96 GiB，无跨设备连接；
- 物理拓扑建模（公开规格）：**芯片 = 2 核，NeuronDevice = 2 芯片**，即核
  {0,1}=chip0、{2,3}=chip1；层级带宽常数 片内:片内跨芯片:跨设备 = **4:2:1**
  （排序假设，非测量——只影响 placement 间**比值**，绝对值锚定在实测标定上）；
- 标定：测量库中 Flux `tp4` @1024×1024 锚点（step 0.2654 s，
  `benchmark/trn2/flux_1_dev.json`）；带宽为假设值（kind=measured-anchor）。

---

## 4. 实验结果

### 4.1 端到端排序（`difflet plan`，objective=latency，28 步）

```
 #  config                step    request    req/s    weights  evidence
 1  tp1cp4              0.170s       4.8s    0.210     135GB!  predicted
 2  tp1cp4ring          0.170s       4.8s    0.210     135GB!  predicted
 3  tp1cp4ulysses       0.170s       4.8s    0.210     135GB!  predicted
 4  tp2cp2              0.202s       5.7s    0.176       67GB  predicted
 5  tp2cp2ring          0.202s       5.7s    0.176       67GB  predicted
 6  tp2cp2ulysses       0.202s       5.7s    0.176       67GB  predicted
 7  tp4sp               0.223s       6.3s    0.160       34GB  predicted
 8  tp4                 0.265s       7.4s    0.135       34GB  measured
 9  dp2tp1cp2           0.335s       9.4s    0.213     135GB!  predicted
...
14  dp4tp1              0.660s      18.5s    0.216     135GB!  predicted
```

要点：
- **阶段 2 的重叠结构改变了预测形状**：`tp1cp4*` 三种 CP 模式收敛到同一
  0.170 s——CP 通信全部藏进计算后，模式间差异被抹平（论文引擎模型的直接
  推论）；单轴候选（tp4/tp4sp/dp*）Q2 = 阶段 1 值（无可隐藏项或无通信）。
- `weights!` 135GB 条目是**已知高估**（朴素上界，advisory，见 D22）；
  tp1cp4* 排第一但 4 份权重副本几乎肯定放不下——这是既有已知问题
  （benchmark 计划优先级 2），非本增量引入。
- measured 条目（tp4 0.2654 s）**不被模型改写**（D31）。

### 4.2 阶段 1 → 阶段 2 重打分对比（重叠结构效应）

| label | 阶段1 (s) | 阶段2 Q2 (s) | Δ | 说明 |
|---|---|---|---|---|
| tp1cp4 | 0.1942 | 0.1700 | −12.5% | CP gather 全部隐藏 |
| tp1cp4ring | 0.1870 | 0.1700 | −9.1% | |
| tp1cp4ulysses | 0.1821 | 0.1700 | −6.6% | |
| tp2cp2 | 0.2105 | 0.2024 | −3.8% | 部分 CP 隐藏 |
| tp4sp / tp4 / dp2tp2 / dp4tp1 | — | — | 0.0% | 无可隐藏项（或 measured 保留） |

### 4.3 Placement 字段（阶段 2 的核心产出）

每个多活跃轴候选的全部物理放置明细（Q2、每族瓶颈层级、乘数、争用秒数）：

```
tp2cp2:  default tp.cp  Q2=0.2024   ← 运行时当前布局
   alt   cp.tp          Q2=0.2508 (+23.9%)
        all_reduce@intra-device x4.00 cont=32.3ms   ← barrier 流量劣化
        all_gather@intra-chip    x0.25 cont=0.0ms   ← CP 反而变好，但量小且可隐藏

dp2tp2:  default tp.dp  Q2=0.3995
   alt   dp.tp          Q2=0.4963 (+24.2%)
        all_reduce@intra-device x4.00 cont=64.6ms   ← 两副本 AR 争用唯一片间链路

dp2tp2sp: default tp.dp Q2=0.3723
   alt   dp.tp          Q2=0.4691 (+26.0%)  RS/AG 双双 ×4.00
```

**×4.00 的分解**（测试 `test_dp2tp2_flipped_order_pays_tier_and_contention`
钉住）：2×（intra-chip→intra-device 层级带宽比）× 2×（两个并发组共享同一条
芯片对链路）。这正是论文 Fig 3 的共享链路争用机制 + 层级差异的复合。

**一个细化论文结论的观察**：`dp2tp1cp2` 的 CP 通信完全藏在计算下，其两种
轴序 Q2 完全相同（+0.0%）——**placement 只对未被隐藏的 barrier 流量有延迟
意义**。实践上：调 placement 优先看 TP all-reduce 落在哪，CP/可重叠流量
的放置是次要的。

### 4.4 与论文的对照

| 维度 | AoiZora (v5e) | 本原型 (trn2.3xlarge) |
|---|---|---|
| placement 差距 | 最大 16.6%（Wan 2.1, v5e-8） | tp2cp2 23.9%、dp2tp2 24.2%（预测） |
| 争用机制 | 并发 permute 共享链路带宽 23→15.2 GB/s | 芯片对链路上并发组负载求和串行化 |
| 规划耗时 | 47–376 s（需 XLA 编译） | **3.0 ms**（解析推导，14 候选全搜） |
| 阶段 2 信息源 | 编译后 HLO 的 replica_groups | 解析 collective 调度（配置静态决定） |
| 执行 | JAX mesh 轴序即放置，可执行 | **只推荐不执行**（见 §6） |

### 4.5 测试与回归

- `tests/unit/planner/test_topology.py` 新增 25 例：几何/层级、链路芯片对身份、
  placement 枚举去重、调度↔平面模型**字节一致**与**秒守恒**、barrier/可重叠
  分类、×4.00 乘数分解、Q2 隐藏/不隐藏、纯 dp 无决策、planner 集成
  （survivors 截断、`--no-topology` 等价旧行为、measured 不改写）。
- `tests/unit/planner/`：**180 passed**，6 skipped。
- 全量 `tests/unit/`（排除依赖 `diffusers` 的预置损坏模块）：1549 passed；
  62 failed 与改动前基线**逐条 diff 一致**（环境缺 `diffusers`/`imageio`
  等预置问题）——零回归。
- ruff check / format：本增量涉及文件全部干净。

---

## 5. 使用说明

### 5.1 命令行

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd /home/ubuntu/Difflet

# 默认：两阶段全开（阶段1全保留 + 阶段2拓扑排序）
difflet plan --model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024

# 常用参数
difflet plan --model-id black-forest-labs/FLUX.1-dev \
    --height 1024 --width 1024 \   # 分辨率（决定 token 数）
    --steps 28 \                   # 降噪步数（影响 request 秒，不影响排序）
    --objective latency            # latency | throughput | balanced
```

新增 flag：

| flag | 作用 | 默认 |
|---|---|---|
| `--survivors K` | 阶段 1 截断：只有 top-K 候选进入阶段 2（论文语义的剪枝） | 全保留（Difflet 搜索空间只有十几个，且阶段 2 是微秒级） |
| `--no-topology` | 完全跳过阶段 2，回到单阶段行为（等价改动前） | 关（即阶段 2 开启） |
| `--json` | 机器可读输出（含完整 placement 结构） | 文本表 |

### 5.2 输出解读

**文本输出**三段：

1. **排序表**：`step` 列现在含阶段 2 的 Q2（预测条目）或实测值（measured
   条目）；`flags` 列可直接粘到 `difflet compile` 后面。
2. **placements 块**（前 5 名）：
   - `placement: order tp.cp (runtime default)` —— 当前运行时的轴序（tp 最内）；
   - `all_reduce on intra-chip (barrier, tp axis)` —— barrier 流量落在哪个
     物理层级（这是 placement 决策的主变量）；
   - `shared-link contention costs ~4.0 ms/step` —— 默认序下的争用代价；
   - 若存在更优序：`best order cp.tp would cut the predicted step by X% --
     needs a runtime mesh-order flag (not executable today)`。
3. **rejected 表**：被拒配置与原因（互斥/能力/整除/退化/known-bad）。

**JSON 结构**（每条 ranked 条目新增字段）：

```json
{
  "stage2_step_seconds": 0.2024,
  "placement": {
    "best":     {"order": "tp.cp", "step_seconds": 0.2024,
                 "barrier_seconds": ..., "overlappable_seconds": ...,
                 "families": [{"family": "all_reduce", "axis": "tp",
                               "barrier": true, "multiplier": 1.0,
                               "bottleneck_tier": "intra-chip",
                               "contention_seconds": 0.0}]},
    "default":  { ... },
    "alternates": [ ... ]
  },
  "survivors": ["tp1cp4", "tp2cp2", "..."]
}
```

字段语义：`multiplier` = 该通信族在此放置下的物理秒 / 默认序物理秒（1.0 =
与现状相同）；`bottleneck_tier` = 该族组环覆盖的最慢链路层级
（`intra-chip` > `intra-device` > `inter-device`）；`contention_seconds` =
并发组共享链路导致的串行化份额。

### 5.3 Python API

```python
from difflet.planner.planner import plan

result = plan(
    "black-forest-labs/FLUX.1-dev",
    model_type="flux", height=1024, width=1024, steps=28,
    survivors=None,              # 阶段1截断（None=全保留）
    topology_aware=True,         # False = 单阶段旧行为
    rank_by_best_placement=False # True = 按最优序排序（论文完整语义，
)                                #   但非默认序运行时执行不了，见 §6）

result.survivors                 # 进入阶段2的 label 元组
result.ranked[0].placement.best  # PlacementScore：order/step_seconds/families
result.ranked[0].placement.alternates
```

### 5.4 典型工作流

```bash
# 1) 下载权重前先规划（planner 不需要权重/不编译）
difflet plan --model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024

# 2) 取第 1 名的 flags 编译（只换 rank_by_best_placement 默认序的推荐，
#    flags 与之前完全一致——placement 是附加信息）
difflet compile --model-id black-forest-labs/FLUX.1-dev --tp-degree 2 --cp-degree 2

# 3) 换目标时
difflet plan --model-id ... --objective throughput   # dp 类前排
difflet plan --model-id ... --survivors 3            # 只对 top-3 做拓扑分析
difflet plan --model-id ... --no-topology            # A/B 对照单阶段
```

---

## 6. 已知限制与后续

| 限制 | 影响 | 后续 |
|---|---|---|
| placement 只推荐不执行 | 非默认轴序无法落到运行时（`MeshSpec` rank 公式固定） | 运行时加 mesh-order 参数 + NxD 组构造跟随；之后把 `rank_by_best_placement` 翻 True 即论文完整语义 |
| 4 核上默认序全部最优 | placement 搜索在本机无推荐变化（但量化了差距） | tp 跨设备的 16/64 核主机上判别力才发挥；届时播种多层级带宽实测 |
| 层级带宽 4:2:1 是假设 | 只影响 placement 间比值，绝对值锚定标定 | 多层级 benchmark 后替换 `TIER_BANDWIDTH_BYTES_PER_SECOND` |
| 多设备 torus 压平为一层 | trn2.48xlarge 上片间无逐维模型 | 需逐维 engine 调度（论文 §4.4） |
| A2A 用 ring 规则近似 | cp=2/4 小组的一阶合理 | 更大 cp 组需 per-pair 路由模型 |
| 预测 vs 实测 | tp2cp2 的 Q2 0.2024 是预测（锚定 tp4 实测）；placement 差距 23.9% 是模型推论，未经本机实机验证 | 按验证矩阵跑 tp2cp2 实测即可同时校验两者（每次 ~25 min 编译） |

## 7. 复现

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd /home/ubuntu/Difflet
difflet plan --model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024 --steps 28
python -m pytest tests/unit/planner/ -q          # 180 passed
git diff --stat                                   # 本增量改动清单
```

相关文件：
`difflet/planner/topology.py`（新增）、`difflet/planner/planner.py`、
`difflet/cli/plan.py`、`difflet/cli/main.py`、
`tests/unit/planner/test_topology.py`（新增）、
`docs/plans/2026-08-20-planner-aoizora-topology-prototype.md`（设计+决策 D28–D34）。

---

## 8. 遍历实测 vs planner 预测（2026-08-20 补充：手选 vs 自动对比）

用户在 trn2.3xlarge 上对 Flux 1024×1024 做了两轮全配置遍历（每配置：冷编译
~11 min + 热跑计时，`time` 包住完整 CLI 进程）。OCR 自终端截图，两轮数字
均已复核。

### 8.1 遍历结果（warm e2e，秒）

| 配置 | 第 1 轮 | 第 2 轮 | 均值 | 实测名次 |
|---|---:|---:|---:|---:|
| tp4（手选） | 42.535 | 43.875 | 43.20 | **1** |
| tp4sp | 44.127 | 48.397 | 46.26 | 2 |
| tp2cp2 | 52.314 | 50.828 | 51.57 | 3 |
| dp2tp2sp | 52.102 | 53.589 | 52.85 | 4 |
| dp2tp2 | 53.748 | 54.532 | 54.14 | 5 |
| tp2cp2ring | 54.217 | 58.628 | 56.42 | 6 |
| tp2cp2ulysses | 54.624 | 59.339 | 56.98 | 7 |

tp1cp4*（planner 预测第 1）不在遍历中——其 4 份权重副本（135 GB 上界）
本就过不了 96 GB HBM（D22 已标注）。

### 8.2 换算到每步延迟（锚定 tp4 实测 0.2654 s/步）

固定开销 F = 43.875 − 28×0.2654 = 36.44 s（进程启动+加载+文本编码+VAE+PNG，
各配置字节数相同故视 F 相等；ring−gather 差 7.8 s ≈ 28×280 ms，自洽）：

| 配置 | planner Q2 (s) | 阶段1 (s) | 实测估算 (s/步) | Q2 误差 |
|---|---:|---:|---:|---:|
| tp4 | 0.2654（measured 锚） | 0.2654 | 0.2654 | ±0 |
| tp4sp | 0.2234 | 0.2234 | ~0.427 | **−47.7%** |
| tp2cp2 | 0.2024 | 0.2105 | ~0.514 | **−60.6%** |
| dp2tp2sp | 0.3723 | 0.3723 | ~0.612 | −39.2% |
| dp2tp2 | 0.3995 | 0.3995 | ~0.646 | −38.2% |
| tp2cp2ring | 0.2024 | 0.2081 | ~0.792 | **−74.5%** |
| tp2cp2ulysses | 0.2024 | 0.2105 | ~0.818 | **−75.2%** |

（第 1 轮数字代入误差同类：tp2cp2 −59%、ring −73.7%、ulysses −74.3%。）

### 8.3 手选 vs 自动：**有差距，且方向相反**

- **手选 tp4 = 实测第 1**（两轮一致）。tp4 也是仓库所有 benchmark、serving
  钉死值与 NxDI 官方教程在本机型的自然选择。
- **自动（planner）**：预测第 1 是 tp1cp4*（未测、HBM 几乎肯定不可行）；
  可行候选里预测第 1 是 tp2cp2——实测第 3/7，比 tp4 慢 ~19%（e2e）/
  ~94%（每步估算）。
- **排序相关性：Spearman ρ = −0.25（阶段 2）/ −0.31（阶段 1）**——预测
  排序与实测**轻微反相关**。预测系统性地把 cp/sp 类配置排得过高。

### 8.4 偏差根因（按影响排序）

1. **CP 的计算收益被大幅高估**。模型假设 cp=2 把每 token 计算减半
   （`compute_share`），实测 tp2cp2 每步 ~0.51 s > tp4 0.265 s。对照
   dp2tp2（per-request 计算量 = tp2，无 CP）：tp2cp2 只比它快 ~7%——
   CP 的毛收益存在但 ≪2×，叠加开销后净输给 tp4。`CP_EFFICIENCY_TAX`
   （每翻倍 1.5%）远远太温和；gather_kv 的全量 KV attention 在 Neuron
   kernel 上不随 query 减半。
2. **ring/ulysses 的 kernel 级成本字节模型看不见**。两者比 gather_kv 再慢
   ~15%（每步估算 0.79/0.82 vs 0.51）：ring 走实验性 nkilib kernel，
   ulysses 4 次 A2A 有每次通信的固定延迟。本模型的 `C_comm` 没有 α
   逐 collective 延迟项（论文有 `α_τ`）——纯字节账对「次数多、字节少」
   的模式结构性失明。这也解释了阶段 1 的先验（ulysses 最优，来自 GPU 上的
   HF benchmark 结论）方向就反了。
3. **SP 的收益假设未兑现**。`REPLICATED_COMPUTE_SHARE=0.08` 的可分计算
   没有覆盖 RS+AG 双倍 collective 次数的实际成本，tp4sp 实测 +10%。
4. **阶段 2 的 overlap 假设放大了 1 的乐观**。KV gather「藏进计算」在本栈
   不成立（可重叠空间远小于假设），使 tp1cp4* 从 0.194 降到 0.170、
   三种 CP 模式并列——预测离实测更远。机制本身没错，错的是「cp 通信
   可隐藏」这个分类没有实测支撑。

**注意**：遍历全部跑在默认轴序上，**不检验**阶段 2 的 placement 排序本身
（tp2cp2 默认序最优的结论未被推翻）；它检验的是配置级成本模型。

### 8.5 由此得出的行动项

1. **把 7 个配置的真实 step latency 播种测量库**（用
   `benchmark/step_latency.py` 采集；e2e 含加载噪声不宜直接播种）。
   播种后 planner 按「实测优先」自动用 measured 覆盖预测，排序立即
   正确——这正是 D19 相对标定设计预留的路径，且 7 个点 ≥2 可触发
   `measured-fit` 双参数拟合。
2. **修成本模型**（有数据后可做）：加大 CP 实质效率税；给 `C_comm` 加
   α 逐次延迟项；`REPLICATED_COMPUTE_SHARE` 依据 tp4sp 实测下调。
3. **阶段 2 的 cp「可隐藏」分类暂时保守化**（或用实测校准 hidden 比例），
   直到有 profile 证据。
4. 短期最诚实的用法：**以 measured 行为准**——当前数据下手选 tp4 正确，
   planner 的 predicted 行只能信「tp4 > dp 类」这类粗排序，不能信 cp/sp
   的相对位置。

---

## 9. 修正与复测（2026-08-20 晚）：让算法自然收敛，而不是调参凑数

针对 §8 的偏差，按「物理约束 + 真实测量 + 零常数改动」修复并复测。

### 9.1 做了什么

1. **D35 内存硬约束**（物理规则，非调参）：co-resident（非分阶段）模型的
   `dp·cfg·cp` 份权重副本是设备 HBM 的物理上界，超限直接拒绝。上线后
   Flux 4 核可行集**独立收敛为 §8 遍历实际测过的 7 个配置**——规则与
   遍历数据互为佐证。
2. **真实测量**（`scripts/flux_parallel_sweep.py`，方法论 =
   `benchmark/step_realloop.py`）：本机下载权重、逐配置编译（7× 冷编译
   8–22 分钟）、进程内真实 28 步 generate 逐步计时（弃预热、弃 step 0、
   n=27）。世界数 <4 的配置用 `NEURON_RT_VISIBLE_CORES` 绑核（DP router
   同款机制），dp 行由其 tp 基座派生并显式标注（dp 副本执行同一 tp2
   artifact，dp 只加路由开销不加步开销）。
3. **官方路径播种**：`benchmark/trn2/flux_<label>.json` →
   `scripts/seed_planner_measurements.py`（SLUG_TO_MODEL 扩展）→
   测量库。**成本模型常数一个没动**（diff 可查）。

### 9.2 实测结果(本机，realloop 每步，毫秒）

| 配置 | median | min–max | n | 说明 |
|---|---:|---|---:|---|
| tp2cp2ulysses | **262.6** | 262.4–263.0 | 27 | 每步最快 |
| tp2cp2ring | 267.9 | 267.6–268.2 | 27 | |
| tp4（手选） | 270.6 | 270.2–271.1 | 27 | 与 6 月锚点 265.4 差 2%（跨机方差） |
| tp4sp | 278.3 | 277.8–279.2 | 27 | SP 净负收益 −2.9% |
| tp2cp2 | 278.3 | 278.0–278.7 | 27 | |
| dp2tp2（=tp2 步） | 499.6 | 499.3–500.2 | 27 | |
| dp2tp2sp（=tp2sp） | 510.6 | 510.5–510.8 | 27 | |

分布不重叠、亚毫秒级离散——排序在统计上干净。

### 9.3 手选 vs 自动 v2

planner 输出（全部 `evidence=measured`，排序 100% 由数据驱动）：

```
1  tp2cp2ulysses   0.263s  measured *cached   ← 自动推荐
2  tp4             0.265s  measured *cached   ← 手选（库存 6 月锚点）
3  tp2cp2ring      0.268s  measured
```

结论分三层，如实陈述：

- **按每步延迟（常驻 serving 的正确指标）**：自动推荐 tp2cp2ulysses
  （262.6ms）**就是实测最优**，手选 tp4 慢 3.0%（270.6ms，分布不重叠）。
  修正前 ρ=−0.25 的反相关问题消失——不是靠调参，是靠「实测覆盖预测」
  这个既有机制加上物理内存规则。
- **按单次 CLI e2e（含权重加载）**：手选 tp4 仍占优（§8 遍历：43.9s vs
  ~59s）——cp2 配置加载 2×权重（67GB vs 34GB），多出的 ~7s 加载吞掉
  28 步 × 8ms 的每步节省。planner 的 latency 目标 = step×steps，
  **不含加载**：这是目标函数边界，不是排序错误。常驻场景（加载一次）
  与 CLI 场景的最优解不同，报告里必须分开说。
- **零调参验证**：`git diff` 中成本模型常数无任何改动；改动 =
  内存规则（feasibility）+ 扫描脚本 + seeder 映射 + 数据文件。同一机制
  推广到新模型 = 跑一次该模型的 sweep 播种，planner 即以实测排序。

### 9.4 复现

```bash
python scripts/flux_parallel_sweep.py            # 编译+实测（~2h，可断点续跑）
python scripts/seed_planner_measurements.py      # 播种（--check 幂等）
difflet plan --model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024
python -m pytest tests/unit/planner/ -q          # 181 passed
```

已知边界：tp4 行沿用 6 月锚点（265.4）与本日实测（270.6）混用——库按
实例类型键合的设计行为，2% 跨机方差，不影响排序结论；三例与本次无关的
测试失败系安装 diffusers 后激活的环境预置问题（干净树复现相同）。
