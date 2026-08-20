# AoiZora 两阶段 Planner 原型 — 拓扑感知的 Placement 排序

Date: 2026-08-20

Status: 原型已实现并测试（本文件 Part 4 的决策记录）。

论文：*AoiZora: Topology-Aware Auto-Parallel Optimization for Inference of
Diffusion Transformers*（arXiv 2606.17566）。前一份 planner 文档
（[2026-07-27-parallelism-planner.md](2026-07-27-parallelism-planner.md)）在
Part 3 调研过它并在 Part 6 把「拓扑感知 placement 打分」列为第二版方向；本增量
把这个方向按论文的两阶段结构落成原型。

## 论文方案 → Difflet 的映射

AoiZora 的核心主张：**逻辑切分只说哪些 rank 通信，不说流量落在哪些物理链路上**。
同一个逻辑配置（如 `tp=2 cp=2`）把 TP all-reduce 组放进一个芯片的两个核、还是
拆到跨芯片链路两侧，纯粹取决于 mesh 轴铺到 core ID 上的顺序——在链路不等价的
拓扑上，这会让完全相同的程序付出不同延迟（论文实测 Wan 2.1 在 v5e-8 上同一逻辑
切分不同物理放置差 16.6%，主因是共享链路争用而非跳数）。

| AoiZora（TPU/JAX） | Difflet 原型（Neuron/Trainium） |
|---|---|
| 工作负载轴候选（CP/TP/FSDP/CFG 组合到逻辑 mesh） | `feasibility` 网格（dp/cfg/cp/tp/sp/cp_mode）——已有，即 P2 |
| 阶段 1：编译前 IR + 隐式通信增广，placement 无关打分 Q1，取 top-K | 已有的 `cost_model`（解析、标定锚定、placement 无关）——即 P3 |
| 阶段 2：编译幸存者、解析 HLO 得到具体 collective（类型/载荷/replica groups） | **解析推导** collective 调度（`topology.collective_schedule`） |
| 物理轴序枚举 + 对称性去重 | `topology.enumerate_placements`（活跃轴全排列，按每轴组的芯片归属签名去重） |
| v5e 2D torus 的逐维引擎模型 + ring/line 展开 + 共享链路争用 | Trainium2 层级（核→芯片→设备）三级链路 + 芯片对级链路身份 + 每链路负载求和 |
| Q2 = RS(C_comp^phys + C_comm^phys, C) 排序目标 | `topology.score_placement`：Q2 = C_comp + barrier + max(0, overlappable − C_comp) |
| 输出 (sharding, placement) 并沿标准 JAX/XLA 路径执行 | 输出 (config, placement)；placement 目前**只推荐不执行**（见 D31） |

两个诚实的替换（论文用编译后 HLO，我们用解析推导）：

1. Difflet 每个配置发哪些 collective 是**静态决定**的（plan 和 run 之间没有编译器
   改写），解析调度就是真实调度；
2. AoiZora 编译每个幸存者的代价（秒级）在 Difflet 上是 **~25 分钟/候选**——这正是
   planner 要避开的开销。测试钉住调度字节量与 `cost_model.comm_bytes` 一致，
   两个视图不会漂移。

## 实现

新模块 `difflet/planner/topology.py`：

- **`PhysicalTopology`**：核→(device, chip, core) 坐标；`link_key` 把通信段身份
  定在**芯片对**上（不是核对）——核 {0,2} 和 {1,3} 过的是同一条片间 NeuronLink，
  争用必须按芯片对计。
- **`enumerate_placements`**：活跃轴的内→外全排列；两个序若每个轴的组的芯片
  归属多重集相同则物理等价、去重；默认序（tp 最内）排第一。
- **`collective_schedule`**：每步每 rank 的 collective 列表（family/axis/字节/
  次数/barrier 标志），与 `cost_model.comm_bytes` 同一套账。barrier = TP
  all-reduce（残差加法是循环携带的硬同步，流水藏不住）与步末 CFG gather；
  CP 的 KV gather / ring 轮转 / ulysses A2A 是可重叠的。
- **`score_placement`**：每族通信算**相对默认序的乘数**（物理秒/默认序物理秒），
  绝对值锚在阶段 1 标定上——只有 placement 之间的**比值**依赖层级带宽常数。
  Q2 组合用论文的引擎模型。争用 = 共享链路的并发组负载求和导致的串行化份额。

`planner.plan()` 变两阶段：阶段 1 全场排序（`--survivors K` 截断，默认全保留——
Difflet 搜索空间只有十几个，而 AoiZora 的阶段 2 要编译所以必须剪）；阶段 2 对
每个幸存者枚举 placement、重打分。CLI（`difflet plan`）新增 `--survivors` /
`--no-topology`，文本输出加 placements 块，JSON 每条目带完整 placement 结构。

## Flux 上的验证（trn2.3xlarge，4 核 = 2 芯片）

`difflet plan --model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024`：

- **tp2cp2 的两种轴序差 1.24×**（Q2 0.2508 vs 0.2024）——与论文在 v5e-8 上
  16.6% 的 placement 差距同一量级。默认序（tp 内）把 barrier 的 TP all-reduce
  放在片内链路，可重叠的 CP gather 吃跨片跳 + 4ms 争用；倒过来的序让 AR 乘
  ×4.00（2× 慢层 × 2 组争用）+ 32ms 争用。
- **dp2tp2**：默认序两个副本各自圈在一个芯片内（零争用）；`dp.tp` 序两个副本的
  AR 都过唯一一条片间链路，×4.00 + 64.6ms 争用。
- **结论与论文一致**：本机拓扑下运行时默认序（tp 最内）已是最优——「高流量、
  不可重叠的组保持紧凑」。planner 的价值是把这件事**量化并解释**，并给更大
  主机（tp 跨设备的 trn2.48xlarge）提供同一套排序机制。
- 阶段 2 的重叠结构让 tp1cp4* 的预测从串行和降到 0.170s（CP 通信全部藏进
  计算）；measured 条目（tp4 0.2654s 锚点）保持实测值不被模型改写。

## Part 4 — 决策记录

**D28 — 阶段 2 用解析 collective 调度替代解析 HLO。** 理由见上（静态决定 +
25 分钟编译成本）。代价是拿不到 XLA 可能做的改写/调度信息；在 Difflet 的 AOT
流程里配置即真相，这个缺口可接受。若未来要升级，`collective_schedule` 的返回
类型就是解析器要产出的形状。

**D29 — 层级带宽常数是排序假设，不是测量。** `TIER_BANDWIDTH_BYTES_PER_SECOND`
（片内 : 片内设备 : 片间 = 4 : 2 : 1）只影响 placement 之间的**比值**（乘数），
绝对秒锚在阶段 1 标定上。多层级实测存在后替换常数即可，接口不变。

**D30 — 排序按默认序的 Q2，不按最优序。** `rank_by_best_placement` 默认 False：
非默认轴序**运行时今天执行不了**（`MeshSpec` 的 rank 布局是固定公式），按不可
执行的序去排序会推荐用户跑不了的东西。最优序以差值形式报告（「best order
cp.tp would cut …% — needs a runtime mesh-order flag」）。运行时加了 mesh-order
开关后把这个开关翻 True 就是论文的完整语义。

**D31 — measured 条目不按 placement 比例外推。** 实测值对应默认序；用预测比值
缩放它会「把假设洗成用户当 ground truth 读的数」。measured 条目照样参与
placement 排名（信息全展示），只是 step 时间不被改写。

**D32 — 阶段 1 保持加法模型做剪枝，不引入论文的 Q1 公式做绝对预测。** Q1 的
max 结构进入了阶段 2 的 Q2（overlappable 藏进计算），但阶段 1 的标定拟合
（`T = C·share + bytes/bw` 最小二乘）依赖加法形式，改它会破坏已有锚点语义。

**D33 — 单活跃轴的候选也过 Q2。** `choose_placement` 只在**完全无通信**
（纯 dp）时返回 None；单 placement 候选照样按重叠结构重打分，否则 tp4 和
tp2cp2 会被两套不同的目标排序。

**D34 — ring 的重叠折扣放在秒分配处，不放调度里。** 调度是纯字节量视图
（与 `comm_bytes` 逐字节一致，测试钉住）；`comm_seconds_by_axis` 对 ring 应用
`RING_OVERLAP_RETENTION`，保证「按轴秒数之和 == 平面总秒数」这一标定守恒
（第二个测试钉住）。

**D35 — 非分阶段（co-resident）模型的权重上界从 advisory 升级为硬可行性规则。**
（2026-08-20 晚，遍历数据驱动。）D22 当年保守的理由是「朴素模型在分阶段模型
上高估」（Wan tp2cfg 137.6 GB 却实测通过）——对 `staged=True` 这仍然成立，
advisory 保留。但对 `staged=False`（Flux/LTX-2 整管线常驻），`dp·cfg·cp` 份
副本是**物理上界**：遍历实测显示 planner 当时预测第一的 `tp1cp4*`
（4 副本 ≈135 GB vs 96 GB 设备）根本不可能出现在任何实测里——planner 的
第一推荐是个跑不了的配置。规则上线后，Flux 在 4 核上的可行集恰好收敛为遍历
实际测过的 7 个配置（独立佐证规则物理正确，非调参凑数）。默认沿用
0.85 预算系数（激活/KV 余量）；若未来某个非分阶段格子实测能跑，说明该模型
的常驻字节被高估，修 `WEIGHTS` 表的数据而不是废规则。

**D36 — 实测数据通过官方 seeder 播种，不手改 measurements.json。**
`scripts/flux_parallel_sweep.py` 按仓库 flux 方法论（真实 28 步 generate 的
逐步间隔、弃首次预热、弃 step 0）逐配置产出
`benchmark/trn2/flux_<label>.json`，`scripts/seed_planner_measurements.py`
（SLUG_TO_MODEL 扩展了 7 个 flux_<label> slug）把它蒸馏进测量库。planner 的
「实测覆盖预测」机制是 D19 预留的路径：排序由数据驱动，**成本模型常数一个
都没动**——这是对「不倒果为因」约束的结构性保证。

## 已知限制 / 后续

- **placement 只推荐不执行**（D30）。落地需要 `MeshSpec` 支持轴序参数 +
  NxD 组构造跟随，是一个独立的运行时增量。
- **多设备 torus 被压平成一个片间层级**（`neuron-ls` 不暴露维度）；trn2.48xlarge
  上需要真正的逐维模型（论文 §4.4 的 per-dimension engine 调度）。
- 4 核上所有可双轴配置的默认序恰好最优；placement 搜索的**判别力**要到
  tp 跨设备的主机（16/64 核）才真正发挥，届时应播种多层级带宽实测替换 D29。
- 阶段 2 的 A2A 用 ring 规则近似（每链路载 per-rank 字节）；对 cp=2/cp=4 的
  小组这是合理的一阶，更大的组需要 per-pair 路由模型。

## 测试

`tests/unit/planner/test_topology.py`（25 例）：几何/层级、链路芯片对身份、
placement 枚举与去重、调度-平面模型字节一致性与秒守恒、barrier/overlappable
分类、×4.00 争用乘数（tier×sharing 分解断言）、Q2 藏匿与不藏匿、纯 dp 无
决策、planner 集成（survivors 截断、`--no-topology` 等价旧行为、measured
不改写）。全套 `tests/unit/planner/` 180 passed。
