# 从计算图分析到高质量 kernel 的 agent 流程（TPU 先行，推广到 Trainium）

2026-11-12。状态：规划 + 分析工具第一版（`benchmark/tpu_profile/roofline.py`、
`benchmark/tpu_profile/dataflow.py`，未 commit）。FLUX demo 等 TPU。

## 1. 出发点

参考 MPK（OSDI'26，Mirage Persistent Kernel）：把整个模型编成一个 persistent
megakernel，在 GPU 的 SM 之间按 task graph 调度，消除 kernel launch、kernel 间
barrier 和算子间空转。

在 TPU 上前提不同：

- XLA 本来就是一个 executable 跑整张图，没有逐 kernel launch；v5e 每芯片一个
  TensorCore（静态调度的 VLIW 流），没有多 SM 并发可调度 → **不需要动态调度**。
- XLA 的 memory-space assignment 已经做了跨算子预取，并把能放下的中间结果放进
  VMEM（trace 里的 `S(1)`）。相邻 XLA fusion 之间的"kernel barrier"基本不存在。
- 真正的边界损失来自 Pallas custom-call：它对 XLA 不透明。GELU epilogue 实验
  （2026-11-11）：kernel 内省掉 60 ms，下游 RS matmul 因为读的不再是 `S(1)` 慢了
  65 ms，净收益为零 → 撤销。（见 §6 的待验证点：原因可能是 `pl.ANY` 输出，而非
  所有 Pallas 输出。）

所以 TPU 版的价值不在运行时调度，而在**分析与判断**：哪里是 XLA 结构上做不到的、
哪里不该碰。生成 kernel 只是其中一小步。

## 2. 实测依据（v5e-4，均为同树背靠背 A/B）

| 改动 | 类型 | 结果 |
|---|---|---|
| AG collective-matmul Pallas kernel（tile 级 all-gather × matmul 流水） | kernel：XLA 做不到的跨芯片重叠 | Wan step 292.4 → 268.0 ms |
| Wan SP 序列 pad 到整 tile（4680 → 4736） | 图改写：对齐 | 269.2 → 262.2 ms |
| split-halves RoPE（q/k 每头通道置换，host 上置换权重） | 图改写：消除 relayout | 262.2 → 249.2 ms |
| RS kernel（matmul × reduce-scatter） | kernel | op 级部分 shape 略快、FFN down +12%；e2e 262.2 → 262.7 ms，不默认开 |
| GELU epilogue 融进 AG kernel | kernel：XLA 已能融合的 elementwise | 250.4 vs 249.4 ms（噪声内），撤销 |

结论：最大收益来自图改写和"XLA 做不到的通信重叠"；op 级的赢不等于 e2e 的赢。

Wan 上界粗算（480×832×9，S=4680）：每 image token 约 12B 参数 → linear ≈ 1.1e14
FLOP，self-attn ≈ 1.8e13，合计 ≈ 1.3e14 FLOP/step；4 × 197 TFLOP/s → 下限约
165 ms。TP 360.7 ms ≈ 46% MFU（与 10-27 profile 的 47% 吻合），SP 249.2 ms ≈ 66%。
手算，待 `roofline.py` 在新 trace 上核对。

## 3. 流程设计

1. **分析**：jax.profiler trace → 每个 HLO op 的时间、`model_flops`、
   `bytes_accessed`、`hlo_category`、`long_name`（含 layout / `S(1)`）、`source`
   （PyTorch 源码行）。都是编译器自己的数，trace 自带。
   - `roofline.py`：每 op `bound = max(flops/peak, bytes/HBM)`，gap = 实测 − bound；
     按 bucket 汇总（matmul / attention / cm-kernel / comm / relayout / elementwise /
     idle）；按 (op, shape, 源码行) 列 top gap。comm、relayout 的 bound = 0（理想是
     隐藏或删除）；`bytes_accessed` 含 VMEM 操作数，bound 截断到不超过实测。
   - `dataflow.py`：从 HLO 行的操作数重建每个 module 的数据流图；compute 消费者沿
     transit（comm / relayout）回溯到 compute 生产者，边代价 = 路上 transit 时间；
     标出操作数来自 VMEM 还是 HBM；列出每个 custom-call 边界的输入来源、输出位置和
     下游速度。
2. **决策**（按 region，单位是包住 gap 源码行的 `nn.Module.forward`）：

   | gap 类型 | 修法 |
   |---|---|
   | PyTorch 写法导致的 relayout | 改写法（split RoPE、pad 对齐） |
   | 暴露的通信 | kernel / 图级分解做通信-计算流水 |
   | XLA 已融合的 elementwise | 不碰 |
   | matmul TFLOP/s 偏低 | 先查 shape / tile，再考虑 kernel |
   | Pallas 边界丢 `S(1)` | 吞并邻居或不切这个边界 |

3. **生成**：LLM 基于模板（AG/RS ring 骨架、emit_pipeline 骨架）写 Pallas，经
   `jax_op` 接入；Mosaic 报错（如 "Offsets along tiled dimensions must be aligned"）
   回喂。
4. **验证**（硬门槛，逐层）：
   1. forward hook 抓真实输入输出作 fixture，interpret 模式比数值 + race detector；
   2. 芯片上与 PyTorch reference 比 bit-exact / 误差（阈值按 bf16 噪声底）；
   3. op 级 bench；
   4. 同树背靠背 e2e A/B（`bench_case.sh`），PSNR + 种子扩展。
5. **安装**：替换 module forward + env 开关（类似 `DIFFLET_TPU_SP_KERNEL`），可 A/B、
   可回退。

### TPU 上的硬约束（生成时强制）

- rank 之间程序必须相同：rank 相关偏移以 tensor 传入，不能是 Python 常量（否则各
  rank 程序分叉 → 错值或卡死，见 2026-11-10 根因）。
- Mosaic 只接受 tile 对齐的 DMA / 切片（bf16 行数 16 的倍数）；interpret 模式抓不到。
- torch_tpu 编译缓存按 jax_op 名 + 签名做 key：改代码必须改名（`_source_tag`）。
- `has_side_effects=True` 会让 torch_tpu SPMD 报错。
- 验收看 e2e，不看 op 级。

## 4. 工具验证（已知答案校准）

在 10-27 的 Wan block 编译 trace（TP 版本，`story/raw/.../trace_block_compile`）上：

- `roofline.py`：MFU 49.2%（当时手算 48%）；comm 28.3%（当时结论 all-reduce 28%）；
  matmul 170 TFLOP/s，attention 168 TFLOP/s；top relayout 是
  `reshape f32[4680,10,64,2] @ wan/modeling_wan.py:498`（交错 RoPE，后来被 split
  RoPE 消除）。
- `dataflow.py`：最贵三条边都是 `matmul(o_proj, linear.py:187) → all-reduce
  45.7 MiB → elementwise(modeling_wan.py:651/655/661)`，各约 8.3 ms，正是后来
  AG/RS kernel 的位置。
- 修过的问题：第一版把 XLA 的 `ConcatBitcast` 误判为 matmul kernel（算出荒谬的
  TFLOP/s）→ 改为精确匹配 `custom_call_target="tpu_custom_call"` + `[N, K]` 权重
  结构识别。
- 局限：只有约 31% 的边能回溯到生产者（`get-tuple-element` / `bitcast` / 参数不执行，
  不在 trace 里）→ 需要 HLO dump 补全。

## 5. FLUX demo 计划

选 FLUX.1-dev 的理由：迭代最快（serve 就绪 130 s）；有已知线索（`flux/rope.py` 用
`weight.copy_` 在设备上置换，Wan 上同样问题每次 forward 拷权重约 50 ms；RS kernel
在 FLUX shape 上慢 6%）；答案未知。当前 step 140.6 ms（TP 175.2）；粗算每 token 约
6.4B 参数、S=4608 → 下限约 94 ms，MFU 约 67%（手算，待核对）。

Wan 作校准集：退回 `868401eb`（无 pad / split RoPE），看 agent 能否自己找回约 −7%。

步骤：
1. `bench_case.sh`（`DIFFLET_TPU_SP=1` + `DIFFLET_BENCH_TRACE_DIR`）采 28 步 trace，
   同时 dump HLO。
2. `roofline.py --steps 28`、`dataflow.py --steps 28`。
3. 对 top region 分类 → 选一个 kernel 类 region 走生成 + 验证。

## 5.1 FLUX 第一轮结果（2026-11-12，v5e-4，IP 136.87.185.129）

trace `/mnt/models/profile/trace-flux-1112`（+ HLO dump `hlo-flux-1112`）：window
139.0 ms/step，**MFU 68.2%**（下限 94.7 ms，与手算 94 ms 吻合）。gap 分解：cm-kernel
7.0、relayout 5.7、attention 3.4、comm 2.8、matmul 1.8、idle 1.8 ms；elementwise
21.9 ms 已贴带宽 roofline，按源码拆开后最大一项是 AdaLN modulation。

工具修复：AG kernel 的 HLO 行跨行（正则需 `DOTALL`），且首个操作数是 SMEM 的 s32 表
（x 改为"第一个 rank≥3 且有匹配 [N,K] 权重的操作数"）。

### 候选与处理

| 候选 | 证据 | 估计 | 状态 |
|---|---|---|---|
| AdaLN modulation GEMV 在每个 rank 上读完整权重 | 4 个 `convert_reduce_fusion`，输入 18×`bf16[18432,3072]` 等，~750 GB/s；`tp_sharding.py` 不切 norm.linear | 8.1 ms/step | **已做**：`DIFFLET_TPU_FLUX_SHARD_MODULATION`（默认开） |
| text 流小 kernel：AG 64 行半块 ~62 TF/s；JAX RS ring `[1,128,1536]` permute 暴露 | roofline top gap + 边视图最贵边 | ~5–6 ms | 待做（kernel 类：text/image 合一次调用） |
| AG 输出 → attention heads layout 的 relayout | `collective_matmul_pallas.py:263` reshape/copy | ~4 ms | 待做 |
| `flux/rope.py` `copy_` 每次 forward 拷权重 | 权重形状 copy 0.05 ms/step | — | **否定** |

### modulation 分片 A/B（同树背靠背，`/mnt/models/profile/mod_ab.sh`，tree md5 2fb0e3b90ac2）

| | `=0` | `=1` |
|---|---|---|
| bench step（synced 均值） | 143.3 ms | 132.6 ms |
| trace window（natural，28 步） | 139.0 ms/step | 131.7 ms/step（−7.3，−5.2%） |
| MFU | 68.2% | 71.9% |
| `convert_reduce_fusion` 合计 | 241.2 ms | 58.8 ms（正好 1/4） |

- `=0` 与当日 baseline（flux-1112）输出 sha 相同（噪声底：同设置 bit-exact）。
- `=0` vs `=1`：PSNR 38.7 dB，mean |d| 0.81/255，0.56% 像素 |d|>16，差异集中在边缘；
  目视一致。
- 数值探针（`benchmark/tpu_profile/modulation_probe.py`，单芯片 compiled，真实权重
  `transformer_blocks.0.norm1.linear`、`single_transformer_blocks.0.norm.linear`）：
  复制版与 4 分片版输出**逐元素相同**（0 / 27648 个不同），与 fp64 误差相同
  （rms 0.24% / 0.25%，即 bf16 输出舍入）。→ e2e 差异不来自 modulation 值本身，
  推测来自图中融合方式变化导致的下游 bf16 中间舍入，经 28 步放大（与 FLUX TP vs
  kernel 39.5 dB 同一量级）。**未验证**：逐 block 定位差异来源；多种子；1 步漂移。
- CPU 单测：`tests/unit/backends/test_tpu_flux_sharded_modulation.py`（tp=2/4，
  float64，误差 ≤1e-12）。
- 副作用：每 rank 少存约 4.8 GB 复制权重（未测 HBM 读数）。

### text 流改 TP（图改写尝试，已撤销）

想法：text 只有 512 token，在 joint block 里不切序列，改回普通 TP——q/k/v 与 FFN up
为本地列并行 matmul（产出与 AG 相同的"全部行 × 1/tp 列"），输出投影与 FFN down 走
TP all-reduce；进入 single block 前用每 rank 的行号 buffer `index_select` 取本地行
（保持各 rank 程序一致）。开关 `DIFFLET_TPU_FLUX_TEXT_TP`。

A/B（同树，`/mnt/models/profile/text_ab.sh`，均开 modulation 分片）：

| | text0 | text1 |
|---|---|---|
| trace window | 130.8 ms/step | 129.7 ms/step（−1.1） |
| MFU | 72.4% | 73.0% |

- 噪声底：text0 与 fm-mod1 同设置，输出 sha 相同，窗口 131.7 vs 130.8 → ~0.9 ms。
  −1.1 ms 在噪声内。
- trace_diff（28 步合计）：AG kernel −140 ms（每步 −5.0，符合预估），但 text 的
  `distributed_all_reduce` 是同步 op、完全暴露 +61 ms，整块 text matmul +45 ms，
  `add_convert_fusion` +14 ms → 净 −32 ms。
- PSNR text0 vs text1 43.9 dB。
- 结论：省下的 AG 低效被暴露的同步 all-reduce 吃掉。**已撤销**（树回到 md5
  2fb0e3b90ac2）。text 流的真正修法回到 kernel 类：text 与 image 合进同一次 AG kernel
  调用（混合行数），或让 text 的归约与 image 计算重叠。

### heads-first attention（图改写，保留）

原 TP processor（照搬 diffusers：`[B,S,H,D]` 是 FlashAttention 系 GPU 库的接口 layout，
在 GPU 上 unflatten/permute 只改 stride）在 `[B,S,6,128]` 上做 norm、拼接、RoPE；TPU 的
(8,128) tile 下 6 填不满、与两端（AG 输出 `(rows,768)`、attention custom-call 的
`operand_layout_constraints` `(S,128)`）都不一致 → 一串 `reshape [4608,6,128,1]` /
`copy [1,4608,6,128]`。改为拆 heads 后立即转 `[B,H,S,D]`（D=128 = tile 宽，每个
tile 恰属一个 head），其后全在 (S,128) tile 上。TPU 专用 `_attention_heads_first`，
共享 processor 未改；开关 `DIFFLET_TPU_FLUX_HEADS_FIRST`（默认开）。CPU 单测
`tests/unit/backends/test_tpu_flux_heads_first.py`（joint/single × 两种 RoPE，float64，
≤1e-12；去掉 RoPE 的变异版 4/4 失败）。

A/B（同树，`/mnt/models/profile/hf_ab.sh`，均开 modulation 分片）：

| | hf0 | hf1 |
|---|---|---|
| trace window | 131.6 ms/step | 128.7 ms/step（−2.9，−2.2%） |
| MFU | 71.9% | 73.6% |
| relayout（reshape+copy，28 步合计） | 119.9 ms | 56.9 ms（每步 −2.25） |

- 噪声底：hf0 与 ft-text0 / fm-mod1 同设置，输出 sha 相同；三次窗口 131.7 / 130.8 /
  131.6 → ~0.9 ms。−2.9 ms 超出噪声。
- 剩余 relayout ≈ 2.0 ms/step，与"每 block 只剩 v 与 attention 输出两次独立转置
  （各 ~17 µs）"的估算一致（未逐条核对）。copy-done +0.47 ms/step；loop fusion 重新
  分配（multiply_convert −45 / add_convert +44 ms）。
- PSNR hf0 vs hf1 45.1 dB（mean |d| 0.43/255）。

### RoPE 去副本（保留）与 GELU 进 ring（撤销）

heads-first 之后重跑分析（trace-fh-hf1）：elementwise 前两项是 GELU（~3.8 ms：FFN-up
输出过一遍 GELU 写回，RS ring 再读）和 RoPE（~3.4 ms：diffusers 的
`x*cos + cat([-x_hi, x_lo])*sin` 把旋转副本写出再读回）。

- RoPE：`_rope_split_halves` 按半区直接算两半输出，浮点运算相同（取负精确）；CPU 上
  float32/bf16 与 diffusers 逐位相同。开关 `DIFFLET_TPU_FLUX_FUSED_ROPE`（默认开）。
- GELU：给 RS ring 加 `gelu=`，在 ring 内对每块行做 GELU。4 芯片 parity：与环外 GELU
  相对 L2 0.084%，与普通集合通信 0.30–0.32%（原 ring 0.32–0.34%）。

三路 A/B（同树，`/mnt/models/profile/rg_ab.sh`）：

| | base | +RoPE | +RoPE+GELU |
|---|---|---|---|
| window | 129.7 | 128.2 | 130.6 ms/step |
| **busy** | **127.1** | **126.5（−0.56）** | **127.8（+1.3）** |
| MFU | 73.0% | 73.8% | 72.5% |
| 输出 | — | 与 base 逐位相同 | PSNR 45.1 dB |

- 噪声：base 与 fh-hf1 同设置、输出相同；window 128.7 vs 129.7（host 空闲波动），busy
  均为 3558.9 ms → **busy 是更低噪声的判据**。
- RoPE：旋转副本相关 fusion −61 ms（28 步），XLA 重新拆出的 fusion/pad +45 ms，净
  busy −0.56 ms/step，保留。
- GELU 进 ring：GELU fusion −106 ms，但 matmul fusion +104 ms、新增
  `bitcast_add_fusion` +40 ms → 净 +1.3 ms/step。原因：tanh-GELU 是超越函数，融进
  matmul 后该 fusion 受 VPU 限制；且每块行与左右两半权重各乘一次，GELU 算了两遍。
  **已撤销**（`collective_matmul.py` 回到 HEAD）。

### 当日累计（FLUX，v5e-4，SP + AG kernel）

最终树（tree md5 8adc392202fa，全部默认开关，`final_check.sh`）：window 139.0 →
128.4 ms/step（−7.6%），busy 137.2 → 126.5，MFU 68.2% → 73.7%；输出与 rg-rope 逐位
相同（sha b18f1224fa31），与当日起点 PSNR 39.85 dB。保留：modulation 分片（−7.3）、
heads-first（−2.9）、RoPE 去副本（busy −0.56）。撤销：text 流 TP 化（噪声内）、GELU
进 ring（+1.3）。

### attention：换 kernel 不值得（op 级否决）

FLUX 现用 torch_tpu 把 `F.scaled_dot_product_attention` 降成的内置 flash kernel
（trace 名 `scaled_dot_product_cudnn_attention`，block 不可调），85% 峰值、gap 3.4 ms。
候选：`DIFFLET_TPU_PALLAS=1` 的 JAX Pallas flash（block 可调）。单芯片 probe
（`benchmark/tpu_profile/attn_block_probe.py`，shape [1,6,4608,128]，compiled，50 次链式
调用取中位）：torch_tpu SDPA 402 µs（162 TFLOP/s）；Pallas flash 最好 1536/1536/1536
438 µs（149 TFLOP/s），512/512/512 488 µs，block_k_major 须整除 4608，2304/2304/768
VMEM 不够。→ 不换，未做 e2e。附带发现：对 fp32 参考的最大相对误差 SDPA 1.9e-2，
Pallas flash 7.4e-3（未深究）。

### text 流 AG kernel：权重常驻 VMEM（kernel 改动，保留）

诊断（读 kernel 源码）：tile 编号是 `hop * tiles_per_hop + tile_in_hop`，每一跳把全部权重
从 HBM 重新读一遍（ring_size 倍）。image 流（512 行半块）计算多、被掩盖；text 流（64 行
半块）计算少，搬权重占大头。"text/image 合并调用"的原假设不对，改为权重常驻。

改动：`all_gather_matmul_body(resident_weights=)`——开头一次发出所有权重 tile 的 DMA、
各进各的槽，第 0 跳等到齐，后续跳复用；`unsupported_reason` 按整份权重算 VMEM。
`collective_matmul` 里 `DIFFLET_TPU_SP_AG_RESIDENT_ROWS`（默认 0，关）：每 rank 行数
≤ 阈值且放得下才常驻，否则照旧；阈值进 op key 和 op 名（`_res128`）。FLUX install
`cm.set_resident_rows_default(128)`。

验证：
- interpret 模式（4 CPU 设备，race detector 开）：streamed/resident × batch 1/2 ×
  等宽/混合宽度，6/6 通过；`unsupported_reason` 断言通过（在节点上跑）。
- op 级（`cm_pallas_bench.py`，FLUX 形状）：wall 时间被 host 调用开销淹没（qkv 0.246 vs
  0.253 ms）；读 trace 的 device 时间：text q|k|v 80.4 → 67.3 µs（−16%），FFN up
  105.3 → 87.8 µs（−17%），输出与 dus ring 相同。比预估少：streamed 模式下一块权重的
  读取本就和当前计算重叠。
- e2e A/B（`res_ab.sh`）：busy 126.5 → 125.4 ms/step（−1.1），window 128.2 → 127.1，
  MFU 73.8% → 74.5%，输出逐位相同（sha b18f1224fa31）。res0 与 final-1112 输出相同、
  busy 相同（噪声底）。
- 默认开关最终树（tree md5 aaadfe0ea9ae，`final2-1112`）：busy 125.4，window 127.2，
  MFU 74.4%，输出与 final-1112 / rs-res128 逐位相同。

### 当日累计（更新）

FLUX window 139.0 → 127.2 ms/step（−8.5%），busy 137.2 → 125.4，MFU 68.2% → 74.4%。
保留 4 项：modulation 分片（−7.3）、heads-first（−2.9）、RoPE 去副本（busy −0.56）、
text 流 AG 权重常驻（busy −1.1，第一个 kernel 改动）。

### 再试的 4 项（2026-11-12 夜）

| # | 候选 | 结果 | 处理 |
|---|---|---|---|
| 1 | text 流 RS（≤512 行）改走 Pallas RS kernel（`DIFFLET_TPU_SP_RS_KERNEL_ROWS`） | busy 125.4 → 126.8（+1.4）：ring 暴露通信 −27 ms、matmul −39 ms，但 RS kernel +101 ms（同样每跳重读权重） | 撤销 |
| 3 | q/k RMSNorm + RoPE 合成一次 f32 计算 | busy +2.5，PSNR 35.6 dB：XLA 拆出更多 fusion，多出 copy/reshape | 撤销 |
| 2 | AG kernel 直接写 heads-first 的 q/k/v | busy −1.4，**输出逐位相同**（见下） | 保留 |
| 4 | GELU 融进 AG kernel epilogue | 未做 | — |

### #2 AG kernel 输出 heads-first（保留）与"数据对不上"的排查

改动：`all_gather_matmul_body(head_dims=)`，每个 tile 拆成 tile_width/d 次 DMA 写到各自 head；
`collective_matmul` 的 `all_gather_matmul(..., head_dims)`（op 名加 `_hf…`，JAX ring 回退同样
输出 heads-first），FLUX `DIFFLET_TPU_FLUX_KERNEL_HEADS`（默认开，仅在 heads-first attention 开时）。

第一版输出写成 6 维 `(B,H,ring,2,half,d)` 再 reshape：busy −1.4 但 **PSNR 41.1 dB**。排查：
1. 芯片上 kernel 输出（`cm_pallas_bench --heads 128`，FLUX 三种 shape）与 dus ring 逐位相同 →
   kernel 无误。
2. `norm_layout_probe.py`（单芯片，同一份 bf16 q，旧 layout 对比新 layout，逐阶段逐位比较）：
   image（4096）、single（4608）全部相同；**text（512，64 行半块）**：bias 加法相同，f32 平方均值
   1512/3072 不同（相对 3.4e-7，1–2 ulp），RMSNorm 输出 3/3.5M 个元素差 ≤2 bf16 ulp → 经 57 block ×
   28 步放大到 41 dB。原因：XLA 看到 6 维输出后的 reshape，64 行维上的归约结果按 `f32[6,4,2,64]`
   `T(2,128)` 铺 tile，求和顺序不同。
3. 同实验改用普通 4 维 `(B,H,S,d)` 输入：512 与 4096 全部逐位相同。

修复：kernel 直接输出 4 维 `(B,H,S,d)`，每块 DMA 到行偏移 `(owner*2+direction)*half`
（运行时值，`pl.multiple_of(...,16)`；要求半块行数是 16 的倍数，否则仍用 6 维）。interpret 8/8
（测试行数改 32 以覆盖 4 维路径），芯片逐位检查 3/3。

A/B（`hk2_ab.sh`，tree md5 774c0cc67dc2）：busy 125.4 → 124.0（−1.47 ms/step），window
127.2 → 125.7，MFU 74.4% → 75.3%，**输出与关闭时逐位相同**（sha b18f1224fa31），reshape −54 ms。

### 当日累计（再更新）

FLUX window 139.0 → 125.7 ms/step（−9.6%），busy 137.2 → 124.0，MFU 68.2% → 75.3%。保留 5 项：
modulation 分片、heads-first attention、RoPE 去副本、text AG 权重常驻、AG kernel 输出 heads-first。

### TPU 经验（本轮新增，每条有 A/B 证据）

- custom-call（含 `pl.ANY` 输出的 Pallas kernel）的输出可以被 XLA 放进 `S(1)`。
- torch_tpu 的 `distributed_all_reduce` 是同步 op，不会与独立计算重叠。
- 超越函数（tanh-GELU）融进 matmul fusion 会让它变成 VPU 受限，可能比单独一遍更慢。
- GPU 上免费的 view（`[B,S,H,D]` unflatten）在 TPU 上是物理 relayout；D=128 时
  heads-first 让每个 tile 恰属一个 head。
- 判定收益用 busy，不用 window。
- 小 shape 的 kernel 用 op 级 wall 时间测不准（host 调用开销淹没），读 trace 的 device 时间。
- 环形 AG kernel 在行数少时被"每跳重读权重"拖慢；权重放得下就常驻 VMEM。RS kernel 同理。
- Pallas 输出的形状会影响下游 XLA 的归约顺序：交出普通 N 维数组，别交高维数组再 reshape。
- "只改 layout"也要逐位验证：先查 kernel 本身，再逐阶段比下游，差 1 个 f32 ulp 也会在 28 步后变成 ~40 dB。

## 5.2 其他模型（2026-11-13）

五个模型 baseline（v5e-4，SP，当日树）：FLUX 125.7 ms/step MFU 75.3%（已优化）；Qwen 173.0 /
65.3%；Wan 249.0 / 71.0%；LTX-2 189.1 / 48.4%；HunyuanVideo 476.5 / 61.9%。工具：大 trace 的
JSON（≤5M 事件）会丢掉全部 device 事件，改用 xprof `hlo_stats`（`benchmark/tpu_profile/xprof_ops.py`，
独立 venv）转出 `hlo_stats.json` 再跑 roofline；在 FLUX 上两条路径逐类一致。

共性问题（上限，ms/step）：A. RoPE/attention 的 [B,S,H,D] layout（HV ~30、Qwen ~15、Wan ~5）；
B. 每步 host 往返的 idle（LTX-2 24.4、Wan 11.9、HV 11.8）；C. 小流通信暴露（LTX-2 ~20、Qwen 8、
HV 4.3）；D. tanh-GELU；E. modulation GEMV（HV 6.1、Qwen 4.2）。个别：HV attention 带 mask 只有
153 TF/s（gap 32.7）、LTX-2 ring 拼接的 dynamic-update-slice ~12、HV RS matmul 119 TF/s、Wan
cross-attn q kernel 124 TF/s。

### device loop（撤销）

想法：Wan/LTX-2/HV 的 DiT 封装每步 `.cpu()` 拷回、scheduler 在 host 上算，这次同步让 host 开销
（`.cpu()` 2.4、UniPC 1.7、Dynamo guard 3.4、H2D ~2.5 ms）全部串行、device 空等 ~10.5 ms/步。改法：
包住 orchestrator 的 `_denoise`，latents（fp32）与 embedding 一次放上 device，scheduler 随之在 device
上算，结束拷回一次。

A/B（Wan，同树，`loop_ab.sh`）：trace 窗口 248.3 → 254.0 ms（+5.7），idle 11.1 → 16.8；bench 同步
step 249.5 → 557.4 ms；PSNR 44.0 dB。原因：scheduler 在 device 上是 eager 执行，每步几十个小算子
逐个下发（trace 每步多 29 个 custom-other），比 CPU 上 1.7 ms 慢得多，host 仍在关键路径上。**撤销**
（代码存档于 session scratchpad）。要消这段 idle 需另想办法：把 scheduler 一步编进图、或减少 guard
检查与每步 H2D。

### HunyuanVideo heads-first + split-halves RoPE（保留）

共享模块 `backends/tpu/core/heads_first.py`（FLUX、Qwen 已改为调用）。HV：dual/single block 的 q/k 通道
在 host 上置换一次，AG kernel 写 [B,H,S,D]，norm、RoPE、joint attention（同样的 mask bounds）在该 layout
上；single block 只在走 SP forward 时置换。开关 `DIFFLET_TPU_HV_HEADS_FIRST`（默认开）。CPU 测试：
dual（有/无 mask）与 single（3 block 串联）对共享实现 float32 量级一致（diffusers RMSNorm 方差用
float32、置换改变求和顺序），去掉置换的变异版失败。

A/B（`one_model_ab.sh`，device loop 两组都关）：window 477.2 → 431.5 ms/step（−45.7，−9.6%），busy
464.7 → 419.7，MFU 61.8% → 68.4%；relayout 49.5 → 12.2，elementwise 58.8 → 51.5；PSNR 44.4 dB；
关闭组与 baseline 逐位相同。

### HunyuanVideo modulation 切分（保留，默认开）

TPU 上 HV 的 AdaLN modulation linear 一直是复制的（neuron/trainium 早已切分）：每个 rank 每步读全部
80 个、6.8 GB，三个 GEMV fusion。改为在 `build_module` 调共享的 `shard_modulation_linears`（列切 +
gather 输出），开关 `DIFFLET_TPU_HV_SHARD_MODULATION`（默认开）。

A/B（`mod_hv_ab.sh`，同树 b1d85c74b299，key mask 两组都关）：window 431.4 → 423.9 ms（−7.5，−1.7%），
busy 419.7 → 413.0，MFU 68.4% → 69.6%；全部来自 elementwise 51.45 → 44.74（其 roofline 26.07 → 19.65，
即 HBM 字节确实少了），comm +0.34；attention / cm-kernel / matmul 不变。PSNR 45.5 dB（gather 顺序）；
两次切分运行（hv-km0、hv-sm1）输出逐位相同。注意：一开始把 hv-km0（默认开）与 hv-sm0（显式关）的 7 ms
差当成了跨运行噪声——比较前先核对每组的全部 env 默认值。

### Qwen-Image：AG kernel + 常驻权重（不启用）

window +1.1 ms、busy +1.4、comm −8.0：image 流 kernel 178 TF/s 对 ring 186；text 流 108–122 TF/s——
每跳 compute（~17–24 µs）与传输（~17 µs）相当，kernel 的逐跳硬同步盖不住。

### LTX-2：AG kernel（待调参后定）

`lk-base` → `lk-ker`：window 189.7 → 187.4（−2.3），busy 164.7 → 162.9；relayout −8（ring 的
dynamic-update-slice 没了）、comm −9.5，但 kernel 在窄 shape 上慢。每调用对两个下界（通信：3 跳 × 2.37 MB
÷ 46.7 GB/s ≈ 152 µs；计算：2·2312·4096·Σw ÷ 197 TF/s）：

| Σw | kernel | 计算下界 | 瓶颈 |
|---|---|---|---|
| 512 | 189 µs（51 TF/s） | 49 µs | 通信 |
| 2×512 | 195 µs | 98 µs | 通信 |
| 1024 | 203 µs | 98 µs | 通信 |
| 3×1024 | 341 µs | 295 µs | 计算（87%） |
| 4096 | 444 µs | 393 µs | 计算（88%） |

每跳 compute/transfer ≈ 2·Σw·link_bw/flops，与 rows 无关：决定 kernel 与 ring 的是输出宽度。窄 shape 的
gather 是纯通信，kernel 阻塞着等；ring 的 async permute 能与周围的算子重叠（`lk-base` 每次只暴露
~10 µs）。tile/resident 救不了窄 shape。AG 调参上限 ~11 ms/step，现实 3–6。

### 按 shape 选参数：方案 3 的基础设施（默认不改行为）

`ops_impl/cm_config.py`：trace 时按 (chip, shape) 解析 AG/RS 的参数——tuning table
（`DIFFLET_TPU_CM_TUNING`，JSON）→ cost model（`DIFFLET_TPU_CM_COST_MODEL=kernel|tile|1`，默认关）→
调用方默认（今天的行为）。table 的 hash 进 op 名与 cache key；`prepare` 检查各 rank 加载同一 table；
`DIFFLET_TPU_CM_LOG_SHAPES` 记录每次解析，作为 sweep 的输入。

- `kernel`：每跳 compute ≥ 1.3 × transfer 才用 kernel，否则 ring；权重逐跳重读比该跳 matmul 贵且放得下时
  常驻。
- `tile`：照 torch_tpu flash attention 的 `CreateTiling`（编译时按 VMEM 容量从大往小减半，直到解析式
  估算放得下；DMA 块与计算子块 kc 分开）——在 `max_tile` 内取能整除各宽度、且 kernel 自己的 VMEM 估算
  （`all_gather_vmem_bytes`）不超过预算（容量 × 3/4，v5e 96 MiB）的最大 tile。`max_tile` 是唯一要
  sweep 校准的数（`lk_tune.sh`：tile 128/256/512/1024 + 消融）。kernel/ring 的选择照搬不了——
  attention 没有通信。

### Wan heads-first（保留，默认开）

`wan/sp.py`：`_global_rms_norm_heads_first`（tp>1 时对 (1,3) 求和再 all-reduce）+ heads-first attention，
开关 `DIFFLET_TPU_WAN_HEADS_FIRST`。A/B（`wan_hf_ab.sh`，同树）：window 248.3 → 244.2 ms（−4.1，−1.7%），
busy 237.2 → 233.4，MFU 71.2% → 72.4%；relayout 8.99 → 3.72（−5.3），elementwise +1.4（norm）。
20 步 PSNR 只有 27.6 dB（seed 42），但 `wan_drift.sh`（`sp_step_drift`）1 步 50.5 dB、4 步 49.2 dB——
与 11-11 kernel 对 ring 的模式一样（每步一致、20 步在 seed 42 上放大到 26 dB），是轨迹放大，不是计算
差异。baseline 与更早的 mt-wan 逐位相同。

### Qwen-Image heads-first（保留，默认开）

`qwen_image/sp.py` 用共享的 `core/heads_first.py`。A/B（`next_ab.sh`，同树 bc4405e08519）：window 173.0 →
159.4 ms（−13.6，−7.9%），busy 172.2 → 158.7，MFU 65.3% → 70.9%；relayout 13.74 → 3.43，elementwise −2.5。
PSNR 50.9 dB。关闭组与早先的 qk-base 逐位相同。

### LTX-2：窄 shape 走 ring 的 tuning table（不采用）与 table 的 graph break

第一次 `lt-tab`：step 190.9 → 232.2 ms，trace 里 25 320 个 module（`lt-ker` 60）、idle 26 → 115 ms。原因：
每次 forward 的 `_op()` 调 `table_tag()`，有 table 时它 `open()` 文件——Dynamo 在每个 op 处断图。修复：
tag 在 `prepare()` 里算一次存进 `_STATE["tuning"]`（回归测试 `test_op_lookup_reads_no_table`）；无 table
时程序不变（lt-ker 与 lt-ker2 逐位相同）。

重跑（同树 4293a09bb0c0）：`lt-ker2` window 186.9 → `lt-tab2` 189.5（+2.6），busy +1.7；cm-kernel −28.1，
但 ring 的 matmul +15.4、relayout（dynamic-update-slice）+8.7、comm +5.8。微基准里窄 shape 的 ring 只快
10–20 µs/次，端到端的 DUS relayout 与小 matmul 效率把它吃掉。结论：同一模型里不混用；`cm_config` 的
`kernel` 部分（只比每跳 compute 与 transfer）判 ring 是错的，不能打开，除非把 ring 的拼接代价算进去。
LTX-2 全 kernel ~187 ms（三次 186.9–187.4）对全 ring 189.7：用户决定 LTX-2 默认**不用** AG kernel（保持 ring）。

### HunyuanVideo broadcast key mask（作废）

hv-km1（mask bounds 只给 1 行 query）编译完成后 4 个 worker 全部停在 futex，13 分钟无进展，手动终止。
HV 走 fused flash 的 key-bounds 路径，本来就把每行 bounds 用 amin/amax 压成一个窗口，broadcast 什么也
省不了，所以 HV 改回逐行 bounds；broadcast 只留给走 SDPA 的 LTX-2。原样重跑 hv-km1b 同样卡住（编译后
4 个 worker 10 s 内 1 个 CPU tick）：确定性 hang。根因未查（怀疑各 rank 程序不一致，`hlo_rank_diff.py`
可验）。最终全模型测试里 LTX-2（当时默认 broadcast）同样在编译后卡死（fin-ltx_2）。设备上从未成功过、收益小：**功能已删除**（`key_bound_rows` / `DIFFLET_TPU_BROADCAST_KEY_MASK` 撤回，`attention.py` 与 `ltx_2/sharding.py` 回到 HEAD），LTX-2 以 fin-ltx_2b 重跑。

### AG kernel 的 tile 与消融（`lk_tune.sh`，LTX-2 shape，ms/调用）

tile 128/256/512/1024：qkv 0.418/0.411/0.407/0.418，FFN 0.527/0.520/0.518/0.531，窄 shape 不变或变慢；
ring(dus) qkv 0.446、FFN 0.500、n512 0.196、n1024 0.211。tile 不是瓶颈（512 最好，比 256 快 1–2%；
`max_tile` 默认 512 与之相符，`tile` 部分不单独做端到端 A/B）。消融（tile 256）：宽 shape 去掉通信时间
不变（通信已被盖住），但 kernel 的 matmul 比 XLA 单芯片同 FLOPs 慢 22–30%（qkv 0.411 vs 0.316，FFN
0.520 vs 0.425）；窄 shape 只剩通信就要 0.19–0.21，比 ring 还慢。假设：kernel 每跳每方向只算 rows/2
（289）行，每块权重复用的行数是 XLA 的 1/8 → `rows_probe.sh` 用 XLA 按 289×8 / 578×4 分块对照验证；
成立则把同一跳两个方向的 halves 合成一个 matmul。

### 全模型终测（2026-11-13 05:45–07:05，同树，各模型默认开关）

`final_all.sh` + `fin_ltx2b.sh`（树 `Difflet-tune`，broadcast key mask 已删），每模型一次带 trace 的 bench，对比今早 baseline（trace window，ms/step；FLUX 28 步按 20 步跑的 roofline 已折算）：

| 模型 | baseline window / MFU | 终测 window / MFU | 变化 | bench step（今早 → 终测） | PSNR vs 今早 |
|---|---|---|---|---|---|
| FLUX | 139.0 / 68.2% | 125.8 / 75.2% | **−9.5%** | 145.0 → 127.6 | （逐项已测，38.7–45 dB） |
| Qwen-Image | 173.0 / 65.3% | 159.4 / 70.9% | **−7.9%** | 180.1 → 164.4 | 50.9 dB |
| Wan 2.1 | 249.0 / 71.0% | 244.6 / 72.3% | −1.8% | ~249.5 → 246.4 | 27.6 dB（1/4 步 50.5/49.2：轨迹放大） |
| HunyuanVideo | 476.5 / 61.9% | 424.7 / 69.5% | **−10.9%** | 476.9 → 426.4 | 41.0 dB |
| LTX-2 | 189.1 / 48.4% | 190.8 / 48.0% | 0（无改动） | ~191 → 192.0 | 逐位相同 |

终测各模型的 bucket 与对应单项 A/B 的"开"组一致（Qwen = qh1，Wan = wan-hf1，HV = hv-sm1），改动合在一起没有互相抵消。LTX-2 与今早 baseline 逐位相同，说明删掉 key mask 后该路径回到原样。剩下最大的空间：LTX-2 idle 26 + comm 24 + relayout 24 ms；HV attention gap 32.7 ms；各模型 AG kernel 内部 ~10% 开销（`rows_probe`）。

### LTX-2 第二轮（2026-11-13 20:00 – 11-14 04:00，同树 A/B，trace window ms/step）

起点 `la1`/`sb1`：window 189.1，bench 190.2，MFU 48.4%。结果：**173.1（−8.5%），bench 174.3，MFU 52.9%，输出与起点逐位相同**。

| 项 | run | window | 结论 |
|---|---|---|---|
| a2v/v2a 共用一次 gather（`SHARED_CA_GATHER`，默认开） | `sg1`/`ab0` | 189.1 → 181.0 | **保留**。per-batch modulation (B,1,D) 与行 gather 可交换：一次 XLA all-gather 取代两个 AG ring；parity 与关闭时相同（0.00475） |
| 跳过 Dynamo guard（`SKIP_GUARDS`，默认开，仅已跑过的输入签名） | `ig1` → `sk1` | 181.1 → 173.1 | **保留**。compiled frame 每次调用先评估 guard（几千个参数/属性），~8 ms/step host 时间，device 空等；`set_stance(skip_guard_eval_unsafe=True)`，0 次 recompile，逐位相同 |
| audio all-reduce 改 jax `psum` | — | 不变 | 撤销：降成同一条 HLO all-reduce |
| audio all-reduce 改 ppermute ring | `ab2` | 180.5（≈0） | 撤销：elementwise +2.6、与 RS ring 抢链路 +1，且 PSNR 34.9 dB |
| attention 只算有效 key（3a） | `at3a` | 180.9 | 撤销：trace 逐位相同，XLA 已把 slice+pad 融合 |
| heads-first（hf1 / hf2） | `hf1`/`hf2` | +11 / +12.6 | 撤销：ring 输出被排成 `{2,0,1:T(2,128)}`，生出 8.5 ms/step 的 strided bias+slice fusion |
| ring 行对齐到 16 | — | +17..20 | 撤销：LTX-2 的 [2312,512] dynamic-update-slice 对齐与否都 ~7.4 µs（整 buffer copy） |
| 常量输入（text embedding 等）上传缓存 | `ci2` | 173.0（未命中） | 撤销：pipeline 每步 `.to(bf16)` 都生成新 tensor；要命中得改共享 `pipeline.py`（未改），预计还能省 4–5 ms |

`sk1` 剩余 idle 16.5 ms/gap：D2H 1.4、`.float` 1.7、scheduler 1.0、两份 embedding H2D ~4.6、`_dit_call` 4.3（peft ~1.6、torch_tpu wrapper ~2.2）、PJRT launch ~2.4。comm 21.8 / relayout 17.5 受 XLA 行为所限。其他模型代码路径未变（`ring_padded_length` 只是把 Wan 的算式搬进 `collective_matmul`，逐值验证相同）。

## 6. 待验证

- **custom-call 输出与 `S(1)`**：已有反例——FLUX trace 里 AG kernel（`pl.ANY` 输出）
  的部分输出就在 `S(1)`（如 `bf16[1,4,2,512,768]`）。所以"Pallas 输出必须在 HBM"
  不成立；GELU 实验中 FFN-up 输出没进 VMEM 更可能是大小（~33 MB）所致。待更正 PR #153
  计划文档里的说法（需用户同意）。
- FLUX `rope.py` 是否每次 forward 拷权重。
- AG kernel 的 q/k/v 输出是否也丢了 `S(1)`。

## 7. 推广到 Trainium

### 直接可搬（与硬件无关）

上界分析、gap 分类与决策顺序、边视图、reference + fixture 验证、同树 A/B 门槛、
已知答案校准。

### 必须在 trn 上重新验证

| TPU 结论 | Trainium 上的问题 |
|---|---|
| 单 TensorCore 单指令流 | NeuronCore 有 Tensor / Vector / Scalar / GpSimd 多引擎并发、semaphore 同步：roofline 要按引擎看利用率和空闲 |
| XLA 自动放 VMEM（`S(1)`），Pallas 边界可能丢 | SBUF / PSUM 显式管理；NKI kernel 边界是否强制 HBM 往返、是否切断编译器融合 → 需 A/B |
| `T(8,128)` 对齐 | SBUF 128 partition、Tensor Engine 的转置要求；"pad 到整 tile"思路可搬，数字要换 |
| Pallas 核内 RDMA 做 tile 级通信重叠 | 当前 SDK 的 NKI 能否核内通信待查；不能则用图级分解（matmul 切块 + 每块异步 collective） |
| torch_tpu 缓存按 op 名；rank 常量致程序分叉 | NEFF 缓存键、rank 相关常量的后果待查 |
| 一轮编译数分钟 | neuronx-cc 更慢 → 单 block 小探针迭代，选迭代快的模型更重要 |

### 工具分层

- **共享层**：统一 op 记录格式（bucket、时长、flops、字节、源码行、生产者/消费者）；
  roofline 表、边视图、fixture 抓取、A/B 脚本。
- **后端适配层**：trace 解析（TPU：xplane / JSON；trn：neuron-profile 导出）、峰值与
  带宽、bucket 规则、kernel 接入（`jax_op` + Pallas ↔ NKI custom op）、约束清单。
- **后端经验库**：每条结论附 A/B 证据；trn 的从零积累。

当前 `roofline.py` / `dataflow.py` 的解析与计算耦合在一起；推广前先把解析抽成 TPU
适配器。

### trn 第一步

校准：选已知答案的案例（如 TP 下 Wan block 的 all-reduce 占比），验证 neuron-profile
适配器 + roofline 能复现手工数字。涉及 Trainium 与其他后端共用的文件时，先给 diff
再改。

## 8. 下一步

1. TPU 到手：FLUX trace + HLO dump → 两份报告。
2. `pl.ANY` vs BlockSpec 输出的 `S(1)` 小实验。
3. 选一个 FLUX region 走完整的生成 + 验证闭环。
4. 抽出 trace 解析适配层，为 trn 做准备。
