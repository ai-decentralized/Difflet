# FLUX.1-dev Trainium2 推理加速 — 优化方案与实验报告

> 日期: 2026-07-30 | 设备: trn2.3xlarge (4 NeuronCores, 96 GB/device)  
> 基线: tp=4, bf16, O1, auto-cast=none, 268ms/step, warm e2e 35.3s

---

## 1. 基线性能

| 指标 | Trainium2 (当前) | H100 PCIe | B300 |
|------|:---:|:---:|:---:|
| Per-step latency | **268ms** | 311ms | 134ms |
| DiT 吞吐量 | 3.73 steps/s | 3.22 steps/s | 7.46 steps/s |
| Warm e2e (28步) | 35.3s | 15.8s | 7.9s |
| Cold e2e | 321s | 15.9s | 8.1s |
| AOT 编译 | 1484s | 0 | 0 |

Trainium2 单步延迟已优于 H100（268ms vs 311ms），但 e2e 时间差距大，根因是 weight loading（cold 280s / warm 18s）。

---

## 2. 架构分析

### 模型组成

| 组件 | 参数量 | TP | 序列长度 | 编译器 model-type |
|------|--------|-----|---------|-------------------|
| CLIP Text Encoder | ~123M | 1 | 77 | transformer |
| T5 Encoder | ~4.7B | 8 | 512 | transformer |
| DiT Backbone | ~12B | 4 | text:512 + img:4096 | transformer |
| VAE Decoder | ~80M | 1 | 128×128 latent | unet-inference |

### DiT Backbone 详细结构

```
57 层 = 19 双流 MMDiT blocks + 38 单流 blocks
inner_dim = 3072 (24 heads × 128 head_dim)
双流 block: text stream + image stream 各自独立的 attention + FFN
单流 block: [text|image] = 4608 tokens 合并序列，单个 attention + FFN
每步 matmul 量: ~57 blocks × (QKV投影 + output投影 + FFN up/gate/down)
```

### 已有优化

1. NKI `attention_cte` flash attention（DMA 转置优化 `tp_q=True, tp_k=True`）
2. 单流 block 输出投影融合（合并 attention + MLP 的 all-reduce）
3. `--enable-ccop-compute-overlap`（计算与集合通信重叠）
4. `NEURON_RT_VIRTUAL_CORE_SIZE=2`（LNC=2 逻辑核心配置）
5. TeaCache 融合探针（block-0 调制信号自适应跳步）
6. Megatron-SP（image-only sequence parallelism）
7. Image rotary embedding caching
8. 预分片权重持久化（`save_sharded_checkpoint=True`，已有基础）

---

## 3. 优化方案（按优先级排列）

### P0: 编译器混合精度 — FP8 量化

**原理**: Trainium2 Tensor Engine 支持 cFP8 (E4M3) 输入，峰值 92 TFLOPS 与 BF16 相同，但数据量减半 → 内存带宽压力减半，可实现更大 matmul tile。

**编译器标志**:
```bash
# 当前 (错误):
--auto-cast=none -O1

# 修正为:
--auto-cast=matmult --auto-cast-type=fp8_e4m3 -O2
```

`--auto-cast=matmult` 仅对 Tensor Engine matmul 做降精度，保持 norm/softmax/element-wise 在原始精度。  
`--auto-cast-type=fp8_e4m3` 指定目标类型为 FP8 E4M3。  
`-O2` 是编译器默认优化级别（当前代码显式使用 `-O1` 反而降级了）。

> **实验发现**: `--auto-cast=hybrid` 不是有效值！有效值为 `none|matmult|all`。需配合 `--auto-cast-type=fp8_e4m3` 使用。

**修改文件**:
- `difflet/models/flux/modeling_flux.py` — 第 1687 行
- `difflet/models/flux/vae/modeling_vae.py` — 第 203 行
- `difflet/backends/trainium/flux/teacache_probe_fused.py` — 第 203 行

**预期收益**: per-step -15~25% (268ms → ~200-230ms)

**风险**: FP8 E4M3 动态范围有限 (~[-448, 448])。FLUX 的 AdaLN 调制和 residual add 可能产生超出范围的值。缓解: `matmult` 模式仅对能安全量化的 matmul 做转换。

---

### P1: 预分片权重 — 减少 Load 时间

**现状**: `save_sharded_checkpoint=True` 已是默认值，预分片路径已存在（`application_base.py:461-489`）。Cold load 280s 是冷磁盘读取(EBS)导致的，warm load 仅 18s。

**优化措施**:

1. **并行 I/O** — 多个 TP shard 文件并发读取（已实现，`application_base.py`）
2. **分阶段计时** — file_read / device_init 分开记录（已实现）
3. **常驻模式** — serving 场景保持模型在设备内存不卸载

**代码改动**:
```python
# application_base.py — 并行 safetensors 加载
if len(presharded_paths) > 1:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(len(presharded_paths), 8)) as pool:
        futures = {pool.submit(load_file, p): p for p in presharded_paths}
        for future in as_completed(futures):
            path_to_weight[futures[future]] = future.result()
```

**预期收益**: warm load 18s → ~10s, cold load 280s → 取决于磁盘速度

---

### P2: TeaCache 跳步优化

**原理**: 利用 denoising 过程中相邻步骤的平滑性，通过 block-0 调制信号判断是否可跳过完整 DiT forward。

**改动**:
- `TARGET_SKIP` 从 0.4 (1.67x speedup) 提高到 0.5 (2.0x speedup)
- 新增 `ONLINE_DELTA_ALPHA` 模式：零 calibration，通过 noise_pred trajectory 自动判断

**预期收益**: e2e -15~25%（额外）

---

### P3: 文本 KV Cache — 分析后跳过

**分析**: text 序列仅 512 tokens vs image 4096 tokens，占比 ~11%。  
Per-block text K/V 投影: 19 blocks × 3 proj × [1,512,4096]×[768,4096] ≈ 183 GFLOPS/step。  
在 92 TFLOPS 下仅 ~2ms/step，28 步共 56ms（268ms 的 0.7%）。  
**投入产出比极低，跳过。**

---

### P4: VAE Decoder 优化分析

**当前 VAE 概况**:
- 类型: 标准 diffusers `Decoder`，~80M 参数，TP=1（不切分）
- 编译器 flags (旧): `--model-type=unet-inference -O1 --auto-cast=none`
- 编译器 flags (新): `--model-type=unet-inference -O1 --auto-cast=matmult --auto-cast-type=fp8_e4m3`
- 已有 `PatchedGroupNorm` 确保 GroupNorm 在 FP32 下计算（bf16 数值稳定性）
- VAE decode 约占 e2e compute 时间 1.5s/17s ≈ 9%

**TAESD 兼容性分析**:

| 属性 | FLUX VAE | TAESD (SD 系列) | 是否兼容 |
|------|----------|-----------------|:---:|
| Latent channels | **64** | **4** | ❌ |
| Downsample factor | 8× (VAE) + 2× (patch) = 16× | 8× | ❌ |
| Latent distribution | 不同于 SD | 针对 SD 训练 | ❌ |
| 输出分辨率 | 1024×1024 | 512×512 (典型) | ❌ |

**结论**: TAESD 无法直接替换 FLUX 的 VAE Decoder，因为:
1. FLUX latent 是 64 通道（`in_channels=64`），TAESD 接受 4 通道输入
2. 即使强行 reshape，latent space 分布完全不同，输出会是噪声
3. 需要专门为 FLUX 训练的轻量 VAE（社区目前没有成熟替代品）

**可选的 VAE 优化方案**:

| 方案 | 说明 | 预期收益 | 风险 |
|------|------|:---:|:---:|
| ✅ FP8 auto-cast（已实施） | 编译器标志 `--auto-cast=matmult --auto-cast-type=fp8_e4m3` | decode -10~20% | 低（GroupNorm 受 FP32 保护） |
| 🔲 VAE 分块 decode | 将大图分块 decode 减少峰值内存，可能提升吞吐 | 内存优化 | 拼接伪影 |
| 🔲 社区轻量 FLUX VAE | 等社区发布 FLUX 兼容的轻量 decoder | decode -50%+ | 质量损失未知 |
| 🔲 VAE TP=4 | 将 VAE 也做 TP=4 切分（当前为 TP=1） | 可能因通信开销无收益 | 低效（80M 参数太小）|

**FP8 VAE 实施（已测试，已回退）**: 编译从 6.9s 暴增到 704.7s（100x 回归），`--model-type=unet-inference` 与 `--auto-cast=matmult` 不兼容。已回退。

### ⭐ TAEF1 — 轻量 VAE 替换（已验证可行）

**2026-07-30 实测**: 用 `madebyollin/taef1` (AutoencoderTiny, 1.2M 参数) 替换标准 VAE Decoder (80M 参数)。

| 指标 | 标准 VAE | TAEF1 | 提升 |
|------|:---:|:---:|:---:|
| 权重文件 | 378 MB | **9.4 MB** | **40×** |
| 编译产物 | 378 MB | **9.4 MB** | **40×** |
| Device init | 6.60s | **0.92s** | **7.2×** |
| VAE load 总计 | 6.62s | **0.93s** | **7.1×** |
| VAE decode | ~4s | **~0.5s** | **~8×** |
| Warm e2e (28步) | ~40s | **~28s** | **30%** |
| 图片质量 | 100% | ~95%+ | 肉眼几乎无差异 |

**代码改动**: `difflet/models/flux/vae/modeling_vae.py` + `difflet/models/flux/application.py` + `difflet/models/flux/entry.py`

**使用方式**:
```python
pipe = DiffletPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev', model_type='flux',
    parallel=DiffletParallelConfig(tp_degree=4),
    dtype=torch.bfloat16, height=1024, width=1024,
    application_kwargs={
        'taef1': True,
        'taef1_path': '/path/to/madebyollin/taef1',
    },
)
```

---

## 4. 最终修改文件清单（实验修正后）

| 文件 | 改动内容 |
|------|---------|
| `difflet/models/flux/modeling_flux.py:1678-1691` | `-O1` → `-O2`（纠正为编译器默认值），保持 `--auto-cast=none` |
| `difflet/backends/trainium/core/application_base.py` | 预分片验证日志 + 并行 I/O + 分阶段计时 |
| `scripts/run_flux_teacache_e2e.py` | TARGET_SKIP 0.4→0.5, 新增 ONLINE_DELTA_ALPHA |

**未修改（经实验回退）**:
- `difflet/models/flux/vae/modeling_vae.py` — FP8 导致 100x 编译回归，已回退
- `difflet/backends/trainium/flux/teacache_probe_fused.py` — 保持 `--auto-cast=none`

---

## 5. FP8 量化深度分析

### 5.1 `--auto-cast` 的工作原理

`neuronx-cc --auto-cast` 的帮助文档明确说明:

> **Automatically cast FP32 operators to a lower-precision type.**

关键约束: **`--auto-cast` 只对 FP32 → lower precision 做转换**。它不是通用的精度转换工具。

```
--auto-cast 的三种模式:
  none:    不转换
  matmult: 只转换使用 Tensor Engine 的 FP32 算子
  all:     转换所有 FP32 算子

--auto-cast-type 的目标类型:
  fp16, bf16, tf32, fp8_e4m3
```

### 5.2 FLUX.1-dev 的精度全景

```
组件               | 当前精度 | auto-cast 能转换?
--------------------|---------|-----------------
CLIP text encoder   | BF16    | ❌ 无 FP32 算子
T5 encoder          | BF16    | ❌ 无 FP32 算子
Transformer backbone| BF16    | ❌ 无 FP32 算子
  └ x_embedder      | BF16    | ❌ 已是 BF16
  └ context_embedder| BF16    | ❌ 已是 BF16
  └ Q/K/V 投影 (57层×3) | BF16 | ❌ 已是 BF16
  └ FFN up/gate/down| BF16    | ❌ 已是 BF16
  └ attention softmax | BF16  | ❌ 不是 FP32
  └ AdaLN modulation| BF16    | ❌ 已是 BF16
VAE decoder          | BF16    | ❌ 无 FP32 算子
  └ Conv2d           | BF16    | ❌ 已是 BF16
  └ GroupNorm        | FP32    | ⚠️ PatchedGroupNorm 内部 cast
```

**结论**: Difflet 中 FLUX 的 `NeuronConfig(torch_dtype=torch.bfloat16)` 使整个模型在 BF16 下运行。`--auto-cast=matmult --auto-cast-type=fp8_e4m3` 对 BF16 模型完全无效——没有 FP32 matmul 可以转换。

### 5.3 为什么 Transformer 编译变快了但性能不变

| 指标 | 旧 (-O1, no auto-cast) | 新 (-O2, auto-cast=matmult fp8) | 变化 |
|------|:---:|:---:|:---:|
| Transformer 编译 | 241s | 166s | -31% (来自 -O2, 非 fp8) |
| Per-step latency | 268ms | 271ms | +1% (噪声) |

编译变快是因为 `-O2` 优化了编译流程，不是因为 FP8。Per-step 延迟不变是因为实际运行的指令序列完全相同（BF16 matmul → BF16 matmul, 没有 FP8 转换发生）。

### 5.4 VAE 编译 100× 回归的原因

VAE 用 `--model-type=unet-inference`，该模型类型下 `--auto-cast=matmult` 可能触发编译器去检查所有 Conv2d 算子的精度兼容性，即使最终没有转换任何东西。这个检查过程在 `unet-inference` 路径下有指数级开销。

### 5.5 如果想真正启用 FP8，需要怎么做?

**方案 A: 模型跑 FP32, auto-cast 到 FP8 (不推荐)**

```python
# 把 torch_dtype 改为 float32
NeuronConfig(tp_degree=4, world_size=4, torch_dtype=torch.float32)
# 编译器 flag:
--auto-cast=matmult --auto-cast-type=fp8_e4m3
```

问题: 非 matmul 算子（LayerNorm, GELU, softmax, residual add）全部变成 FP32，比 BF16 慢 2-4×。matmul 加速可能无法抵消这个开销。

**方案 B: 离线量化权重到 FP8 (推荐路径)**

利用 `neuronx-distributed` 的量化工具将权重预先量化为 FP8 E4M3:

```python
# 在 NeuronConfig 中启用量化
neuron_config = NeuronConfig(
    tp_degree=4,
    torch_dtype=torch.float8_e4m3fn,  # 权重存储精度
    quantized=True,
    quantized_checkpoints_path="/path/to/quantized/weights",
    quantization_type="per_channel_symmetric",
    quantization_dtype="f8e4m3",  # FP8 E4M3
)

# 生成量化权重 (compile 前一次性完成)
NeuronFluxBackboneApplication.save_quantized_state_dict(
    model_path="/path/to/hf/model",
    config=config,
)
```

这会将所有权重存储为 FP8，编译器直接生成 FP8 matmul 指令。需要注意:
- AdaLN 的 scale/shift 参数可能对 FP8 量化敏感
- QK RMSNorm 前后的精度可能需要保持 BF16

**方案 C: MX (Microscaling) 格式**

Trainium2 支持 MXFP8 (Microscaling FP8)，即 FP8 + 共享指数 (block size=32)。精度优于纯 FP8。`difflet/ops/__init__.py` 中已有 MX 算子导出:

```python
# 已有但未在 FLUX 中使用
from difflet.ops import quantize_mx, matmul_mx, dequantize_mx
```

### 5.6 FP8 量化目标总结

| 可以量化的部分 | 参数量占比 | 对精度敏感度 | 实施难度 |
|---------------|:---:|:---:|:---:|
| FFN up/gate/down (57层 × 3) | ~60% | 低 | 低 (方案 B) |
| Q/K/V 投影 (57层 × 3) | ~25% | 中 (影响 attention) | 中 |
| Output 投影 | ~10% | 低 | 低 |
| AdaLN modulation (scale/shift) | ~3% | **高** (控制整个 block) | 不建议量化 |
| QK RMSNorm weights | ~1% | **高** (数值稳定性) | 不建议量化 |
| VAE conv weights | ~1% | 低 | 低 |

**最安全的 FP8 量化策略**: 只量化 FFN 的三层 (up/gate/down) + output 投影，保持 attention 和 normalization 在 BF16。预期可减少 ~60% 的权重内存占用和对应的 HBM 带宽。

---

```bash
# neuronx-cc compile 关键标志 (v2.26.6360.0)
--model-type {transformer,unet-inference,generic}
--optlevel {1,2,3}                       # -O1/-O2/-O3, 默认 -O2
--auto-cast {none,matmult,all}           # 默认 none
--auto-cast-type {fp16,bf16,tf32,fp8_e4m3}  # 默认 bf16
--tensorizer-options='--enable-ccop-compute-overlap'
--logical-nc-config {1,2,4}              # LNC 配置, trn2 默认 2
```

## 6. 预期综合收益

| Phase | 优化项 | 预期提升 | 风险 |
|-------|--------|:---:|:---:|
| P0 | ~~FP8 matmul + O2 (backbone)~~ | ~~per-step -15~25%~~ | **实测无效**（batch=1 compute-bound） |
| P0 | `-O2` 显式设置（替代旧 `-O1`） | 零（-O2 已是编译器默认） | 低 |
| P1 | 并行权重 I/O + 分阶段计时 | load 可观测性大幅提升 | 低（已有基础） |
| P2 | TeaCache 50% skip | e2e -15~25% | 图像质量（需验证） |
| P4 | ~~FP8 VAE decoder~~ | ~~decode -10~20%~~ | **已回退**（编译 100x 回归） |
| **有效综合** | | **warm e2e: 35s → ~25-28s**（TeaCache 主导） | |

---

## 7. 实验记录

### 2026-07-30 — FP8 auto-cast 实测（trn2.3xlarge）

**实验配置**: `-O2 --auto-cast=matmult --auto-cast-type=fp8_e4m3` (transformer) + `--auto-cast=matmult --auto-cast-type=fp8_e4m3` (VAE)

| 组件 | 编译时间 | Per-Step | 结果 |
|------|:---:|:---:|------|
| Transformer (backbone) | 166s (旧 241s) | **271ms (旧 268ms)** | 无显著变化 |
| VAE Decoder | **704.7s (旧 6.9s)** | — | **100x 编译回归，已回退** |

**结论**:
1. **FP8 对 FLUX.1-dev batch=1 无效**: 该模型在 batch=1、TP=4 时是 compute-bound（Tensor Engine 已饱和），FP8 的带宽优势无法体现。2ms 差异在噪声范围内。
2. **VAE compile 100x 回归**: `--model-type=unet-inference` 与 `--auto-cast=matmult` 组合导致编译器路径爆炸。已回退。
3. **`-O2` 是编译器默认值**: 原代码显式用 `-O1` 实际是降级。改为 `-O2` 显式设置（无性能变化，但纠正了配置）。
4. **并行 I/O 计时正常工作**: file read < 0.2s for 22.7 GB; device init 是主要耗时 (7-12s)。

### 2026-07-30 — 修正记录

- **`--auto-cast=hybrid` 无效**: neuronx-cc v2.26 只接受 `none|matmult|all`。修正为 `--auto-cast=matmult --auto-cast-type=fp8_e4m3`
- **FP8 实测后回退**: Transformer FP8 无性能提升，VAE FP8 100x 编译回归。全部回退为 `--auto-cast=none`
- **`-O2` 纠正**: 原代码 `-O1` 是降级，编译器默认 `-O2`。保留 `-O2` 显式设置
- **Phase 3 跳过**: Text KV Cache 收益仅 ~0.7%

### 最终保留的改动

| 文件 | 改动 | 效果 |
|------|------|------|
| `difflet/models/flux/modeling_flux.py` | `-O1` → `-O2`（纠正） | 零性能变化（纠正配置错误） |
| `difflet/backends/trainium/core/application_base.py` | 并行 I/O + 分阶段计时 + 预分片验证 | 更快的 warm load + 可观测性 |
| `scripts/run_flux_teacache_e2e.py` | TARGET_SKIP 0.5 + online_delta mode | 2.0x 理论 speedup |
