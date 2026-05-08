# Session log - 2026-05-08 - M1 closure

## 0. Closure 目标

本 log 用来封闭 M1：确认 Flux 当前可稳定运行的协议，补齐 28-step 性能基线，记录本轮脚本调整，并明确哪些内容已经可以进入 M2，哪些内容仍作为后续风险项保留。

M1 closure 的判断标准不是“所有长期优化都完成”，而是“NovaPipeline + Flux 在 Trainium3 上有一条清晰、可复现、可测试、可继续扩展到 Wan/Hunyuan 的主路径”。

---

## 1. 本轮代码与脚本状态

### 1.1 删除迁移类脚本

`/home/ubuntu/nova/scripts` 里原先的 fork/scaffold 迁移脚本已删除：

- `scripts/_fork_inventory.py`
- `scripts/add_fork_banner.py`
- `scripts/rebase_from_nxdi.py`
- `scripts/rewrite_imports.py`
- `scripts/__init__.py`
- `scripts/__pycache__/`

原因：这些脚本服务于 M0 初始 fork/rebase 搭建，不是当前开发者日常要直接运行的验证入口。保留它们会让 `scripts/` 的语义混乱：读者不知道应该运行迁移工具，还是运行测试工具。

### 1.2 新增可直接运行的验证脚本

新增脚本：

- `scripts/test_imports.sh`：验证 Nova 关键模块能导入。
- `scripts/test_unit.sh`：运行 `pytest tests/unit -q`。
- `scripts/check_quick.sh`：串行执行 import check + unit tests。
- `scripts/flux_smoke.sh`：运行 1-step Flux smoke，默认输出 `/tmp/flux_smoke.png`。
- `scripts/flux_baseline_28.sh`：运行 28-step Flux baseline，默认输出 `/tmp/flux_28step_baseline.png`。

脚本默认优先使用：

```bash
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python
```

并自动设置：

```bash
PYTHONPATH=/home/ubuntu/nova
PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH
TORCH_DISABLE_ADDR2LINE=1
NEURON_RT_LOG_LEVEL=ERROR
NEURON_RT_NUM_CORES=4
```

`scripts/` 已从 `.gitignore` 的忽略列表移除，否则这些可运行入口无法进入版本管理。

### 1.3 导入循环修复

运行 `scripts/check_quick.sh` 时发现一个真实导入问题：

```text
ImportError: cannot import name 'ModelEntry' from partially initialized module 'nova.registry'
```

原因：

- `nova.registry` 导入 `nova.pipeline.parallel_config`
- Python 会先执行 `nova/pipeline/__init__.py`
- 旧的 `nova/pipeline/__init__.py` 立即导入 `NovaPipeline`
- `NovaPipeline` 又导入 `nova.registry`
- 最终形成 partially initialized module 循环

修复：

- `nova/pipeline/__init__.py` 改为懒加载 `NovaPipeline` / `NovaParallelConfig`
- 行为与顶层 `nova/__init__.py` 一致
- 直接导入 `nova.pipeline.parallel_config` 不再触发 `nova.pipeline.nova_pipeline`

验证：

```bash
./scripts/check_quick.sh
```

结果：

```text
ok import nova
ok import nova.registry
ok import nova.pipeline.nova_pipeline
ok import nova.pipeline.compile_cache
ok import nova.pipeline.parallel_config
ok import nova.models.flux.application
13 passed in 0.99s
```

---

## 2. Flux 最终运行协议

M1 最终确认：Flux 默认入口是单 Python 进程 + 多 NeuronCore runtime。

推荐命令：

```bash
NEURON_RT_NUM_CORES=4 \
PYTHONPATH=/home/ubuntu/nova \
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python \
examples/flux_example.py \
  --model black-forest-labs/FLUX.1-dev \
  --tp-degree 4 \
  --skip-warmup \
  --num-inference-steps 28 \
  --prompt "a cat" \
  --output /tmp/flux_28step_baseline.png
```

等价的脚本入口：

```bash
./scripts/flux_baseline_28.sh
```

不要把当前 Flux 默认路径写回：

```bash
torchrun --nproc_per_node=4 examples/flux_example.py ...
```

M1.4 已验证 torchrun MPMD 会把当前 NxDI diffusion traced artifact 带入不兼容的 communicator/load 形态，典型现象包括：

- 每个进程只可见单个 NeuronCore，但 component 尝试加载 rank `0...3`
- TP=4 component 的 global communicator 变成 1
- CLIP/VAE 这类 TP=1 replicated component 出现 world-size/config 不一致

torchrun/MPMD 不是废弃方向，但它不是 M1 的默认成功路径。后续如果要支持，需要单独设计运行协议。

---

## 3. Flux 28-step 性能基线

### 3.1 运行命令

本轮实测使用新脚本：

```bash
./scripts/flux_baseline_28.sh 2>&1 | tee /tmp/nova_flux_28step_baseline.log
```

脚本展开后的关键参数：

```text
model: black-forest-labs/FLUX.1-dev
tp_degree: 4
NEURON_RT_NUM_CORES: 4
height/width: default 1024 x 1024
num_inference_steps: 28
prompt: a cat
dtype: torch.bfloat16
skip_warmup: true
output: /tmp/flux_28step_baseline.png
```

### 3.2 结果

输出文件：

```text
/tmp/flux_28step_baseline.png
/tmp/nova_flux_28step_baseline.log
```

图片确认：

```text
PNG image data, 1024 x 1024, 8-bit/color RGB, non-interlaced
```

核心指标：

| 指标 | 数值 |
|---|---:|
| HF snapshot fetch | 25 files, cache hit |
| compile cache | `/home/ubuntu/.cache/nova/flux/8c331a7a33f60f68` |
| `from_pretrained` | 36.7s |
| 28-step forward | 7.8s |
| denoise 平均吞吐 | 3.78 it/s |
| denoise 稳态吞吐 | 约 3.86-3.87 it/s |

日志关键行：

```text
[nova] from_pretrained done in 36.7s (compile cache at /home/ubuntu/.cache/nova/flux/8c331a7a33f60f68)
100%|██████████| 28/28 [00:07<00:00,  3.78it/s]
[nova] forward done in 7.8s
[nova] image saved to /tmp/flux_28step_baseline.png
```

### 3.3 解释

这次是 cache-hit baseline，不包含 AOT compile 时间。`from_pretrained=36.7s` 主要由以下部分构成：

- HF/diffusers pipeline 组件加载
- T5 / transformer / CLIP / decoder 的 traced artifact 加载
- CPU 侧权重 sharding
- traced model weight initialization

forward 的 7.8s 才是本次 28-step 推理主指标。`--skip-warmup` 用于跳过额外 warmup forward；它不会跳过真实 28-step denoising。

---

## 4. 组件加载形态

最终加载顺序：

1. `text_encoder_2`
2. `transformer`
3. `text_encoder`
4. `decoder`

最终并行形态：

| 组件 | TP | DP | WORLD | 说明 |
|---|---:|---:|---:|---|
| T5 `text_encoder_2` | 4 | 1 | 4 | 真正 TP=4 |
| Flux transformer | 4 | 1 | 4 | 真正 TP=4 |
| CLIP `text_encoder` | 1 | 4 | 4 | replicated component，但 runtime world 仍为 4 |
| VAE `decoder` | 1 | 4 | 4 | replicated component，但 runtime world 仍为 4 |

M1.4 的关键结论是：CLIP/VAE 虽然不是 tensor-parallel 模块，但不能按 `world_size=1` 的 artifact 混入同一个 Flux runtime。它们必须与 TP=4 主 pipeline 保持一致的 WORLD=4，否则 load 时会出现 rank range、parallel state 或 trace config 不一致。

---

## 5. 当前已知警告

baseline 期间仍出现单机 OFI/EFA 相关 warning：

```text
CCOM WARN NET/OFI Failed to initialize rdma protocol
CCOM WARN NET/OFI aws-ofi-nccl initialization failed
CCOM WARN OFI plugin initNet() failed is EFA enabled?
```

判断：

- 对当前单实例 Flux baseline 不阻塞。
- 本次 28-step 完整 forward 已成功。
- 多实例、多节点或 EFA 通信路径需要在后续 M 阶段单独验证。

还出现 `neuronx_distributed.modules.moe.blockwise` 的 optional import warning：

```text
No module named 'neuronxcc.nki._private.blockwise_mm'
```

判断：

- 当前 Flux baseline 未被阻塞。
- 这是 Neuron/NxD 环境里的 optional kernel import warning，不作为 M1 blocker。

---

## 6. M1 exit criteria

| 项目 | 状态 | 备注 |
|---|---|---|
| NovaPipeline 基础 API | 完成 | `NovaPipeline.from_pretrained(...)` 和 `pipe(...)` 已有单测覆盖 |
| compile cache key / manifest | 完成 | dtype alias、python patch version、model_path 排除等回归测试通过 |
| Flux component compile/load | 完成 | cache-hit 加载稳定 |
| Flux 1-step smoke | 完成 | M1.4/M1.5 已跑通 |
| Flux 28-step baseline | 完成 | forward 7.8s，输出 1024² PNG |
| 默认运行协议 | 完成 | 单 Python 进程 + `NEURON_RT_NUM_CORES=N` |
| 日常测试脚本 | 完成 | `scripts/check_quick.sh` 通过 |
| torchrun MPMD | 未完成 | 后续独立设计，不阻塞 M1 |
| HF numerical reference | 未完成 | 建议作为 M1.6 或 M2 前置质量任务 |
| load time 优化 | 未完成 | `from_pretrained=36.7s` 后续可用预分片权重优化 |

M1 可以按工程 closure 关闭。剩余项目是质量/性能增强项，不影响进入 M2 的架构 spike。

---

## 7. 下一步建议

推荐下一步进入 M2，但不要直接复制 Flux：

1. 先做 Wan 2.2 架构 spike。
2. 明确 Wan 的模块边界、输入输出张量、视频 shape、text encoder、VAE/video decoder 关系。
3. 先验证 Wan 是否也能沿用单 Python 进程 + `NEURON_RT_NUM_CORES=N`。
4. 再决定哪些 Flux 抽象可以复用，哪些必须新写。

如果要更严格地封闭 M1，则在 M2 前插入一个短任务：

```text
M1.6 - Flux numerical reference
```

最小目标：

- 固定 prompt/seed/steps/shape
- 跑 HF diffusers reference
- 对比 Neuron 输出的 latent 或 image tensor
- 记录 cosine similarity、mean diff、max diff
- 给出可接受阈值

这个任务更偏质量门禁，不影响当前 M1 主干结论。
