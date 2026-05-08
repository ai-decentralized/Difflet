# Nova — Trainium3 Diffusion 推理框架（fork-and-own 版）

## Context

**问题**：团队拥有 AWS Trainium3 + 完整 Neuron 工具链，想做一个**像 xDiT 那样精瘦聚焦、但完全 Trainium-native** 的 diffusion 推理框架。需要"承包"整个 diffusion-on-Neuron 这一层——长期维护、深度优化，不受上游 release cadence 制约。

**最终选择（多轮讨论后定型）**：
1. **定位**：xDiT-like，focused、Python API + 明确的 Neuron 运行协议，不堆 serving 抽象（无 scheduler、无 HTTP、第一版无 CLI）。M1 实测修正：Flux 当前稳定入口是单 Python 进程 + `NEURON_RT_NUM_CORES=N`，不是 torchrun MPMD。
2. **写法**：严格沿用 NxDI 的范式（subclass HF Pipeline、`Application` = `nn.Module` composing 子组件、AOT `compile()` + `load()`、用 NXD parallel layer + nkilib kernel）
3. **来源**：从 NxDI **fork**（不依赖）相关代码到我们 repo，Layer 3 工具链保持 pip 依赖
4. **工作负载**：diffusion-only，模型范围 = Flux + Wan2.2 T2V/I2V + HunyuanVideo + HunyuanVideo-1.5 + Qwen-Image + LTX-2 + Z-Image
5. **并行**：TP + CP + CFG（跟 NxDI 走，第一阶段不做 Ulysses）

**关键代码侦察结果**（已验证）：
- NxDI 已有完整 Flux 端到端实现：`models/diffusers/flux/{pipeline,modeling_flux,application}.py` + `clip/` + `t5/` + `vae/`，~2173 行
- NxDI Pipeline 模式：`NeuronFluxPipeline(FluxPipeline)` 仅覆盖 `__call__`，其余走 `super().__call__()`
- NxDI Application 模式：`NeuronFluxApplication(nn.Module)` 内 `self.pipe = FluxPipeline.from_pretrained(...)`，然后**替换四个组件** (`text_encoder`/`text_encoder_2`/`transformer`/`vae.decoder`) 为 `NeuronXxxApplication`，提供 `compile()` / `load()` / `__call__()`
- NxDI 模型层使用 NXD parallel layer：`ColumnParallelLinear` / `RowParallelLinear` / `LayerNorm` + `CustomRMSNorm` + `gather_from_*` / `reduce_from_*` / `scatter_to_*`
- Attention：调 `nkilib.core.attention.attention_cte(q, k, v, scale, causal_mask, tp_q, tp_k, tp_out)`，硬件感知通过 `NEURON_RT_VIRTUAL_CORE_SIZE` env 切 `attention_cte[2]` vs `attention_cte`
- AOT 编译/运行时分离：`compile(workdir)` 一次写到磁盘，`load(workdir, start_rank_id, local_ranks_size)` 多次 SPMD 加载——不可消除的 Trainium 硬约束
- Diffusion 在 NxDI 当前**没有任何 console script / 统一 CLI**，用户写 Python 脚本。M1 Flux 硬件验证表明当前 NxDI diffusion artifact 的 load 更适合单进程多 NeuronCore runtime；torchrun MPMD 作为后续多机/多进程研究项保留，不作为 Flux 默认入口。

---

## Fork 边界（已与用户确认）

```
┌──────────────────────────────────────────────────────────┐
│ Layer 1：NxDI 的 diffusion 专属代码  → ✅ Fork 进 Nova    │
│   neuronx_distributed_inference/models/diffusers/        │
│     {flux/, activations.py, embeddings.py,               │
│      normalization.py, padder.py}    ~5K 行              │
├──────────────────────────────────────────────────────────┤
│ Layer 2：NxDI 共用 inference 基础设施 → ✅ Fork 进 Nova   │
│   models/{application_base,config,model_wrapper,         │
│     layer_boundary_marker,encoder_base}.py               │
│   modules/{attention/, custom_calls.py,                  │
│     checkpoint.py, padding.py}                           │
│   utils/{hf_adapter,diffusers_adapter,distributed,       │
│     runtime_env,compile_env,snapshot,accuracy}.py        │
│   ~5–10K 行（attention_base.py 单文件 2488 行）          │
├──────────────────────────────────────────────────────────┤
│ Layer 3：NXD / nkilib / nki / 编译器 → ❌ 保持 pip 依赖   │
│   neuronx_distributed.{parallel_layers,kernels,modules,  │
│     trace,quantization,operators}                        │
│   nkilib.core.{attention,rmsnorm,qkv,mlp,moe,...}        │
│   nki.{isa,language,collectives,compiler}                │
│   torch_neuronx, neuronx_cc, libneuronxla                │
│   几十万行，AWS 持续更新——fork 不现实                    │
└──────────────────────────────────────────────────────────┘
```

**边界判断依据**：Layer 1+2 是 Python 业务代码（应用框架、模型组装、AOT 工作流），fork 之后我们能**裁剪 LLM-only 路径**（KV cache、speculative、autobucketing、on_device_sampling、fused_spec），让基类纯净化。Layer 3 是编译器+kernel 工具链，深度绑定硬件，fork 等于自维护一个 toolchain。

---

## 仓库布局

```
nova/                                       # 新仓库
├── nova/
│   ├── __init__.py                         # 顶层 export: NovaPipeline, register_model
│   │
│   ├── core/                               # ★ Layer 2 fork：NxDI 共用基础设施
│   │   ├── application_base.py             # ← NxDI models/application_base.py 改造（去 LLM-only 路径）
│   │   ├── model_wrapper.py                # ← NxDI models/model_wrapper.py
│   │   ├── config.py                       # ← NxDI models/config.py（去 fused_spec_config、on_device_sampling）
│   │   ├── encoder_base.py                 # ← NxDI models/encoder_base.py
│   │   ├── layer_boundary_marker.py        # ← NxDI 同名
│   │   └── modules/
│   │       ├── attention/                  # ← NxDI modules/attention/ 全套
│   │       │   ├── attention_base.py       #   2488 行，CP+DP+TP 全功能
│   │       │   ├── gqa.py
│   │       │   ├── sink.py
│   │       │   ├── attention_process_groups.py
│   │       │   └── utils.py
│   │       ├── custom_calls.py             # ← NxDI 同名（CustomRMSNorm 等）
│   │       ├── checkpoint.py               # ← NxDI 同名
│   │       └── padding.py
│   │
│   ├── layers/                             # ★ Layer 1 helpers fork：diffusion 通用层
│   │   ├── activations.py                  # ← NxDI models/diffusers/activations.py
│   │   ├── embeddings.py                   # ← NxDI models/diffusers/embeddings.py（FluxPosEmbed 等）
│   │   ├── normalization.py                # ← NxDI models/diffusers/normalization.py（NeuronAdaLayerNormZero 等）
│   │   └── padder.py                       # ← NxDI models/diffusers/padder.py
│   │
│   ├── models/                             # ★ 模型实现
│   │   ├── flux/                           # ← Layer 1 fork：完整搬过来
│   │   │   ├── application.py              # NeuronFluxApplication
│   │   │   ├── pipeline.py                 # NeuronFluxPipeline (subclass FluxPipeline)
│   │   │   ├── modeling_flux.py            # 1511 行 DiT
│   │   │   ├── clip/modeling_clip.py
│   │   │   ├── t5/modeling_t5.py
│   │   │   └── vae/modeling_vae.py
│   │   │
│   │   ├── wan/                            # ★ 新写，结构对齐 flux/
│   │   │   ├── application.py              # NovaWanApplication(nn.Module)
│   │   │   ├── pipeline.py                 # NovaWanPipeline(WanPipeline)
│   │   │   ├── modeling_wan.py             # WanDiT，用 ColumnParallelLinear + attention_cte
│   │   │   ├── t5/                         # umT5 编码器
│   │   │   └── vae/                        # WanVAE 3D
│   │   │
│   │   ├── hunyuan_video/                  # ★ 新写
│   │   ├── hunyuan_video_15/               # ★ 新写（HunyuanVideo-1.5）
│   │   ├── ltx_2/                          # ★ 新写
│   │   ├── qwen_image/                     # ★ 新写
│   │   └── z_image/                        # ★ 新写
│   │
│   ├── utils/                              # ★ Layer 2 utils fork
│   │   ├── hf_adapter.py                   # load_pretrained_config
│   │   ├── diffusers_adapter.py            # load_diffusers_config
│   │   ├── distributed.py                  # get_dp_rank_spmd 等
│   │   ├── runtime_env.py                  # set_env_vars
│   │   ├── compile_env.py                  # set_compile_env_vars
│   │   ├── snapshot.py                     # snapshot 钩子
│   │   └── accuracy.py                     # 数值校验工具
│   │
│   ├── pipeline/                           # ★ Nova 自创的薄包装层（xDiT 风格 + sglang 借鉴）
│   │   ├── nova_pipeline.py                # NovaPipeline.from_pretrained 统一入口
│   │   ├── compile_cache.py                # 自动 compile/load 缓存：(model_id, shape, parallel_cfg, dtype) → cache_key
│   │   ├── parallel_config.py              # NovaParallelConfig dataclass
│   │   └── path_resolver.py                # HF 缓存路径解析（4 级 fallback，借 sglang）
│   │
│   └── registry.py                         # @register_model 装饰器（借 xDiT）
│
├── tests/
│   ├── unit/
│   ├── numerical/                          # vs HF diffusers 数值对照
│   └── e2e/
├── benchmark/
├── examples/                               # 仿 xDiT examples/ — 每个模型一个 *_example.py
│   ├── flux_example.py
│   ├── wan_example.py
│   └── run.sh
├── pyproject.toml
├── NOTICE                                  # NxDI fork 出处声明
└── docs/
```

---

## 核心设计决定

### 1. NovaPipeline — 唯一对外 API（xDiT 风格）

```python
# nova/pipeline/nova_pipeline.py
class NovaPipeline:
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,                    # HF id 或本地路径
        parallel: NovaParallelConfig,     # tp_degree, cp_enabled, cfg_parallel_enabled
        dtype: torch.dtype = torch.bfloat16,
        compile_cache_dir: str | None = None,  # None 时用默认 ~/.cache/nova/
        height: int | None = None,
        width: int | None = None,
    ) -> "NovaPipeline":
        # 1. registry.resolve(model_id) → 找到对应 ApplicationCls
        # 2. compile_cache.lookup(model_id, parallel, shape, dtype) → cache_key
        # 3. ApplicationCls(...) instantiate
        # 4. 若 cache miss → app.compile(cache_key); 若 cache hit → 直接 app.load(cache_key)
        # 5. return self

    def __call__(self, **kwargs):
        return self.app(**kwargs)         # 转给 NeuronXxxApplication.__call__

    @classmethod
    def precompile(cls, model_id, parallel, ...):
        """显式 AOT 编译，不 load。CI 环境用。"""
```

**用户使用模式**（和 NxDI Flux 当前模式 + xDiT 模式同时兼容）：

```python
# user_run.py
from nova import NovaPipeline, NovaParallelConfig
import torch

pipe = NovaPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    parallel=NovaParallelConfig(tp_degree=8, cp_enabled=True),
)
out = pipe(prompt="A cat walking", num_inference_steps=30, num_frames=49)
out.save("out.mp4")
```

启动（M1 Flux 已验证）：`NEURON_RT_NUM_CORES=8 python user_run.py`

注意：早期计划按 xDiT 习惯写成 `torchrun`。M1.4/M1.5 在 trn3pd98.3xlarge 上确认，当前 Flux/NxDI diffusion path 的 TP artifact 在 torchrun MPMD 下会遇到 `global communicator 1` 问题；默认文档与 examples 已改为单进程多 core。

第一次跑：触发 compile（分钟级，写到 `~/.cache/nova/wan_2.2_T2V_h720_w1280_tp8_cp_bf16/`）；后续：cache hit 直接 load。

### 2. Application 模式（严格沿用 NxDI 范式）

每个新模型一个 `nova/models/<name>/application.py`，模板**严格对齐** NxDI Flux：

```python
# nova/models/wan/application.py
class NovaWanApplication(nn.Module):
    def __init__(self, model_path, t5_config, backbone_config, vae_config, ...):
        super().__init__()
        self.pipe = NovaWanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.pipe.text_encoder = NeuronUmT5Application(...)
        self.pipe.transformer  = NeuronWanBackboneApplication(...)
        self.pipe.vae          = NeuronWanVAEApplication(...)

    def compile(self, compiled_path): ...     # 逐组件 compile 写到磁盘
    def load(self, compiled_path, start_rank_id, local_ranks_size, skip_warmup): ...
    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)
```

每个 `Neuron<X>Application` 继承 fork 后裁剪过的 `nova.core.application_base.NeuronApplicationBase`。

### 3. Pipeline 模式（subclass HF）

```python
# nova/models/wan/pipeline.py
class NovaWanPipeline(WanPipeline):       # ← diffusers.WanPipeline
    def __call__(self, ...):
        # 默认走 super().__call__()
        # 启用 cfg_parallel/cp 时走优化路径（仿 NxDI flux/pipeline.py 的 _call_with_parallel_cfg）
```

### 4. 模型实现（用 fork 进来的层 + Layer 3 依赖）

```python
# nova/models/wan/modeling_wan.py
from nova.core.modules.custom_calls import CustomRMSNorm
from nova.core.modules.attention.attention_base import NeuronAttentionBase
from nova.layers.embeddings import apply_rotary_emb
from nova.layers.normalization import NeuronAdaLayerNormZero

# Layer 3 直接 import（pip 依赖）
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear, RowParallelLinear, SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
)
from nkilib.core.attention.attention_cte import attention_cte


def wan_attention(q, k, v, scale):
    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    if vc_size == 2:
        return attention_cte[2](q, k, v, scale, causal_mask=False, tp_q=True, tp_k=True, tp_out=False)
    return attention_cte(q, k, v, scale, causal_mask=False, tp_q=True, tp_k=True, tp_out=False)
```

### 5. Compile cache 自动管理（Nova 自创，xDiT/NxDI 都没有）

```python
# nova/pipeline/compile_cache.py
def cache_key(model_id, parallel_cfg, height, width, num_frames, dtype) -> str:
    # 内容哈希：model_id + tp/cp/cfg/dp 各 degree + (h,w,nframes) + dtype
    return hashlib.sha256(...).hexdigest()[:16]

def cache_path(cache_dir, key) -> Path:
    return Path(cache_dir) / key

# nova_pipeline.py 内部：
# if cache_path(cache_dir, key).exists() and (cache_path/"manifest.json").exists():
#     app.load(str(cache_path))
# else:
#     app.compile(str(cache_path))
#     write_manifest(cache_path, {"nxd_version": ..., "neuronx_cc_version": ...})
```

manifest.json 记录 NXD/neuronx_cc 版本——升级工具链后 hash 不一致 → 强制重编译。

### 6. Registry（xDiT 装饰器风格）

```python
# nova/registry.py
@register_model(
    hf_paths=["Wan-AI/Wan2.2-T2V-A14B-Diffusers", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"],
    detector=lambda p: "wan2.2" in p.lower(),
)
class WanModelEntry:
    application_cls = NovaWanApplication
    default_parallel = NovaParallelConfig(tp_degree=8, cp_enabled=False)
    default_shape    = {"height": 720, "width": 1280, "num_frames": 49}
```

`NovaPipeline.from_pretrained` 内部按 4 级 fallback 解析 `model_id` → `WanModelEntry`（路径 4 级 fallback 借 sglang 的实现）。

### 7. 不做的事

- **不做 scheduler / HTTP server / OpenAI API**（和 xDiT 一致）
- **不做 CUDA fallback**（纯 Trainium，dev 验证另开分支）
- **不做 Platform 抽象层**（只跑 Neuron，引入插件式 detection 是无收益复杂度）
- **不做 Stage 模型**（HF Pipeline 的 `__call__` 已经是事实 stages，subclass+override 够用）
- **不做 Ulysses 并行**（第一阶段，跟 NxDI 走 TP+CP+CFG；CP 不够时再加）
- **不做 LLM-only 路径**：fork 进来时立刻删除 KV cache、speculative decoding、on_device_sampling、autobucketing、fused_spec_config 等

---

## Fork 操作规程（关键，定下来避免后续混乱）

### 1. 文件迁移（M0 一次性完成）

每个从 NxDI fork 进来的文件顶部加：

```python
# Forked from neuronx-distributed-inference v0.9.17334+ced6ae4e
# Original path: neuronx_distributed_inference/<original/path>
# Fork date: 2026-MM-DD
# Modifications: <初始为空，后续每次实质改动追加>
```

保留原 Apache 2.0 copyright header。

### 2. import 批量改写

`neuronx_distributed_inference.models.X` → `nova.models.X`
`neuronx_distributed_inference.modules.X` → `nova.core.modules.X`
`neuronx_distributed_inference.utils.X` → `nova.utils.X`

工具：用 `libcst` 做 AST-level 改写，**不**用 sed（避免破坏字符串字面量里的 import path）。

### 3. 顶层 NOTICE 文件

```
This product includes software derived from NeuronX Distributed Inference (NxDI):
  https://github.com/aws-neuron/neuronx-distributed-inference
  Apache License 2.0, Copyright 2024 Amazon Web Services, Inc.

Specifically forked at version 0.9.17334+ced6ae4e:
  - nova/core/{application_base,model_wrapper,config,...}.py
  - nova/core/modules/{attention/, custom_calls.py, ...}
  - nova/layers/{activations,embeddings,normalization,padder}.py
  - nova/models/flux/* (Flux DiT, CLIP, T5, VAE)
  - nova/utils/{hf_adapter,diffusers_adapter,...}.py
```

### 4. 首次裁剪（fork 之后立刻做）

裁剪 `nova/core/application_base.py` 中的 LLM-only 路径，目标删 ~30% 行数：

- `fused_spec_config` 全相关
- `on_device_sampling_config` 全相关
- `LoraModelManager` / `lora_serving` 全相关（diffusion LoRA 走另一条路径）
- KV cache 相关 (`load_state_dict_KV_*`、`prune_state_dict` 中 KV 部分)
- `chunked_prefill` 相关
- `speculative_*` 全相关
- `autobucketing` 全相关

裁剪 `nova/core/modules/attention/attention_base.py`：
- 删除 KV cache manager 相关分支（diffusion 没有 KV cache）
- 保留 CP + DP + TP，**不动** softmax/scaling/rotary
- 第一版可以留空 stub，第二版深入裁

### 5. Upstream rebase 策略（按需）

M0 阶段的迁移/rebase 脚本已在 M1 closure 后退役，`scripts/` 只保留日常可运行验证入口。后续如果需要重新同步 NxDI Layer 1+2，单独在 `tools/fork_maintenance/` 或临时分支恢复 rebase 工具，不再占用主线 `scripts/`：

- 列出我们 fork 的每个文件 + 当前 NxDI commit hash
- 拉最新 NxDI，做 3-way merge
- 冲突部分进入 `conflicts/` 目录人工解决
- 通过后更新文件头 `Modifications:` 记录

---

## 关键文件清单（实施前必读）

按 fork 顺序：

**M0 — fork 进来时必读（要一行一行过）：**
1. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/models/application_base.py` (822 行)
2. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/models/config.py`
3. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/models/model_wrapper.py`
4. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/modules/attention/attention_base.py` (2488 行)
5. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/modules/attention/{gqa,sink,attention_process_groups,utils}.py`
6. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/modules/custom_calls.py`
7. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/modules/checkpoint.py`

**M1 — Flux fork 时必读：**
8. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/models/diffusers/flux/{pipeline,modeling_flux,application}.py`
9. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/models/diffusers/{activations,embeddings,normalization,padder}.py`
10. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/models/diffusers/flux/{clip,t5,vae}/modeling_*.py`

**新写模型时参考：**
11. `/opt/aws_neuronx_venv_pytorch_2_9/.../neuronx_distributed/parallel_layers/{layers,mappings,parallel_state,layer_norm}.py`
12. `/opt/aws_neuronx_venv_pytorch_2_9/.../neuronx_distributed/kernels/{flash_attn,ring_attention_kernel}.py`
13. `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../nkilib/core/attention/{attention_cte,attention_tkg}.py`
14. `/vllm/vllm_neuron/{platform,platform_overrides,worker/neuronx_distributed_model_runner}.py` — 当 Worker / runtime env 配置遇到坑时参考 vllm-neuron 是怎么搞的

**xDiT 借鉴模式时看：**
15. `/home/ubuntu/xDiT/xfuser/parallel.py` (xDiTParallel 类，44 行) — NovaPipeline 模板
16. `/home/ubuntu/xDiT/xfuser/model_executor/pipelines/register.py` — 装饰器 registry 风格

---

## Milestones

### M0 — Fork 与基础设施（2 周）

- 建立 `nova/` 仓库骨架
- Fork Layer 2（`core/`、`utils/`、`layers/`）+ import 批量改写
- 首次裁剪 `application_base.py` / `attention_base.py` 的 LLM-only 路径
- Fork Layer 1 Flux（整个 `models/flux/`）
- `pyproject.toml` 锁住 NXD/nkilib/torch_neuronx/neuronx_cc 版本（和当前 venv 一致）
- 验证：能在我们 repo 内 import 通过、Flux 应用能 instantiate
- 输出：M0 fork 清单与早期迁移记录；M1 closure 后迁移脚本已删除，主线 `scripts/` 改为测试/benchmark 入口

### M1 — NovaPipeline + Flux 跑通（2 周）

- 写 `nova/pipeline/{nova_pipeline,compile_cache,parallel_config,path_resolver}.py`
- 写 `nova/registry.py` 装饰器
- 用 NovaPipeline 包装 fork 后的 Flux，跑通 `pipe = NovaPipeline.from_pretrained("FLUX.1-dev", ...); pipe(prompt=...)`
- 验证 compile cache 命中/miss 行为正确
- 数值对照：和 HF diffusers FluxPipeline 在 CPU bf16 上同 prompt+seed 的 latent 余弦相似度 > 0.95
- 性能基线：单 Trainium3 instance Flux.1-dev 1024×1024 @ 28 steps latency

M1 实测修正：

- Flux 默认运行协议为单进程多 core：

```bash
NEURON_RT_NUM_CORES=4 python examples/flux_example.py \
  --model black-forest-labs/FLUX.1-dev \
  --tp-degree 4 \
  --prompt "a cat" \
  --output out.png
```

- 不使用 `torchrun --nproc_per_node=4` 作为 Flux 默认 smoke，因为当前 NxDI diffusion traced artifact 在 MPMD 下会把 TP=4 component 的 global communicator 初始化成 1。
- M1 closure 实测基线：`scripts/flux_baseline_28.sh` 在 cache hit 下完成 1024×1024 / 28 steps，`from_pretrained=36.7s`，forward `7.8s`，denoise 平均 `3.78 it/s`，输出 `/tmp/flux_28step_baseline.png`。详见 `cclogs/06-M1-closure.md`。

### M2 — Wan 2.2 T2V/I2V（4 周）★大头★

- `nova/models/wan/` 全套新写
  - `modeling_wan.py`：用 ColumnParallelLinear + attention_cte + AdaLN
  - `application.py`：composing UmT5 + WanDiT + WanVAE3D
  - `pipeline.py`：subclass `diffusers.WanPipeline`
  - `vae/`：3D VAE 移植，处理 tiled decoding
- 验证 CP（context parallel）在长序列（49 帧 720p）下能跑
- 数值对照 + 性能基线（vs H100 SGLang Diffusion 8-GPU USP）

### M3 — HunyuanVideo + HunyuanVideo-1.5（5 周）

- HunyuanVideo 标准版（架构和 Wan 类似但有差异：双 stream attention 等）
- HunyuanVideo-1.5（新版本，可能涉及不同的 attention 模式，待开工时再调研）
- 复用 M2 沉淀的 3D VAE / CP 经验

### M4 — Qwen-Image + LTX-2 + Z-Image（4 周）

- 三个图像/轻视频模型，相对简单
- 主要风险：LTX-2 的两段 pipeline（base + HQ refinement）需要特殊处理 application 组合

### M5 — 优化与稳定（开放）

- 自定义 NKI kernel 替换 nkilib 中性能不够的部分（如有）
- 多 instance scaling（torchrun 多机）
- 动态 shape 支持（autobucketing 简化版，仅 height/width 而非 LLM 的 seqlen）
- Compile cache 压缩 + 远程 S3 共享

---

## 验证方案

### 数值正确性

每个模型 fork/新写后必须通过：

```python
# tests/numerical/test_<model>_vs_diffusers.py
def test_wan_2_2_t2v_numerical_match():
    nova_pipe = NovaPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", ...)
    hf_pipe   = WanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", torch_dtype=torch.bfloat16)

    nova_latents = nova_pipe.app.pipe._denoising_loop(prompt="cat", seed=42, output_type="latent")
    hf_latents   = hf_pipe(prompt="cat", generator=torch.Generator().manual_seed(42), output_type="latent").images

    assert F.cosine_similarity(nova_latents.flatten(), hf_latents.flatten()) > 0.95
```

### 端到端

```bash
# 单实例 Flux（M1 已验证入口；N 要匹配 tp-degree / visible NeuronCores）
NEURON_RT_NUM_CORES=4 python examples/flux_example.py \
    --model black-forest-labs/FLUX.1-dev --tp-degree 4 --prompt "a cat" --output out.png

# Wan T2V（M2 需重新验证运行协议；先沿用单进程多 core 假设）
NEURON_RT_NUM_CORES=8 python examples/wan_example.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers --tp-degree 8 --cp-enabled \
    --prompt "a cat walking" --num-frames 49 --output out.mp4
```

### 性能基线（每个模型 release 时记录）

| 模型 | 配置 | Trainium3 latency 目标 | 对照 |
|---|---|---|---|
| Flux.1-dev 1024² 28 steps | tp=4, single process, `NEURON_RT_NUM_CORES=4`, cache hit, `--skip-warmup` | `from_pretrained=36.7s`; forward `7.8s`; denoise avg `3.78 it/s` | 单 H100: ~6s |
| Wan2.2-T2V 720p 49f | tp=8 + cp | ≤ 8min | 8×H100 USP: ~5min |
| HunyuanVideo 720p 129f | tp=8 + cp | TBD M3 | TBD |

数字 placeholder——实际目标 M1/M2 跑出来再校准。

### Fork 健康度

每季度 CI 检查：
- `scripts/check_nxdi_drift.py` 列出我们 fork 的文件 vs 上游最新 commit 的 diff 数
- 超过阈值（如单文件 >50 行 drift）触发 review
- 每季度 rebase 一次 + 写 changelog

---

## 风险与未决问题

1. **`attention_base.py` 裁剪的连锁影响**——2488 行 LLM/diffusion 共用代码，删 KV cache 相关分支可能误删 diffusion 也用的工具函数。**对策**：M0 裁剪保守（先重命名 unused 路径为 `_legacy_*`，跑通后再删），不在第一遍就激进清洗。

2. **Wan/Hunyuan 的 cross-attention 形态和 Flux 不同**——NxDI Flux 是 self-attention + joint attention，没有 cross-attention 的 process group；Wan/Hunyuan 是 video DiT 跨 text-video，可能需要扩展 `attention_process_groups`。**对策**：M2 开工前用 1 周 spike 验证 NxDI `attention_base.py` 是否原生支持 cross-attention，不支持则同步开始扩展。

3. **3D VAE 的 tiled decoding 在 Trainium 上的内存模型**——VAE decode 高分辨率会爆 HBM；HF diffusers 的 `enable_tiling` 是 GPU pattern，Trainium 需要预先确定切片维度（编译期定）。**对策**：M2 把 tiled VAE 实现作为单独子任务（~1 周）。

4. **Compile time 长**——每个模型 (model, shape, parallel_cfg) 组合都要 AOT 编译，分钟到十分钟级。**对策**：compile_cache 要跨 dev 机/CI 共享（`compile_cache.py` 第二版加 S3 后端）；CI 跑 numerical test 时复用 cache。

5. **upstream NxDI 重大重构风险**——AWS 半年内重写 `application_base.py` API，rebase 成本激增。**对策**：fork 进来后**保持 API 接触面狭窄**——`NovaPipeline` 只调 `app.compile() / app.load() / app(**kwargs)` 这三个方法，内部接口出问题不影响外部用户。

6. **NxDI 的 LoRA 路径是 LLM-flavored**——diffusion LoRA 通常作用在 cross-attention Q/K/V/O linear 上，NxDI 现有 LoRA 是 LLM 模式，可能需要重写。**对策**：第一阶段**不做 LoRA**，M5 阶段单独做。

7. **Flux 现成实现可能依赖 NxDI utils 我们没 fork**——比如 `utils/snapshot.py`、`utils/exceptions.py`、`utils/argparse_utils.py`。**对策**：M0 fork 时跑一遍 `flux/application.py` 的 import 闭包，连同 transitive deps 全 fork 进来；不重要的 utils（如 argparse）就改写成本地等价物。

---

## 输出物（每个 milestone 末交付）

- M0：仓库骨架 + Fork tooling + LLM 路径裁剪 PR + `pyproject.toml` 锁版本
- M1：NovaPipeline + Flux 端到端跑通 + numerical/perf 基线。M1.5 closeout 记录单进程多 core 启动协议与 28-step latency。
- M2：Wan2.2 T2V/I2V 跑通 + 长序列 CP 验证
- M3：HunyuanVideo + 1.5 跑通
- M4：Qwen-Image + LTX-2 + Z-Image 跑通
- M5：优化 + LoRA + 多机 scaling

总周期估计：**16–20 周**到 M4 完成（覆盖你列的所有起步模型），M5 持续。
