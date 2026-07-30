# FLUX.1-dev Trainium2 推理加速 — TAEF1 轻量 VAE

> 分支: `feat/taef1-flux-vae-optimization`  
> 设备: trn2.3xlarge (4 NeuronCores, 96 GB HBM)  
> 基线: tp=4, bf16, 268ms/step, warm e2e 40s

---

## 1. 成果概要

用 `madebyollin/taef1` (AutoencoderTiny, 1.2M 参数) 替换标准 VAE Decoder (80M 参数)，作为**可选功能**。

| 指标 | 标准 VAE | TAEF1 | 提升 |
|------|:---:|:---:|:---:|
| 参数量 | 80M | 1.2M | 67× |
| 权重文件 | 378 MB | 9.4 MB | 40× |
| VAE Load (device init) | 6.6s | 0.9s | 7.3× |
| VAE Decode | ~4s | 69ms | ~58× |
| Denoise loop | 7.6s | 7.6s | 不变 |
| Warm Load 总计 | ~36s | ~17s | 2.1× |
| Warm E2E 总计 | ~44s | ~25s | 1.8× |

**TAEF1 是完全可选的** — 不传参数时使用标准 VAE，行为与修改前完全一致。

---

## 2. 使用方式

### 2.1 标准 VAE（默认，无需任何改动）

```python
from difflet import DiffletPipeline, DiffletParallelConfig
import torch

pipe = DiffletPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    model_type="flux",
    parallel=DiffletParallelConfig(tp_degree=4),
    dtype=torch.bfloat16,
    height=1024, width=1024,
)
out = pipe(
    prompt="a red fox sitting in a snowy forest",
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=torch.Generator().manual_seed(42),
)
out.images[0].save("output.png")
```

### 2.2 TAEF1 轻量 VAE（通过 `application_kwargs` 启用）

```python
pipe = DiffletPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    model_type="flux",
    parallel=DiffletParallelConfig(tp_degree=4),
    dtype=torch.bfloat16,
    height=1024, width=1024,
    application_kwargs={
        "taef1": True,
        "taef1_path": "madebyollin/taef1",  # 自动从 HuggingFace 下载
    },
)
out = pipe(
    prompt="a red fox sitting in a snowy forest",
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=torch.Generator().manual_seed(42),
)
out.images[0].save("output_taef1.png")
```

---

## 3. 核心实现

改动集中在 3 个文件，6 个文件总计 193 行。

### 3.1 VAE 模型层 (`difflet/models/flux/vae/modeling_vae.py`)

**问题**: `DecoderTiny` (TAEF1) 和 `Decoder` (标准 VAE) 构造函数参数完全不同。

标准 Decoder 需要: `in_channels, out_channels, up_block_types, block_out_channels, layers_per_block, norm_num_groups, act_fn, mid_block_add_attention`

DecoderTiny 需要: `in_channels, out_channels, num_blocks, block_out_channels, upsampling_scaling_factor, act_fn, upsample_fn`

**解决**: `get_decoder_config()` 根据模型类型分发正确的参数字典:

```python
def get_decoder_config(model_cls, load_config_dict, height, width, ...):
    if model_cls is DecoderTiny:
        return {
            "in_channels": load_config_dict["latent_channels"],      # 16
            "out_channels": load_config_dict["out_channels"],        # 3
            "num_blocks": load_config_dict["num_decoder_blocks"],    # [3,3,3,1]
            "block_out_channels": load_config_dict["decoder_block_out_channels"],
            "upsampling_scaling_factor": load_config_dict.get("upsampling_scaling_factor", 2),
            "act_fn": load_config_dict.get("act_fn", "relu"),
            "upsample_fn": load_config_dict.get("upsample_fn", "nearest"),
        }
    # 标准 Decoder
    return {
        "in_channels": load_config_dict["latent_channels"],
        "out_channels": load_config_dict["out_channels"],
        "up_block_types": load_config_dict.get("up_block_types", []),
        "block_out_channels": load_config_dict.get("block_out_channels", []),
        "layers_per_block": load_config_dict.get("layers_per_block", 1),
        "norm_num_groups": load_config_dict.get("norm_num_groups", 32),
        "act_fn": load_config_dict.get("act_fn", "silu"),
        "mid_block_add_attention": load_config_dict.get("mid_block_add_attention", True),
    }
```

**DecoderTiny 没有 GroupNorm** — 跳过 `PatchedGroupNorm` monkey-patch:

```python
def _create_model():
    is_tiny = self.model_cls is DecoderTiny
    if not is_tiny and self.config.neuron_config.torch_dtype == torch.bfloat16:
        torch.nn.GroupNorm = PatchedGroupNorm  # 只在标准 VAE 需要
    model = self.model_cls(**self.config.decoder_config)
    ...
```

### 3.2 Application 层 (`difflet/models/flux/application.py`)

`NeuronFluxApplication` 新增 `taef1`/`taef1_path` 参数（默认 `False`/`None`）:

```python
class NeuronFluxApplication(MultiComponentApplication):
    def __init__(self, ..., taef1=False, taef1_path=None):
        ...
        # 加载 pipeline（VAE 保留为 CPU 版本，decoder 将被替换）
        self.pipe = pipeline_class.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            text_encoder=None, text_encoder_2=None, transformer=None,
        )
        ...
        if taef1:
            from diffusers import AutoencoderTiny
            # 整个 VAE 替换为 AutoencoderTiny
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taef1_path, torch_dtype=torch.bfloat16,
            )
            # decoder 编译为 Neuron NEFF（DecoderTiny 而非 Decoder）
            self.pipe.vae.decoder = NeuronVAEDecoderApplication(
                model_path=taef1_path, config=self.decoder_config,
                model_cls=DecoderTiny,
            )
        else:
            # 默认路径：标准 VAE decoder
            self.pipe.vae.decoder = NeuronVAEDecoderApplication(
                model_path=self.vae_decoder_path, config=self.decoder_config,
            )
```

### 3.3 入口层 (`difflet/models/flux/entry.py`)

工厂函数透传 `taef1` kwarg:

```python
def create_flux_application(*, model_path, parallel, dtype, shape,
                             backend="trainium", **kwargs):
    taef1 = bool(kwargs.pop("taef1", False))
    taef1_path = kwargs.pop("taef1_path", None)
    ...
    configs = create_flux_config(..., taef1=taef1, taef1_path=taef1_path)
    return NeuronFluxApplication(model_path, *configs, ...,
                                 taef1=taef1, taef1_path=taef1_path, **kwargs)
```

---

## 4. 配套优化

除 TAEF1 外，本次还包含两项辅助改动:

### 4.1 编译器 `-O1` → `-O2` (`modeling_flux.py`)

`-O2` 是 neuronx-cc 的默认优化级别。原代码显式使用 `-O1` 是降级。无性能变化，但消除了配置错误。

### 4.2 预分片权重并行 I/O + 计时 (`application_base.py`)

- 多个 TP shard 文件并发加载（ThreadPoolExecutor）
- 分阶段计时：`file_read` / `device_init` 单独可见
- 编译后验证分片文件完整性

### 4.3 TeaCache 调优 (`run_flux_teacache_e2e.py`)

- TARGET_SKIP 0.4→0.5 (理论 speedup 2.0x)
- 新增 `ONLINE_DELTA_ALPHA` 模式（零 calibration）

---

## 5. 实验记录: 试了但无效的方向

| 实验 | 结果 | 原因 |
|------|------|------|
| FP8 auto-cast backbone | per-step 不变 (268→271ms) | `--auto-cast` 只转 FP32 算子，FLUX 全程 BF16 |
| FP8 auto-cast VAE | 编译 6.9s→704.7s (100×) | `unet-inference` 路径不兼容 |
| Text KV Cache | 收益 <1% (2ms/step) | text 仅占序列 11%，不值得 SPMD 重编译 |

---

## 6. 改动清单

| 文件 | 改动 | 类型 |
|------|------|------|
| `difflet/models/flux/vae/modeling_vae.py` | 支持 DecoderTiny: `get_decoder_config()` 分发, 跳过 PatchedGroupNorm | TAEF1 核心 |
| `difflet/models/flux/application.py` | `NeuronFluxApplication` 新增 `taef1`/`taef1_path` 可选参数 | TAEF1 核心 |
| `difflet/models/flux/entry.py` | 工厂函数透传 `taef1` kwarg | TAEF1 核心 |
| `difflet/models/flux/modeling_flux.py` | `-O1` → `-O2`（纠正为编译器默认值） | 辅助 |
| `difflet/backends/trainium/core/application_base.py` | 预分片并行 I/O + 分阶段计时 | 辅助 |
| `scripts/run_flux_teacache_e2e.py` | TARGET_SKIP 0.5 + online-delta mode | 辅助 |

---

## 7. 复现

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd /home/ubuntu/Difflet

# 标准 VAE (默认)
python -c "
from difflet import DiffletPipeline, DiffletParallelConfig
import torch

pipe = DiffletPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev', model_type='flux',
    parallel=DiffletParallelConfig(tp_degree=4),
    dtype=torch.bfloat16, height=1024, width=1024,
)
out = pipe(
    prompt='a red fox sitting in a snowy forest, sharp detail',
    num_inference_steps=28, guidance_scale=3.5,
    generator=torch.Generator().manual_seed(42),
)
out.images[0].save('output_std.png')
print('Standard VAE: done')
"

# TAEF1 轻量 VAE
python -c "
from difflet import DiffletPipeline, DiffletParallelConfig
import torch

pipe = DiffletPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev', model_type='flux',
    parallel=DiffletParallelConfig(tp_degree=4),
    dtype=torch.bfloat16, height=1024, width=1024,
    application_kwargs={'taef1': True, 'taef1_path': 'madebyollin/taef1'},
)
out = pipe(
    prompt='a red fox sitting in a snowy forest, sharp detail',
    num_inference_steps=28, guidance_scale=3.5,
    generator=torch.Generator().manual_seed(42),
)
out.images[0].save('output_taef1.png')
print('TAEF1 VAE: done')
"
```
