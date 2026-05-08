# Session log — 2026-05-08 — M1：NovaPipeline 框架层

## 0. 本轮目标

M0 已经完成 NxDI diffusion 路径和共享基础设施的 fork，Flux 相关模块可以在正确 Neuron venv 下 import。M1 的第一步不是立刻写 Wan，而是先把 Nova 自己的薄框架层落地：

- 提供 xDiT-like 的统一入口：`NovaPipeline.from_pretrained(...)`
- 把 model id → model implementation 的选择逻辑收进 registry
- 为 Trainium AOT `compile()` / `load()` 建立 cache 管理框架
- 接入现有 fork 进来的 Flux implementation，作为第一个模型条目
- 用不触发 Neuron 编译的 dummy application 测试框架逻辑

本轮完成的是 **M1 foundation**，也就是“框架层可用”，还不是“Flux 真实端到端可跑”。

---

## 1. 新增文件总览

```
nova/
├── registry.py                         # 模型注册与解析
├── pipeline/
│   ├── __init__.py                     # 导出 NovaPipeline / NovaParallelConfig
│   ├── nova_pipeline.py                # 统一 from_pretrained / precompile / load / call
│   ├── parallel_config.py              # TP / CP / CFG parallel 配置
│   ├── compile_cache.py                # cache key + manifest.json
│   └── path_resolver.py                # 本地路径 / HF snapshot 解析
└── models/flux/
    └── entry.py                        # Flux registry factory

tests/unit/
└── test_pipeline.py                    # 框架层单元测试，使用 dummy application
```

这批文件都是 Nova 自创代码，不属于 NxDI fork 区，因此后续应按 `black` / `ruff` 维护；当前 venv 缺工具，见第 7 节。

---

## 2. `NovaParallelConfig`

文件：`nova/pipeline/parallel_config.py`

新增 dataclass：

```python
@dataclass(frozen=True)
class NovaParallelConfig:
    tp_degree: int = 1
    cp_enabled: bool = False
    cfg_parallel_enabled: bool = False
```

行为：

- `tp_degree` 必须 `>= 1`
- `cp_enabled` 和 `cfg_parallel_enabled` 互斥
- `world_size` 计算规则：
  - 普通 TP：`world_size = tp_degree`
  - CP 或 CFG parallel：`world_size = tp_degree * 2`

这个规则与 fork 进来的 Flux helper 保持一致：

```python
get_flux_parallelism_config(
    backbone_tp_degree,
    context_parallel_enabled,
    cfg_parallel_enabled,
)
```

设计取舍：

- 暂时不引入 DP degree 字段，避免在 M1 过早扩展没有被 Flux 路径实际消费的抽象。
- 先只表达当前 NxDI Flux 真正支持的三类并行：TP / CP / CFG parallel。
- 后续 Wan/Hunyuan 如果需要更显式的 DP/PP，可以扩展此 dataclass，但不破坏已有字段。

---

## 3. Registry 设计

文件：`nova/registry.py`

核心类型：

```python
@dataclass(frozen=True)
class ModelEntry:
    name: str
    application_factory: Callable | str
    hf_paths: tuple[str, ...]
    detector: Callable[[str], bool] | None
    default_parallel: NovaParallelConfig
    default_shape: dict[str, int | None]
```

### 3.1 为什么 factory 支持字符串

Flux application import 会触发 Neuron/NXD 相关模块导入。如果 registry 初始化阶段直接 import：

```python
from nova.models.flux.application import NeuronFluxApplication
```

会让任何轻量操作（例如 `from nova import NovaPipeline`、列出 registry）都可能触发 `torch_xla` / `torch_neuronx` 初始化。

因此本轮使用 lazy factory 字符串：

```python
application_factory="nova.models.flux.entry:create_flux_application"
```

只有在真正调用：

```python
entry.create_application(...)
```

时才 import `nova.models.flux.entry`，进而 import Flux Neuron application。

### 3.2 内置 Flux 注册

内置条目：

```python
name="flux"
hf_paths=(
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
)
detector=lambda model_id: "flux" in model_id.lower()
default_parallel=NovaParallelConfig(tp_degree=8)
default_shape={"height": 1024, "width": 1024, "num_frames": None}
```

注意：`detector` 目前是宽松匹配，便于本地路径如 `/models/FLUX.1-dev` 自动识别。后续如果新增名字中也包含 `flux` 的非 BFL 模型，需要收紧 detector。

### 3.3 支持显式 model_type

`NovaPipeline.from_pretrained(..., model_type="flux")` 会绕过 detector，直接取 `_REGISTRY["flux"]`。这对本地路径尤其重要，因为本地目录名未必能可靠推断模型类型。

---

## 4. Compile Cache 设计

文件：`nova/pipeline/compile_cache.py`

新增 `CacheSpec`：

```python
CacheSpec(
    model_id,
    model_path,
    model_name,
    parallel,
    dtype,
    height,
    width,
    num_frames,
    revision,
)
```

### 4.1 cache key 输入

当前 cache payload 包含：

- `model_id`
- resolved `model_path`
- registry `model_name`
- `revision`
- `parallel`: `tp_degree` / `cp_enabled` / `cfg_parallel_enabled`
- `dtype`
- shape: `height` / `width` / `num_frames`
- toolchain versions:
  - `python`
  - `torch`
  - `torch-neuronx`
  - `torch-xla`
  - `neuronx-cc`
  - `neuronx-distributed`
  - `nki`
  - `libneuronxla`
  - `diffusers`
  - `transformers`

cache key 生成方式：

```python
sha256(json.dumps(payload, sort_keys=True))[:16]
```

cache 目录结构：

```text
~/.cache/nova/<model_name>/<cache_key>/
└── manifest.json
```

可通过环境变量覆盖默认 cache root：

```bash
NOVA_COMPILE_CACHE=/path/to/cache
```

也可在 API 中传：

```python
NovaPipeline.from_pretrained(..., compile_cache_dir="/path/to/cache")
```

### 4.2 manifest 行为

compile 完成后写入：

```json
{
  "cache_key": "...",
  "cache_payload": { ... }
}
```

cache hit 条件：

- `manifest.json` 存在
- `manifest["cache_payload"] == spec.payload()`

当前只检查 manifest，不检查每个组件产物目录是否完整。原因是不同模型组件目录不同，M1 先保持框架通用；M1 Flux smoke 之后可以为 Flux 增加 artifact completeness check。

### 4.3 当前行为

`NovaPipeline.from_pretrained(...)`：

- cache miss：`app.compile(compiled_path)` → `write_manifest(...)`
- cache hit：跳过 compile
- 默认 `load=True`，随后调用 `app.load(compiled_path, ...)`

`NovaPipeline.precompile(...)`：

- 等价于 `from_pretrained(..., load=False)`
- 用于 CI 或预编译任务

---

## 5. Path Resolver

文件：`nova/pipeline/path_resolver.py`

解析顺序：

1. 如果 `model_id` 是本地存在路径，返回绝对路径。
2. 否则调用 `huggingface_hub.snapshot_download(...)`。

支持参数：

- `revision`
- `local_files_only`

当前没有实现 plan 中提到的“四级 fallback”。原因：M1 先做最小可靠路径，避免在没有真实 HF 权重验证前写复杂路径猜测逻辑。后续可扩展：

- 显式本地目录
- HF cache snapshot
- 环境变量 model root
- registry 默认 mirror root

---

## 6. `NovaPipeline`

文件：`nova/pipeline/nova_pipeline.py`

### 6.1 Public API

核心入口：

```python
NovaPipeline.from_pretrained(
    model_id,
    parallel=None,
    dtype=None,
    compile_cache_dir=None,
    height=None,
    width=None,
    num_frames=None,
    model_type=None,
    revision=None,
    local_files_only=False,
    force_compile=False,
    skip_compile=False,
    load=True,
    start_rank_id=None,
    local_ranks_size=None,
    skip_warmup=False,
    debug_compile=False,
    application_kwargs=None,
)
```

返回对象保存：

- `app`
- `model_id`
- `model_path`
- `model_entry`
- `compiled_path`
- `cache_spec`
- `shape`
- `parallel`
- `dtype`

### 6.2 执行流程

`from_pretrained` 内部步骤：

1. `resolve_model(model_id, model_type)` 找到 registry entry。
2. 解析 parallel：用户传入优先，否则使用 entry 默认值。
3. 解析 dtype：用户传入优先，否则默认 `torch.bfloat16`。
4. 合并 shape：registry default shape + 用户传入的 height/width/num_frames。
5. `resolve_model_path(...)` 把 model id 转成实际本地路径。
6. 创建 `CacheSpec`。
7. 计算 `compiled_path`。
8. 调用 `entry.create_application(...)` 构造模型 application。
9. 根据 manifest 判断 cache hit/miss。
10. cache miss 或 `force_compile=True` 时调用 `app.compile(...)`。
11. `load=True` 时调用 `app.load(...)`。
12. 返回 wrapper。

### 6.3 compile/load 兼容处理

不同 application 的签名可能略有差异，因此用了 `inspect.signature`：

- 如果 `compile` 支持 `debug` 参数，则传入 `debug=...`
- 如果 `load` 支持 `start_rank_id` / `local_ranks_size` / `skip_warmup`，则按名字传入

这主要是为了兼容 NxDI 风格 application，同时避免未来新模型 application 必须一字不差复刻 Flux 签名。

### 6.4 `__call__`

`NovaPipeline.__call__(*args, **kwargs)` 只做一件事：

```python
return self.app(*args, **kwargs)
```

框架层不尝试重新定义 diffusion 参数，也不封装 prompt/steps/output 等模型级语义。这样可以保持对 HF pipeline 子类的兼容。

---

## 7. Flux Entry

文件：`nova/models/flux/entry.py`

新增 factory：

```python
def create_flux_application(model_path, parallel, dtype, shape, **kwargs):
    ...
```

内部流程：

1. lazy import：
   - `NeuronFluxApplication`
   - `create_flux_config`
   - `get_flux_parallelism_config`
2. 从 shape 取 `height` / `width`，默认 1024。
3. 用 `NovaParallelConfig` 计算 Flux 需要的 `world_size`。
4. 调用 `create_flux_config(...)` 生成：
   - CLIP config
   - T5 config
   - Flux backbone config
   - VAE decoder config
5. 实例化：

```python
NeuronFluxApplication(
    model_path,
    *configs,
    height=height,
    width=width,
    **kwargs,
)
```

重要约束：

- 这里没有修改 fork 进来的 `nova/models/flux/application.py`，避免污染 NxDI fork diff。
- `application_kwargs` 可从 `NovaPipeline.from_pretrained(...)` 透传到 Flux application，例如未来传自定义 `pipeline_class`。

---

## 8. 测试

文件：`tests/unit/test_pipeline.py`

为了不触发 Neuron 编译和真实模型下载，测试注册了一个 dummy model：

```python
@register_model(
    name="unit_dummy",
    application_factory=create_dummy_application,
    detector=lambda model_id: model_id.endswith("unit-dummy-model"),
    default_parallel=NovaParallelConfig(tp_degree=2),
    default_shape={"height": 64, "width": 64, "num_frames": None},
)
```

Dummy application 实现：

- `compile(...)`：记录调用并写一个 `compiled.txt`
- `load(...)`：记录调用参数
- `__call__(...)`：返回收到的 args/kwargs

### 8.1 覆盖点

`test_pipeline_compiles_and_loads_on_cache_miss`

- 本地 model path 解析
- 默认 parallel 使用 registry entry
- 默认 shape 合并
- cache miss 时调用 compile
- compile 后写 manifest
- 默认 load=True 时调用 load

`test_pipeline_skips_compile_on_cache_hit`

- 第一次 from_pretrained 写 manifest
- 第二次相同 spec 命中 cache
- cache hit 时不调用 compile
- 仍然调用 load

`test_parallel_config_rejects_conflicting_parallel_modes`

- CP 和 CFG parallel 同时启用时报错

`test_pipeline_call_delegates_to_application`

- `NovaPipeline.__call__` 直接转发到 application
- `skip_compile=True, load=False` 可用于纯构造/测试场景

---

## 9. 验证命令与结果

### 9.1 单元测试

命令：

```bash
PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \
PYTHONPATH=/home/ubuntu/nova \
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python -m pytest tests/unit
```

结果：

```text
collected 4 items
tests/unit/test_pipeline.py ....                                         [100%]
4 passed in 0.04s
```

### 9.2 API import smoke

命令：

```bash
PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \
PYTHONPATH=/home/ubuntu/nova \
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python -c \
'from nova import NovaPipeline, NovaParallelConfig; from nova.registry import registered_models; print(NovaPipeline.__name__, NovaParallelConfig(tp_degree=8, cfg_parallel_enabled=True).world_size, [m.name for m in registered_models()])'
```

结果：

```text
NovaPipeline 16 ['flux']
```

### 9.3 语法编译

命令：

```bash
PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \
PYTHONPATH=/home/ubuntu/nova \
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python -m compileall -q \
  nova/pipeline nova/registry.py nova/models/flux/entry.py tests/unit/test_pipeline.py
```

结果：通过。

---

## 10. 未完成与限制

### 10.1 没有真实 Flux compile/load/forward

本轮只验证框架层，未做：

- 下载/加载 `black-forest-labs/FLUX.1-dev`
- `NeuronFluxApplication` 实例化 smoke
- AOT `compile()`
- `load()`
- `pipe(prompt=...)` forward

原因：需要可用 NeuronCore、本地权重路径、足够编译时间和磁盘空间。

### 10.2 cache manifest 还没有 artifact 完整性检查

现在 cache hit 只看 `manifest.json` payload 是否一致。真实 Flux 编译后还应检查组件目录：

```text
text_encoder/
text_encoder_2/
transformer/
decoder/
```

是否存在，以及是否包含预期 Neuron artifact。这个应在 Flux smoke 后补。

### 10.3 `black` / `ruff` 未执行

尝试命令：

```bash
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python -m black --check ...
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python -m ruff check ...
```

结果：

```text
No module named black
No module named ruff
```

`pyproject.toml` 的 `dev` extra 声明了 `black==26.1.0`、`ruff==0.15.1`，但当前 venv 未安装 dev extra。后续可安装 dev 依赖后补跑格式检查。

### 10.4 `path_resolver` 仍是简化版

当前只支持：

1. 本地路径
2. HF `snapshot_download`

Plan 里的多级 fallback 尚未实现。建议等 Flux example 和本地权重路径约定明确后再扩展。

---

## 11. 下一步建议

### M1.1：补 Flux example

新增：

```text
examples/flux_example.py
```

目标 API：

```python
from nova import NovaParallelConfig, NovaPipeline

pipe = NovaPipeline.from_pretrained(
    args.model,
    model_type="flux",
    parallel=NovaParallelConfig(
        tp_degree=args.tp_degree,
        cp_enabled=args.cp_enabled,
        cfg_parallel_enabled=args.cfg_parallel_enabled,
    ),
    height=args.height,
    width=args.width,
    compile_cache_dir=args.compile_cache_dir,
)

image = pipe(prompt=args.prompt, num_inference_steps=args.steps).images[0]
image.save(args.output)
```

### M1.2：补真实 Flux smoke 脚本

建议新增：

```text
tests/e2e/test_flux_smoke.py
```

或先放：

```text
examples/flux_smoke.py
```

要求：

- 必须标记 `@pytest.mark.neuron`
- 默认跳过，只有设置 `NOVA_FLUX_MODEL_PATH` 时运行
- 编译目录使用临时目录或显式 `NOVA_COMPILE_CACHE`
- 第一次只验证 instantiate + compile/load，不要求出图质量

### M1.3：补 cache 负例测试

新增测试：

- height 改变 → cache key 改变 → compile 再次发生
- dtype 改变 → cache key 改变
- parallel config 改变 → cache key 改变
- manifest 被篡改 → cache miss

### M1.4：初始化 git baseline

当前 `/home/ubuntu/nova` 不是 git repo。M1 继续推进前建议：

```bash
git init
git add .
git commit -m "Initial Nova diffusion fork and pipeline foundation"
```

这样后续 fork drift、Nova 自创代码、Wan/Hunyuan 新增代码才有清晰 diff 边界。
