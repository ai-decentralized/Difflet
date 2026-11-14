# `difflet bench`：统一 benchmark 入口 — 设计

**日期：** 2026-11-16
**状态：** 草稿，待用户审阅通过。没有实现任何代码。讨论中提出的问题都已有答复，见第 16 节。
**事实核对基线：** main `d70faffc`。标注"已核实"的条目都对着代码或真实 parser 跑过。
**审阅：** 已由一个独立的审阅者以实施者视角读过一遍。它指出的 5 处与代码矛盾的地方已改正，见第 3 节表格的后七行和第 7 节。

标记约定：【用户定】是用户在讨论中明确决定的；【我的选择】是我替用户做的默认，可以推翻；【未决】需要用户回答。

## 1. 目的

让团队每个人用同一条命令跑 benchmark，并且任何一个数字都能被另一个人在自己的机器上原样复现。每个人有自己的 trn3。

现状（已核实）：

- `benchmark/bench.py` + `harness.py` + `models.py::MATRIX` 是一套设计合理的框架，但实际出数走的是五条旁路：`cold_warm_e2e.py`、`warm_e2e.py`、`step_realloop.py`、Wan 专用的 `neuron_cold_warm.py`，以及 mxfp8 分支上纯 shell 的 `scripts/quant_bench/`。
- 计时靠正则解析日志行。`difflet` CLI 没有结构化的计时输出。
- 结果 JSON 不记录 commit、命令行、环境变量、设备绑定、缓存状态。22 份结果有 5 种 schema。
- 环境写死在代码里：5 种 venv 路径、3 种 repo 路径、4 种 HF cache 路径。默认的 trainium adapter 指向的 venv 在当前 trn3 主机上不存在。
- `RESULTS.md` 全部手写。DP 的现有结果是从 tp2 推导的，不是测出来的。
- 功能 flag 在 benchmark 里被重复声明，已经不同步：`step_realloop.py` 只抄了 4 个 TeaCache flag 中的 2 个；`bench.py` 的额外 flag 只传给 generate 不传给 compile，所以会改变编译产物的功能没法通过它测。

## 2. 目标、非目标、反目标

**目标**

1. 唯一入口 `difflet bench`【用户定】，所有人和所有脚本都从这里进。
2. 用参数选 metric 和模型【用户定】。第一版 6 个 metric：`compile`、`e2e_cold`、`e2e_warm`、`dit_step`、`memory`、`quality`。
3. 功能 flag 自动继承自 `difflet compile` / `difflet generate`【用户定】。引擎加新 flag，bench 不改代码。
4. parallelism 包括 DP【用户定】。DP 的指标是 `throughput`，离线批量和在线压测两种都要做，但排在最后，现在只登记不实施【用户定】，见第 7 节末尾。
5. 有内存指标【用户定】，能替代 quant_bench。
6. 每条结果自带复现所需的全部信息，并能用一条命令原样重跑。
7. 第一版只支持 PyTorch-native 的 neuron backend【用户定】。

**非目标（第一版不做）**

- trainium 的 AOT backend【用户定】，以及 H100、B300、TPU、CPU 作为被测目标和 stock diffusers 参考引擎。旧的 `benchmark/` harness 原样冻结，继续给它们用。
- `throughput` 不在第一版的实施步骤里。它是登记在案的后置阶段。
- `e2e_resident`，即模型已加载时单个请求的处理时间【用户定：不要】。
- 看板、结果上传、带绝对基线的 CI 性能门禁。
- 任何脚本的删除。清理清单只登记，见第 14 节【用户定】。

**反目标（做成这样就算失败）**

- 在 bench 里重新声明引擎的 flag。
- 产出不带出处的数字，或者运行无效却看起来成功。
- 为了测量而改变被测路径却不标注。
- 在共享主机上清全局 page cache、按名字或端口杀进程、占用别人正在用的设备。

## 3. 约束（已核实的事实）

| 事实 | 对设计的影响 |
|---|---|
| `compile` 有 37 个 flag，全部包含在 `generate` 的 53 个里，没有 compile 独有的 flag | flag 路由可以完全从 parser 推出 |
| 同一个 flag 出现两次时后者生效 | 覆盖 = 把参数接在后面，不需要合并逻辑 |
| generate 的 12 个布尔开关都没有 `--no-` 形式 | 命令行只能打开开关，不能关掉 |
| TeaCache 的 flag 与 `--requests`、`--dp` 互斥，CLI 直接拒绝 | 只影响后置的 `throughput`：带 TeaCache 的 recipe 测不了离线吞吐 |
| DP 路由器永远从 core 0 开始给副本分核，并覆盖传入的 `NEURON_RT_VISIBLE_CORES`。修复已存在但未合并，在分支 `fix/dp-router-visible-cores` 上 | 只影响后置的 `throughput`：做它之前先合并这个修复 |
| 只要带 `--requests`，哪怕 dp=1，也会经过 DP 路由器和 worker | 第一版的 metric 都是单请求运行，不带 `--requests`，不经过路由器 |
| 批量请求是按阶段处理的：一个阶段把所有请求处理完，下一个阶段才开始，每个阶段是一个独立进程 | 只影响后置的 `throughput`，决定了它的算法，见第 7 节末尾 |
| 路由器转发给 worker 的 flag 是一份手写名单，漏了 `--shapes`、`--taef1`、`--taef1-path`、`--vae-in-process`、`--first-frame`、`--last-frame` | 只影响后置的 `throughput`：用到这些 flag 的 recipe 在批量模式下跑的是另一个配置，做它之前要修 |
| 不显式给 `--backend neuron` 时，CLI 走 AOT 路径；自动检测不会切到 neuron | recipe 必须显式写 backend |
| `--quant` 只在 neuron backend 上可用，且 main 上只有 Qwen-Image 和 FLUX 支持 | 量化 recipe 必须带 `backend: neuron`；视频模型的量化在 mxfp8 分支上 |
| repo 里已有 neuron 的依赖锁文件 `requirements-neuron-native.lock`，torch 同为 2.14.0，但锁的是 CPython 3.14 | 实施时把锁文件更新为第 10 节的基准环境 |
| neuron backend 每次 DiT forward 后都 `torch.neuron.synchronize()` 并把输出拷回主机 | `dit_step` 和 e2e 可以取自同一次运行 |
| neuron backend 已有 `step_hook`、`forward_hook`，以及编译块数、未预热形状、回退算子这些证据，目前只打在日志里 | 结构化计时主要是把现有 hook 接到一个文件上 |
| `benchmark*` 被 pyproject 排除在安装包外；native-fp8 venv 里没有 `difflet` 命令，靠 `python -m difflet.cli.main` 运行 | bench 代码要进包；子进程必须用当前解释器启动 |
| `difflet` CLI 的 import 耗时 22 ms，不加载 torch | 新子命令必须延迟 import，不能改变这一点 |
| 初始和最终 latent 的 dump 只有 Qwen-Image 和 LTX-2 两个 pipeline 有 | `quality` 要覆盖全部模型需要补 dump |
| 当前 trn3.48xlarge 主机上没有任何 venv 装了 NxD，AOT backend 在这里跑不了 | 第一版不做 AOT，这一条不再是阻碍 |
| 这台主机上能跑 neuron 的三个 venv 都是 Python 3.12.3、torch 2.14.0、torch-neuronx 2.14.3.0.1；torch 的 wheel 只有 cp312 的构建 | 基准环境就取它，见第 10 节 |
| 冷编译一次要几十分钟 | 需要续跑，以及"warm 运行不得编译"的保护 |

## 4. 命令接口

```bash
difflet bench list                                  # 本机可用的 recipe、suite、metric
difflet bench run --recipe qwen_image --metrics dit_step,e2e_warm
difflet bench run --recipe flux_1_dev --metrics dit_step,quality -- --teacache-cadence 2
difflet bench run --suite quant_ab --name <实验名> [--select <正则>] [--resume]
difflet bench run --repro <result.json>             # 按结果里的记录原样重跑
difflet bench reference --from <result.json>        # 用一次设备运行的初始 latent 生成 CPU fp32 参考
difflet bench compare <A.json> <B.json>
difflet bench report [<结果目录>]
```

`run` 的参数：

| 参数 | 含义 |
|---|---|
| `--recipe NAME[,NAME]` 或 `all` | 选 recipe，也接受一个 YAML 文件路径 |
| `--metrics LIST` | 缺省为 `compile,e2e_cold,e2e_warm,dit_step,memory`。`quality` 要显式选 |
| `--suite NAME` 或路径 | 跑一组运行，见第 6 节 |
| `--devices LIST` | Neuron device 编号。bench 翻译成 `NEURON_RT_VISIBLE_CORES` |
| `--cpus LIST` | 主机 CPU 集合。给了 `--devices` 就必须同时给，bench 不替用户猜 |
| `--quality-ref` | `cpu_fp32`，或一份结果 JSON 的路径（以那次运行为对照）。suite 里由 `compare.control` 提供 |
| `--name` | suite 运行的实验名，续跑以它为单位，见第 6 节 |
| `--repeats N` | 覆盖各 metric 的默认重复次数 |
| `--dry-run` | 只打印解析后的配置、将要执行的命令、环境变量差异，不碰设备 |
| `--resume` | 已有有效结果的运行跳过 |
| `--check-env REF.json` | 把本机环境与一份参考结果里记录的环境做对比 |
| `-- <引擎 flag>` | 原样透传给 compile / generate |

说明：

- 现有的 `difflet run` 是"下载加编译加生成"，与 `difflet bench run` 无关，帮助文本里要写清。
- `bench` 和 `clean`、`cache`、`plan` 一样，进入分支后才 import。
- 所有子进程用 `sys.executable -m difflet.cli.main` 启动。

## 5. Recipe

recipe 是一组有名字、进版本库的引擎 flag，加少量 bench 自己的字段。

```yaml
# difflet/bench/recipes.yaml
qwen_image:
  args:                       # key 就是 difflet generate 的 flag 名
    backend: neuron           # 必填。不写的话 CLI 会走 AOT 路径
    model-id: Qwen/Qwen-Image
    revision: 75e0b4be04f60ec59a75f475837eced720f823b6
    tp-degree: 4
    height: 1024
    width: 1024
    steps: 50
    guidance-scale: 7.5
    seed: 42
    prompt: a cat sitting on a bench

qwen_image_mxfp8:             # 只写与基础 recipe 的差异
  extends: qwen_image
  args: {quant: mxfp8}

wan21_official:
  args:
    backend: neuron
    model-id: Wan-AI/Wan2.1-T2V-14B-Diffusers
    revision: 38ec498cb3208fb688890f8cc7e94ede2cbd7f68
    tp-degree: 4
    height: 480
    width: 832
    num-frames: 81
    steps: 50
    guidance-scale: 5.0
    seed: 42
    prompt: a cat walking through a garden
    negative-prompt: "Bright tones, overexposed, static, blurred details, ..."
  threads: 48                 # 可选；固定主机线程数
  env:
    force: {}                 # 属于被测配置，覆盖用户环境
    default: {}               # 只在用户没设置时生效
```

上面的取值来自 quant_bench 现在实际在用的配置。

规则：

1. **机械翻译。** `k: v` 变成 `--k v`；`true` 变成 `--k`；`false` 表示不出现。bench 不为任何引擎 flag 建字段。
2. **用真实 parser 校验。** `args` 的每个 key 必须是 `difflet generate` 认识的 flag。未知的 key、未知的顶层字段一律报错。`revision` 和 `backend` 必填。第一版所有 recipe 的 backend 都是 neuron，但仍要显式写，因为 CLI 不写时走的是 AOT 路径。
3. **路由从 parser 推出。** 全部参数给 generate；`difflet compile` 的 parser 认识的那部分同时给 compile。这个子集也就是"会改变编译产物"的参数集合，不需要人工分块。
4. **优先级：recipe < suite 变体 < 命令行 `--`。** YAML 各层的合并发生在翻译成 flag 之前，所以变体可以把开关设回 `false`。命令行做不到，因此基础 recipe 保持开关全关，由变体去打开。
5. **bench 自管的 flag** 不允许出现在 recipe 或透传里：`output`、`cache-dir`、`work-dir`、`keep-work-dir`、`requests`、`requests-dir`、`worker-index`、`force`。
6. **命令行覆盖的运行**记录覆盖内容，结果目录名带参数指纹，不进报告主表。值得保留的配置再升级为命名 recipe。
7. **环境变量是 recipe 的一部分。** 启动器先清掉继承来的引擎相关变量，再应用 recipe 的 `env`，最终生效值写进结果。
8. **主机相关的内容不进 recipe**：解释器、缓存根目录、模型目录、设备编号。

初始内容有两个来源：`MATRIX` 里唯一的 neuron 条目 `wan_2_1_neuron`，以及 quant_bench 的 `bench.sh` 里已有的 10 组 neuron 配置，即 7 个模型的基础配置和 3 组官方设置。`MATRIX` 里另外 7 条是 AOT recipe，不迁移。【我的选择】文件随包放在 `difflet/bench/` 下，用 YAML；PyYAML 是 transformers 和 huggingface_hub 的硬依赖，已在所有 venv 里，只需在 pyproject 里声明。

`dp` 大于 1 的 recipe 属于后置的 `throughput`，第一版不收。TP、CP、SP、CFG 并行都是普通的 flag，第一版的 metric 直接覆盖。

## 6. Suite

suite 是一份有名字的运行清单【用户感兴趣，我的形状】。repo 里已有四种手写形式：`bench.py --all`、Flux 并行配置字典、quant_bench 的两臂、两份 `drive_all.sh`。

```yaml
# difflet/bench/suites/quant_ab.yaml
description: bf16 对 MXFP8
metrics: [compile, e2e_cold, e2e_warm, dit_step, memory, quality]
recipes: [qwen_image, flux]
variants:
  bf16: {}
  mxfp8: {args: {quant: mxfp8}}
compare: {control: bf16, candidate: mxfp8}
order: ABBA                   # 对照、候选、候选、对照
repeats: 2
```

- 唯一的乘积是 recipes × variants。没有更一般的矩阵语法。
- 每个运行有稳定的 id：`<recipe>/<variant>`。`--select` 按正则选，`--dry-run` 只打印。
- 同一个运行的一次编译，供它的所有 metric 复用。
- 加载时就校验所有 recipe 名和变体参数。一个运行失败不影响其余，失败本身也落盘。

**实验和续跑。**【我的选择】一次 suite 运行是一个"实验"，用 `--name` 起名。实验有一份清单文件，列出计划的每个运行及其结果路径和状态。

- 续跑以实验为单位：`--resume` 只跳过本实验清单里已经有效的运行，并且要求当前 commit 和 recipe 文件哈希与清单一致，不一致就拒绝续跑。
- 同名实验已存在而没带 `--resume` 时拒绝运行。
- 单独的 `bench run --recipe` 不受这条限制，永远可以追加一份新结果。

**`order` 和 `repeats` 的含义。**【我的选择】每个臂只编译一次。之后对每个计时类 metric，按对照、候选、候选、对照的顺序交替启动进程，这算一轮。`repeats` 是轮数，所以每个臂得到的样本数是轮数的两倍。设了 `order: ABBA` 时，它取代第 7 节里各 metric 的默认次数。

**变体的结果**放在 `<recipe>@<suite>.<variant>` 目录下。变体定义在进了版本库的 suite 文件里，不算命令行覆盖，但也不进报告主表，只出现在该 suite 的对比报告里。

## 7. Metric

| metric | 怎么跑 | 时钟 | 默认次数 |
|---|---|---|---|
| `compile` | 新进程 `compile`，编译缓存根指向一个新建的空目录 | 子进程启动到退出 | 1 |
| `e2e_cold` | 新进程 `generate`；启动前逐文件驱逐 recipe 涉及的权重和编译产物的 page cache | 子进程启动到退出 | 2 |
| `e2e_warm` | 新进程 `generate`；启动前显式预读同一批文件 | 子进程启动到退出 | 3 |
| `dit_step` | 取自 `e2e_warm` 的各次运行，不另起进程 | 相邻两步时间戳之差 | 全部步 |
| `memory` | 与 `e2e_warm` 的运行同时采样 | 每秒采样，按阶段归属 | — |
| `quality` | 单独一次 `generate`，打开 dump | 不计时 | 1 |

通用规则：

- 每个 metric 保留全部原始样本，并报 n、中位数、均值、最小、最大、p90。
- 每个 metric 有 `valid` 和 `invalid_reasons`。某个 metric 对某个 recipe 不适用时记为"不适用及原因"，其余 metric 照常跑。
- 打开了 dump 或 profiling 的运行不产出任何时间类指标。
- 每条结果带测量方法的版本号。定义改了就升版本，`compare` 拒绝跨版本比较。

各 metric 的补充：

**`compile`。** 用把缓存根重定向到临时目录的方式得到"冷"，不删用户已有的缓存。分项时间来自引擎的计时输出。同时记编译缓存的磁盘占用。没有选 `compile` 时，bench 仍先执行一次编译作为准备，使用持久缓存，不计入指标。

**`e2e_cold`。** "冷"只指 page cache；编译缓存是热的。用 `posix_fadvise` 逐文件驱逐，不需要 root，也不影响同机其他人。驱逐前后各量一次文件的驻留比例，并记录子进程实际从磁盘读入的字节数。达不到就把这次样本标为"驱逐未验证"，不冒充冷启动。结果里把 page cache、编译缓存、是否新进程分成三个字段记录。

**`e2e_warm`。** 这类运行不得发生编译。引擎报告的缓存未命中数、新编译块数或未预热形状不为零时，该样本无效。结果里同时给出阶段分解：各组件的加载、文本编码、去噪、解码各占多少，这样不需要 `e2e_resident` 也能看出纯计算部分的时间。

**`dit_step`。** 每次 `e2e_warm` 运行只有一个请求，取它的全部去噪步，各次运行的样本合并统计。neuron backend 的生产路径本来每次 forward 都同步，bench 不再加同步，结果里记录同步方式。同时报"每个调度步"和"每次真实 DiT forward"两组时间，TeaCache 跳步或 CFG 时两者数量不同。每个编译图的首次执行剔除。多 rank 时各 rank 分别记录，主数字取 rank 0，另报跨 rank 的最大和最小，不混合平均。自检：各步之和与引擎报告的去噪阶段总时间相差超过 2% 时，`dit_step` 无效【我的选择，初值待真机校准】。

**`memory`。** 主数字是每个 rank 的 HBM 峰值。用 `neuron-monitor` 每秒采样并只保留绑定的核，这是 quant_bench 现有的测法。各阶段是不同的进程，采样按计时文件里的阶段起止归属到阶段，报每个阶段的峰值和总的最大值。另记进程内的 `torch_neuronx.max_memory_allocated()` 作为核对，两者差异过大时标记。顺带记录：每个 rank 的 DiT 权重大小、主机内存峰值、读盘字节数。

**`quality`。**

- 在编码前的帧上算，不在 mp4 上算。同时比较最终 latent 的相对 L2 和余弦。
- 报 PSNR 和 SSIM 的均值、最小值、最差帧序号。
- 哈希取编码前 uint8 帧的字节，浮点转 uint8 的规则固定并写明。mp4 或 png 的文件哈希是次要字段，连同编码设置和编码器版本一起记。
- 参考有两种，结果里写明是哪一种：`cpu_fp32` 是冻结保存的 CPU fp32 输出；`control` 是同一 recipe 特性关闭时的设备运行。
- 参考的选择：suite 里由 `compare.control` 指定对照臂。单独运行时用 `--quality-ref` 指定，值是 `cpu_fp32` 或一份结果 JSON 的路径。两者都没有时报错并说明该传什么，bench 不自己去找"上一次运行"。
- CPU fp32 参考从设备运行 dump 出的初始 latent 起步，不依赖 seed。它由 `difflet bench reference --from <result.json>` 生成，输入是一次带 dump 的设备运行。
- 【我的选择】参考存放在 `$DIFFLET_BENCH_ROOT/references/<参考 id>/`，不进 git。参考 id 由模型 revision、shape、步数、prompt、guidance、初始 latent 的哈希、参考实现版本算出。目录带校验清单：每个文件的 sha256 和大小；带版本前缀，不原地覆盖。主机之间靠拷贝目录共享，清单保证拿到的是同一份。每条 quality 结果记录参考 id 和清单的哈希。
- 缺参考或缺工具时报错，不跳过。
- 【用户定】latent 和编码前帧都 dump，第一版给所有模型加上。
- 【我的选择】`control` 模式因此对所有模型可用。`cpu_fp32` 模式还需要每个模型各自的 CPU fp32 参考生成脚本，现在只有 Qwen-Image 和 LTX-2 有；其余模型的生成脚本逐个补，不阻塞第一版。
- 门限：先测量并落盘，再判定。门限是 suite 里的可选字段，第一版不内置任何门限数值。

### 后置阶段：`throughput`（只登记，暂不实施）【用户定】

离线和在线两种都要做，排在第一版所有步骤之后。先做离线，再做在线。

**离线批量**

- 怎么跑：recipe 带 `dp: N`；bench 生成请求文件，每个副本至少 3 个请求，同 prompt 同 seed；用 `generate --requests` 一次跑完。
- 算法：引擎的批量模式按阶段处理，文本编码阶段处理完所有请求才轮到去噪，再到解码，每个阶段是一个独立进程。所以不能用"某个时间窗内完成了几个请求"来算。定义为：对每个副本取它非首个请求的平均处理时间，各副本的倒数相加，单位是每秒图片数或视频数。
- 有效性：只有各副本的同一阶段在时间上确实重叠时才有效，重叠比例写进结果；所有副本输出的哈希必须一致。
- 同时报：整批的墙钟时间（含加载）、相对 dp=1 的扩展效率、单个请求的处理时间。最后一项就是被去掉的 `e2e_resident` 所指的量，在这里作为副产品给出。
- 前置条件：合并 DP 分核的修复；路由器改为按 parser 转发 flag；引擎的计时输出支持每个请求的起止。
- 限制：带 TeaCache 的 recipe 测不了，CLI 会拒绝。

**在线压测**

- 怎么跑：bench 启动一个 `difflet serve`，等它就绪，由压测客户端按设定的请求速率或并发数持续发请求，结束后关闭服务。
- 测什么：每个负载水平下延迟的中位数、p90、p99；实际完成的每秒请求数；在给定延迟上限下能承受的最大请求速率。
- 它走的是服务栈的代码路径，模型常驻，与 `generate` 不是同一条路径。它的数字不能和第一版的 e2e 指标直接比较。
- 需要新写：压测客户端、服务的启停和就绪检测、请求到达的模型、预热段丢弃的规则。recipe 还要能表达 `serve` 的参数。
- 可借鉴 vLLM 的 `bench serve` 和 SGLang 扩散部分的 `bench_serving`：请求速率和并发两条扫描轴、达标率、对客户端自身开销的检查。

## 8. 引擎侧的改动

这些是对产品代码的改动，范围尽量小。

1. **结构化计时输出。** 一个内部开关，默认关闭，关闭时代码路径与现在完全一致。打开后每个进程或 rank 写一个 JSON 文件，至少包含：
   - 经过校验和 `--mode` 改写之后的最终参数；
   - 各阶段起止：各组件的加载、文本编码、去噪、解码；
   - 每个请求的起止、每步时间戳、每次 DiT forward 的耗时、输出文件路径和哈希；
   - 生效证据：backend、执行模式、编译块数、未预热形状、回退算子、编译缓存的命中和未命中；
   - 每个请求完成后就落盘，进程被杀也留下已完成的记录。

   开关通过环境变量传递，因为 neuron backend 的各阶段运行在 torchrun 之下，环境变量会自动继承。名字和文件格式留给实施者。
2. **dump。**【用户定：都 dump】增加编码前帧的 dump，沿用现有 latent dump 的环境变量机制；给其余五个 pipeline 补上初始和最终 latent 的 dump。两者默认关闭，只在测 `quality` 的那次运行打开。
3. **后置，随 `throughput` 一起做：DP 路由器从已绑定的核池里给副本分核。** 这个修复已经写好，在分支 `fix/dp-router-visible-cores` 上，带测试，尚未合并。
4. **后置，随 `throughput` 一起做：DP 路由器按 parser 推导要转发的 flag**，取代现在的手写名单。原则是除了路由器自己消费的那几个，其余全部转发。
5. **把 `main()` 里的参数校验段提成一个函数**，bench 在启动任何子进程之前调用，使错误组合在一秒内失败。
6. **注册 `bench` 子命令。**

## 9. 结果记录

```
benchmark/results/<设备>/<recipe>[+<指纹>]/<run_id>.input.json   # 启动前写
benchmark/results/<设备>/<recipe>[+<指纹>]/<run_id>.json         # 结束时写
$DIFFLET_BENCH_ROOT/<run_id>/                                     # 日志、输出、dump、临时编译缓存
```

一次 `bench run` 对一个 recipe 写一份结果，包含这次测的所有 metric。设备名由实例型号推导，可以显式覆盖。backend 是 recipe 的一部分，所以路径里不单列。

目录名有三种：`<recipe>` 是没有任何覆盖的运行；`<recipe>@<suite>.<variant>` 是 suite 变体；`<recipe>+<指纹>` 是带命令行覆盖的运行。

所有人的机器都往同一个进了 git 的目录里追加，靠 `run_id` 区分。每条结果记录一个主机标识，用来区分不同人的机器，不写主机名。

结果的内容：

- `schema_version`、测量方法版本、`run_id`、时间。
- recipe：名字、recipe 文件的哈希、覆盖内容、参数指纹。
- 请求的参数，以及子进程报告的最终生效参数。两者不一致时结果无效。
- 每个 metric 的样本、统计量、有效性。
- 出处：git commit 和是否有未提交改动；完整的已安装包列表；Neuron SDK 的 wheel 版本和系统包版本；实例型号；绑定的核和 CPU；按前缀白名单采集并过滤掉密钥类变量的环境变量快照；完整命令行；缓存目录的解析后路径；编译产物 manifest 里的 `cache_inputs`。
- 日志和输出文件的路径与 sha256。
- 一条可以直接执行的复现命令。

规则：

- 只追加，不覆盖，不原地 patch。写文件用"临时文件加重命名"。
- 输入配置在启动前落盘，编译失败时也能查到用了什么参数。
- 无效的运行写明无效，并以非零退出码结束。不写全零的占位结果。
- 小的 JSON 进 git；日志、媒体、dump 留在 `$DIFFLET_BENCH_ROOT`，不进 git。

## 10. 环境与设备

- **主机配置只有三个环境变量，只在一处读取并有默认值**：`DIFFLET_BENCH_ROOT`、`HF_HUB_CACHE`、`DIFFLET_COMPILE_CACHE`。其余文件不许再写路径。解释器就是启动 bench 的那个。
- **基准环境。**【用户定】PyTorch 2.14。【我的选择】具体取 AWS SDK 自带的 torch-neuronx 2.14 环境：Python 3.12.3、torch 2.14.0、torch-neuronx 2.14.3.0.1、neuronx-cc 2.28.7631。它是这台主机上实际跑数的环境，同一 AMI 的每台 trn3 上都有，而且主机上的 torch 只有给 Python 3.12 的构建。
- **依赖锁定。** 把 repo 里的 neuron 锁文件更新为上面的基准环境。完整的包列表随每条结果保存，`--check-env` 以它为参照。
- **CPU 和线程数。** `--cpus` 必须显式给出并记录。线程数默认等于 CPU 集合的大小，recipe 可以用 `threads` 固定它。这是测量定义的一部分：quant_bench 的记录显示 48 线程和 96 线程会让 HunyuanVideo-1.5 生成出不同的片段，主机侧的时间也不同。线程数不同的两份结果，`compare` 要标出来，输出哈希相等的规则也只在线程数相同时适用。
- **`--check-env`。** 对比本机与参考结果的包版本、SDK 版本、实例型号，列出差异。
- **环境清洗。** 启动器丢掉继承来的引擎相关变量：`DIFFLET_*`、`TORCH_NEURONX_*`、编译缓存位置、线程数。开关只能从 recipe 进。这是把现有 neuron adapter 的做法提到公共层。
- **设备。** 由调用者用 `--devices` 显式指定，bench 记录到结果里。启动前检查设备是否被占用，被占就拒绝。bench 不自动挑设备。
- **进程。** 子进程以新 session 启动，清理时只结束自己记录的进程组。

## 11. `compare` 和 `report`

**`compare A B`**

- 标题是两份结果的参数差异。
- 测量方法版本或同步方式不一致时拒绝比较。
- 延迟类指标和吞吐类指标各按自己的方向算比值。
- 对照臂多次重复的输出哈希必须一致，否则整组无效。
- 每步时间用来解释，端到端时间用来裁决。跨机器只比每步时间。
- 报告均值、样本数、每个样本。

**`report`**

- 从结果 JSON 生成每个 recipe 的报告和每台设备的汇总页，取代手写的 `RESULTS.md`。
- 主表只收没有命令行覆盖的有效运行。带覆盖的进单独一节并显示差异。
- 【我的选择】同一个 recipe 有多份有效结果时，主表显示时间最新的一份，并注明还有几份更早的结果和它们来自几台主机。加 `--all` 列出全部历史。
- 统一渲染"与默认值不同的参数"，不为任何功能写专门的列。

## 12. 对 bench 自身的测试

放在 `tests/unit/bench/`，全部能在现有的 CPU CI 里跑：

- 用假时钟固定每个 metric 的计时窗口边界。
- recipe 的翻译、校验、路由、三层优先级。
- 自管 flag 的拒绝规则。
- 结果 schema 和必填的出处字段。
- suite 的展开、选择、续跑判定。
- 有效性规则：发生编译的 warm 样本、哈希不一致的对照臂、参数不一致的运行。

## 13. 实施步骤

每一步一个提交并推到远端。

| 步 | 内容 | 需要真机 |
|---|---|---|
| 0 | `difflet/bench/` 骨架、子命令、`recipes.yaml`、翻译与路由、`list`、`--dry-run` | 否 |
| 1 | 结果记录、出处采集、`--repro` | 否 |
| 2 | 环境：清洗、主机配置、`--check-env`、设备和 CPU 绑定、占用检查 | 是 |
| 3 | 引擎的结构化计时输出；校验函数提取 | 是 |
| 4 | `compile`、`e2e_cold`、`e2e_warm`、`dit_step`、`memory` | 是 |
| 5 | `quality`，含全部模型的 latent 和编码前帧 dump、`bench reference`、冻结参考 | 是 |
| 6 | `report` 和 `compare` | 否 |
| 7 | suite | 是 |
| 8 | 清理和文档更新。只有用户明确说执行才做 | 否 |
| 后置 | `throughput`：先离线批量，再在线压测。现在只登记，用户说开始才做 | 是 |

第 3 步完成后做一次对照验证：同一配置打开和关闭计时输出交错各跑至少 5 轮，差异落在轮次间自然抖动之内才算通过。

## 14. 清理清单（只登记，不执行）【用户定】

**与 bench 无关、现在就能删**

- `benchmark/backfill_breakdowns.py`
- `difflet/utils/benchmark.py`，连同 pyproject 里对应的一行

**随对应能力落地后删**

| 落地的能力 | 可删的文件 |
|---|---|
| neuron 上全部 metric 对齐 | `benchmark/neuron_cold_warm.py`、`neuron_wan_bench.py`、`neuron_report.py`、`adapters/neuron.py`、`tests/unit/benchmark/` 下三个 neuron 测试文件 |
| suite 加 `compare` 加 `quality` 加 `memory` | mxfp8 分支上的 `scripts/quant_bench/`，保留其依赖锁文件 |
| `quality` 加 `compare`，并确认 TAEF1 在 neuron 上可用 | `scripts/bench_taef1_compare.py` |

**AOT 专用，第一版不替代，等 AOT 纳入 bench 之后再删**

`benchmark/cold_warm_e2e.py`、`step_latency.py`、`step_realloop.py`、`parse_compile.py`、`parse_generate.py`、`adapters/trainium.py`，以及 `scripts/flux_parallel_sweep.py`、`parallel_phase_sweep.py`、`flux_baseline_28.sh`。

**候选，只登记**【用户定】

13 个一次性的 TeaCache 实验脚本：`scripts/run_*teacache*.py`、`run_fused_e2e_ab.py`、`run_qwen_online_delta_e2e.py`、`run_qwen_fixed_cadence_ab.py`、`verify_teacache_speedup.py`。它们的端到端 A/B 部分 bench 能替代，校准和信号门限部分不能。

**逻辑搬进 `difflet/bench/`，脚本文件消失**

`scripts/quality_vs_reference.py`、`hv15_frame_metrics.py`、`compare_difflet_runs.py`、`cpu_fp32_reference.py`，以及它们的两个单元测试。

**必须保留**

- AOT、H100、B300、TPU 还在用的：`benchmark/{bench,harness,models,report,warm_e2e}.py`、trainium、diffusers 和 TPU 的 adapter、两份 `drive_all.sh`、`benchmark/tpu_profile/`。
- 不是 benchmark 的：`scripts/verify_cli.py`、各种 smoke 和 parity 脚本、`profile_*.py`、`tests/numerical/neuron_benchmark.py`。
- `scripts/seed_planner_measurements.py` 是结果的下游读取者，改读取路径，不删。

**删之前要先做的**

- 旧 `bench.py` 里的 neuron 分支随 neuron adapter 一起去掉。
- `neuron_cold_warm.py` 独有的检查先搬成 bench 的有效性规则。
- 过渡期 `MATRIX` 和 `recipes.yaml` 并存。`MATRIX` 里的 `wan_2_1_neuron` 迁走后删除，其余 7 条 AOT 条目留给旧 harness。
- quant_bench 的报告里有一行"warm 的第 2 个请求"，这就是被去掉的 `e2e_resident`。第一版没有对应的数字，替换 quant_bench 之后这一行会消失，直到后置的离线 `throughput` 把它作为副产品带回来。

## 15. 文档引用更新【用户定：改成新命令】

共 54 份文档引用旧命令。第一版只做 neuron 之后，其中绝大多数引用的是 AOT、H100、TPU 的命令，这些命令继续有效，不能改。

| 文档 | 数量 | 做法 |
|---|---|---|
| 引用 neuron 路径旧命令的：`benchmark/trn2/wan_2_1_neuron.md`、`ltx_2_neuron.md`、device-verify skill 的功能目录 | 3 | 对应脚本删除时改成 `difflet bench` |
| 只引用 AOT、H100、TPU 命令的 | 51 | 第一版不动，等各自的 backend 纳入 bench 时再改 |

【我的选择】只改指向已被替代的命令的文档，否则文档会指向跑不通的命令。

## 16. 决定记录

**用户定的**

1. 入口是 Difflet CLI 的子命令，名字是 `difflet bench`。
2. 用参数选 metric 和模型。
3. 功能 flag 从 compile / generate 的 CLI 自动继承。
4. parallelism 包括 DP。
5. 第一版只做 PyTorch-native 的 neuron backend，AOT 先不做。
6. 加内存指标。
7. 文档引用改成新命令。
8. 待删脚本只登记，暂不删除。13 个 TeaCache 实验脚本也只登记。
9. neuron 的基准环境用 PyTorch 2.14。
10. recipe 用 YAML 配置。
11. suite 保留。
12. 不要 `e2e_resident`。
13. `throughput` 离线和在线两种都实现，但放在最后，现在只登记不做。
14. latent 和编码前帧都 dump。

**我的选择，可以推翻**

1. 在用户列的四个 metric 之外增加 `compile`。
2. 允许命令行覆盖 recipe，覆盖的运行单独标记。
3. recipe 和 suite 的文件随包存放在 `difflet/bench/` 下。
4. `dit_step` 取自 `e2e_warm` 的运行，不另起进程；`memory` 也在这些运行里采样。
5. `quality` 在编码前帧上算；`cpu_fp32` 参考的生成脚本先只有 Qwen-Image 和 LTX-2，其余逐个补。
6. `e2e_cold` 用逐文件驱逐代替全局清缓存。
7. bench 不自动挑设备。
8. AOT、H100、TPU 的文档命令暂时保留，只改指向已被替代命令的 3 份文档。
9. 第一版不内置任何性能或质量门限。
10. backend 是 recipe 的必填项；`dp` 大于 1 的 recipe 留给后置的 `throughput`。
11. suite 的续跑以命名的实验为单位；`order: ABBA` 作用于计时类进程的启动顺序，每臂只编译一次。
12. 报告主表显示每个 recipe 最新的一份有效结果。
13. `--cpus` 必须显式给；`--quality-ref` 必须显式给或由 suite 提供；CPU fp32 参考存在 bench 根目录下，靠拷贝共享。
14. 基准环境的 Python 取 3.12.3，即 AWS SDK 自带环境的版本。
15. 线程数默认等于 `--cpus` 的大小；输出依赖线程数的 recipe 用 `threads` 固定，比如 HunyuanVideo-1.5。
16. 后置的 `throughput` 先做离线、再做在线；离线的算法见第 7 节末尾。

**未决**

没有。讨论中提出的问题都已答复。

## 17. 留给实施者的

- `difflet/bench/` 内部的模块划分。
- 计时输出的开关名和文件格式，只要包含第 8 节列的字段。
- 环境变量白名单的具体前缀和密钥过滤规则。
- `run_id` 和参数指纹的具体格式。
- 驻留比例的测量手段和"驱逐已验证"的判定阈值。
- 各自检阈值的初值，以真机数据校准后写进文档。

## 18. 风险

- **引擎计时输出动的是产品 CLI。** 需要确认默认关闭时没有任何行为变化，并通过第 13 节的对照验证。
- **后置的 `throughput` 要动 DP 路由器。** 分核修复和 flag 转发都需要在真机上验证多副本启动不回归。
- **补 dump 要改五个 pipeline。** 这是第一版里对产品代码改动面最大的一项，需要确认 dump 关闭时没有任何行为和性能变化。
- **共享主机上的冷启动经常无法验证。** 别的会话映射着同一份权重时驱逐会部分失败，样本会被标为"驱逐未验证"。
- **`neuron_cold_warm.py` 是 Wan 在 neuron 上唯一有 68 个单元测试保护的测量路径。** 删它之前，bench 必须在同一份 recipe 上跑出一致的数字。
- **每秒采样会漏掉短于一秒的内存尖峰。** 靠进程内峰值核对来发现。
- **CPU 与设备即使都是 fp32，位置编码的初始化也可能有细微差别。** `quality` 的门限要先测参考自身的重复性再定。
- **过渡期两套 harness 并存。** 旧 harness 冻结期间不再接受新功能。

## 附录：调研出处

设计参考了对 vLLM、vLLM-Omni、SGLang、SGLang-Omni、AWS Neuron 官方工具的调研，共 14 份报告、280 条结论，每条都由独立的核对者对照源码检查过，无一被推翻。没有一家同时做到统一入口和结果可复现。本设计借用的做法：

| 做法 | 出处 |
|---|---|
| recipe 就是引擎自己的 flag，展开成参数交给引擎的 parser | vLLM 的 `--config` 和 bench 子命令；SGLang 的 preset 脚本 |
| suite 是"默认值加具名运行列表"，结果存在则跳过 | vLLM 的 CI 测试清单和 `bench sweep` |
| 计时埋在引擎里并写成文件 | SGLang 的 `--perf-dump-path` |
| 逐文件驱逐 page cache | vLLM-Omni 的启动基准 |
| "warm 不得编译"的硬保护 | AWS `libtorch_neuronx_lite` 的缓存命中断言 |
| 请求的配置与实际生效的配置分开记录 | AWS `neuron-campaign` |
| 环境变量分"强制"和"仅未设置时生效" | AWS `vllm_omni_neuron` 的 env profile |
| 对照、候选、候选、对照的顺序，以及用输出哈希判定对照臂有效 | SGLang 的 preset 脚本 |
| 测量方法带版本号 | SGLang 的 nightly 对比脚本 |
| 给 benchmark 代码本身写假时钟单元测试 | SGLang-Omni |
| 质量在编码前帧上算，报均值和最小值 | vLLM-Omni 的 LTX 精度测试 |
| CPU fp32 参考从注入的初始 latent 起步 | AWS 的精度评估文档；repo 现有的 `cpu_fp32_reference.py` |

明确不采用的做法：用 shell 拼命令字符串的 runner；清理时杀全机进程；把绝对毫秒基线提交进仓库当跨机器标准；结果固定文件名覆盖写；失败时写占位结果；用引擎自己某台机器上的输出当质量参考。
