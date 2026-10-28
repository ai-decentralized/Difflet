<!-- Proposed 2026-10-26, awaiting the user's approval of decisions D2-D10. Produced by a read-only audit; inferred points are marked. -->

# 四模型 native / AoT TP4 指标计划：可复用数据、需重测项与执行方案（待批准）

审计范围：worktree `/home/ubuntu/Difflet/.claude/worktrees/hunyuan-neuron`（feat/hunyuan-neuron，HEAD 6b456a2）和 origin/main 540e5f0。这次只读了代码和数据，没有改代码、没有提交，也没有碰 Neuron 设备。标注 (inferred) 的条目是从代码或数据推出来的，没有实测。

## 1. 结论摘要

- **native 计时数据：3 个模型可复用，LTX-2 要全部重测。**
  - 可复用：HunyuanVideo、Qwen-Image、Wan2.1 的 DiT step、warm e2e、cold e2e、compile（50 steps，TP4，bf16 DiT）。
  - 双核 attention 已核实，不用改代码。三个模型的 native NEFF 都拆成 `sg00 attention_cte_lnc0` 和 `sg01 attention_cte_lnc1` 两个子图，结构和 AoT 的 `attention_cte[2]` 一样（例如 `/home/ubuntu/bench-2026-10-25/roots/wan_native_tp4/neuron/neff/17/78/17788144....neff`）。
  - 这些数据是在较早的 commit 上测的：HV 在 eadddec，Qwen 在 4f6e1de，Wan 在 8b2408c。从那之后的改动只碰了 HV 专用路径 (inferred)。每个模型补跑 1 次 quality generate，用 hash 核对就能确认数据仍适用于 HEAD。
  - LTX-2 native 的数据是在另一台实例、旧工具链上测的，原始日志不在本机（ltx_2_neuron.md:7），所以不能复用。
- **AoT 全部 4 个模型、所有指标都要在 origin/main 540e5f0 上重测。**
  - 本分支的 AoT 数据都早于 bf16 修复。修复包括 fc6a580、b4bdf35（Qwen）、4d4cc60（LTX-2）、7fe5308（HV）、1c8a64e（Wan umT5）。修复前 Qwen、LTX-2、HV 的 DiT all-reduce 是 fp32，Wan 的 umT5 all-reduce 也是 fp32。
  - 可以作为参照、但不能直接用的数据：
    - Qwen 361.9 ms/step 和 PSNR 24.72 dB / SSIM 0.957：代码等同 main，但只跑了 20 steps。
    - LTX-2 433.5 ms：单核 attention，20 steps。
    - Wan 955.8 ms：修复前测的，但 Wan DiT 的 all-reduce 本来就是 bf16，所以预计不变 (inferred)。
- **PSNR / SSIM：四个模型、两个后端都没有在 50 steps 下对比过 CPU 参考，必须全部新跑。**
  - native 和 AoT 在 torch 2.14 和 torch 2.9 下抽出的 bf16 噪声不同。所以 Qwen、LTX-2、HV 各需要 2 个 CPU fp32 参考（native 噪声一个、AoT 噪声一个）。
  - Wan 两边都用 fp32 抽噪声，结果一致（已实测 sha 16aeaed7），1 个参考就够。
  - 合计 7 个 50-step CPU 参考，约 13.5-15.5 h CPU (inferred)。
- **最大的阻塞点：**
  - (a) 磁盘。新建的 AoT/native root 约 216-246 GB，`/` 只剩 215 GB 可用。
  - (b) LTX-2 native 默认把 Gemma 放在设备上，但现有 snapshot 的 text_encoder index 导致载入失败（找不到 `language_model.model.*` 键，CPU 已验证）。不改代码的绕法是 shadow HF cache。
  - (c) CPU 参考约 14 h，只能串行跑，而且不能和设备计时重叠。
  - (d) 不改代码时，native 有 3 个组件做不到 bf16：Wan umT5、Wan VAE、Qwen VAE。
- **AoT 必须从 origin/main 的独立 detached worktree 跑。** 设 `PYTHONPATH`，不要 merge：dry-run 显示与本分支冲突 21 个文件。AoT venv 的 editable install 默认指向另一个 worktree，必须用 PYTHONPATH 覆盖。

## 2. 总表

数据来源缩写：
- N25 = benchmark/trn2/native_vs_aot_tp4_2026-10-25.md
- HVN = benchmark/trn2/hunyuan_video_neuron.md
- CB = /home/ubuntu/campaign/bf16-reduce
- 统一 workload：seed 42，prompt "a cinematic shot of a red fox running through a snowy forest"，TP4，50 steps。

| 模型 / 后端 | DiT step | warm e2e | cold e2e | compile | PSNR | SSIM |
|---|---|---|---|---|---|---|
| **HV1.0 native**（320x512x61，g6.0） | 可复用（808.2 ms，HVN:20；B26/hunyuan_video/s4_tp4/gen1-3） | 可复用（126.7 s，n=3，HVN:22） | 可复用（457.8 s，HVN:23,92；含 rank3 6.4 s 小编译） | 可复用（274.5 s，fresh root，HVN:21,94） | 需重测（现有 56.08 dB 只比了 VAE，HVN:96-104；没有 DiT 端到端 CPU 参考） | 需重测（没测过） |
| **HV1.0 AoT** | 需重测（813.9 ms 是修复前 d6ffb9a 测的，7fe5308 改了 84 个 DiT all-reduce） | 需重测（132.9 s，修复前，只有 n=2） | 需重测（617.3 s，修复前；测量范围本身合格） | 需重测（4347.7 s 是修复前测的，而且是下限：复用了 shared weights） | 需重测（没测过；AoT VAE tiling 会把 PSNR 封顶在约 42.5 dB，HVN:319） | 需重测 |
| **Qwen-Image native**（1024²，g4.0） | 可复用（359.0 ms，N25:52） | 可复用（82.3 s，N25:55） | 可复用（502.0 s，N25:56） | 可复用（187.0 s，N25:54） | 需重测（只有 device-vs-device 结果，qwen-image-neuron-evidence.md:222） | 需重测 |
| **Qwen-Image AoT** | 需重测（418.2 ms 是修复前测的；361.9 ms 是修复后但只有 20 steps，CB/runs/qwen/summary.json，跑新 cell 时会顺带测到） | 需重测（75.4 s 修复前；60.6 s 是 20 steps） | 需重测（504.2 s 修复前；498.8 s 是 20 steps） | 需重测（1238.5 s 修复前；1091.3 s 复用了 NEFF） | 需重测（24.72 dB 是 20 steps，CB/runs/qwen/quality_bf16_vs_cpu.json；只在 20-step 降级方案里可用） | 需重测（0.957，同上） |
| **Wan2.1 native**（480x832x9，g5.0 CFG） | 可复用（1063.6 ms，N25:195） | 可复用（128.5 s，N25:198） | 可复用（542.7 s，N25:199；DiT 从 bf16 weight store 读；含约 6 s 小编译） | 可复用（600.3 s，N25:197） | 需重测（没有 Wan CPU 参考；保留下来的 latents.pt 可以算 latent 指标） | 需重测 |
| **Wan2.1 AoT** | 需重测（955.8 ms 修复前测的，预计不变 (inferred)；main 的 stage key 变了，强制重编译） | 需重测（120.0 s 用的是 `--host-vae` fp32，而且是修复前） | 需重测（447.5 s 同上） | 需重测（445.1 s 修复前，不含 device VAE 约 1.5 h 的编译） | 需重测 | 需重测 |
| **LTX-2 native**（480x704x49，g1.0） | 需重测（350.7 ms 是另一台实例、旧工具链测的，无原始日志，ltx_2_neuron.md:17） | 需重测（90.8 s，同上） | 需重测（588.3 s，同上） | 需重测（306.6 s，同上，neuron_final.md:183） | 需重测（只有 device-vs-device 结果，ltx_2_neuron.md:164） | 需重测 |
| **LTX-2 AoT** | 需重测（433.5 ms 是单核 attention、20 steps；419.2 ms 双核但修复前） | 需重测（54.8 s：单核，20 steps） | 需重测（767.5 s：单核，20 steps） | 需重测（555.9 s 复用了 pre-sharded weights） | 需重测（34.14 dB 是 20 steps、单核，而且是 mp4 对 mp4） | 需重测（0.939，同上） |

小计：可复用 12 格（HV、Qwen、Wan 的 native 计时）。需重测 36 格，其中 PSNR/SSIM 16 格全部需重测。

## 3. 与用户要求的冲突 / 需要你决定的点

**D1. native 双核 attention：不需要改代码，也不需要你决定。**
- 依据：native NEFF 的 lnc0/lnc1 子图结构见第 1 节。
- `ops_impl/attention.py:168` 用 `grid=(get_logical_neuron_cores(),)`，这台机器 LNC=2。
- 每次 native 编译都带 `--lnc 2`（probe JSON，例如 `/home/ubuntu/bench-2026-10-25/hunyuan_video/tp4_default/compile/probe/663812-rank1.json`）。
- 81ea51e 记录的"可能需要改代码"这一条可以关闭。
- LTX-2 native 走同一条路径，但本机没有它的 NEFF (inferred)。编译后扫一次 NEFF 就能确认。
- AoT 端各模型的双核 attention：
  - Qwen 和 HV 在 main 上默认就是 `_VIRTUAL_CORE_SIZE=2`（origin/main qwen_image.py:34，hunyuan_video.py:31）。
  - Wan 和 LTX-2 必须 export `NEURON_RT_VIRTUAL_CORE_SIZE=2`，编译和每次 generate 都要。
  - cache key 不包含这个变量，所以 root 必须是新的；还要核对 DiT 的 HLO hash 确实变了（Wan 不应再是 MODULE_c984433f，LTX-2 不应再是 MODULE_20da5719）。

**D2. 不改代码时做不到 bf16 的组件（都在 native 端）**
- Wan umT5 是 fp32，写死在 `difflet/models/wan/neuron_application.py:329`。text_encoder.py:10-13 说明了原因：bf16 下数值脆弱。
- Wan VAE 是 fp32（`cli/orchestrators/wan.py:470-478` 用的是默认 dtype）。
- Qwen VAE 是 fp32（`qwen_image.py:880-886`）。
- 建议：接受这些偏差，在报告里注明。代价是 native Wan cold e2e 多出 172.5 s 的 fp32 umT5 读取，quality 和 e2e 也会受影响。
- 另一个选择：VAE 调用处各改 1 行（`dtype=torch.bfloat16`），但这违反"不动 code"，bf16 数值也没验证过。umT5 不建议改。
- AoT 在 main 上四个模型的所有组件默认都是 bf16。
- 为保持 bf16，要避开两个会退回 fp32 的选项：Wan 和 HV 的 `--host-vae`。

**D3. AoT 用 latest main**
- 新建 detached worktree：`git -C /home/ubuntu/Difflet worktree add --detach /home/ubuntu/aot-main origin/main`。
- 确认里面包含 7fe5308、1c8a64e、4d4cc60、b4bdf35。
- 所有 AoT 命令都设 `PYTHONPATH=/home/ubuntu/aot-main`，并记录 `difflet.__file__`。
- 本会话的 git guard 拒绝了 `-C` 指向共享 checkout，所以这一步需要在不受该 worktree 隔离的会话里执行。
- 结果是 native（feat/hunyuan-neuron）和 AoT（main）在不同的 commit 上跑，这是结构上避免不了的（main 上没有 `difflet/backends/neuron`）。
- 已知风险：HV AoT 修复后 warm e2e 从 109.8 s 变成 123.2 s（n=1，issue #110）。

**D4. LTX-2 native 的 snapshot index**
- 问题：`build_weight_map`（`backends/tpu/core/checkpoint.py:34-60`）先选中了 `diffusion_pytorch_model.safetensors.index.json`，其中 0 个 `language_model.model.*` 键，设备上的 Gemma 载不进来。
- 方案 A（推荐，不改代码）：建 shadow HF cache（symlink 整个 snapshot，删掉那个 index，设 `HF_HUB_CACHE` 只对 native 生效）。CPU 已验证这样能拿到 1065 个键，其中 626 个是 `language_model.model.*`。`snapshot_download` 是否尊重 `HF_HUB_CACHE` 需要先跑一次 CPU 调用确认。
- 方案 B：`--host-text-encoder`。仍是 bf16，但 placement 不同，warm e2e 也会变慢。
- 方案 C：改代码修复。

**D5. cold e2e 的精确定义（建议如下，需要你确认）**
- 计时前 `sync; echo 3 > /proc/sys/vm/drop_caches`，确认 Cached 约 0.2 GiB。
- 每次是全新进程、全新 `--work-dir`，跑完全部 50 steps，text encoder、DiT、VAE 都在计时内从磁盘读。不复用任何 embeddings、latents 或 stage 输出。
- 编译产物允许存在：NEFF / NKI cache，以及"compile 阶段写出的权重格式"（native 的 bf16 weight store、AoT 的 pre-sharded weights）。理由是两边对称，而且 AoT 没有不改代码就能关掉 pre-shard 的开关。
- 如果你认为 weight store 也算复用：
  - native 的 Wan 和 LTX-2 改用 `DIFFLET_NEURON_WEIGHT_STORE=0`。LTX-2 以前直读实测是 758 s，用 store 是 577 s（20 steps，neuron_feature_matrix.md:144）。
  - AoT 做不到。
  - native 的 HV 和 Qwen 本来就直读 checkpoint（"no gain"），不受影响。
- 已知瑕疵：native HV cold 里有 6.4 s、Wan cold 里有约 6 s 的首次小编译。可以接受并标注，也可以每个模型补跑一次 cold（约 10 min 一次）。

**D6. CPU 参考的成本**
- 50-step fp32 参考，每个的估算：
  - Qwen 约 1.35 h，需 2 个（CB qwen 20-step 实测 2198 s，稳态 82 s/step）
  - LTX-2 约 1.3 h，需 2 个（实测 79 s/step）
  - Wan 约 2-2.5 h，需 1 个 (inferred)
  - HV 约 3-3.8 h，需 2 个（从 5-frame 的 25 s/step 外推 (inferred)）
  - 合计约 13.5-15.5 h
- 内存峰值约 60-82 GB，只能串行跑，不能和设备计时重叠。
- 可选方案：
  - (a) 全部 50 steps，一个通宵（推荐，和计时 workload 一致）。
  - (b) 只把 quality 降到 20 steps，约 5.5-7 h。Qwen 和 LTX-2 的 AoT-noise 20-step 参考可以复用（先核对 initial.pt；LTX-2 参考要从 final_video.pt 无损重解码）。PSNR/SSIM 标注为 20-step。
  - (c) 混合：Qwen、LTX-2、Wan 用 50 steps，HV 用 20 steps。省约 3.5-4 h。
- 无论选哪个，都建议加一个可选的 CPU bf16-vs-fp32 对照（借助 AMX 多花约 35-40% 时间），作为"bf16 漂移基线"。参考值：Qwen 对 CPU 只有约 24.7 dB，原因之一是 CPU 参考用 fp32 文本编码和 masked attention（evidence.md:153）。

**D7. 磁盘**
- 新 root 预计需要：Qwen AoT 53 GB、LTX-2 AoT 39 GB + native 24 GB、HV AoT 60-90 GB、Wan AoT 40 GB，合计约 216-246 GB，`/` 只剩 215 GB。
- 方案 A：批准删除修复前的 AoT root：
  - `bench-2026-10-25/roots/qwen_aot_tp4`（53 GB）
  - 两个 Wan AoT root（约 78 GB）
  - HV `aot_tp4`（59 GB）
  - 以及 `~/.cache/difflet/_shared_weights` 里的旧条目
- 方案 B：每个 AoT cell 跑完（包括 quality generate）就删掉它的 root。
- 两种都需要你批准。

**D8. 改不了的 placement 差异**
- native 在 host 上编码文本：Qwen、HV，以及 LTX-2 走 `--host-text-encoder` 时。AoT 在设备上编码文本，但 LTX-2 AoT 是在 host 上编码。
- 这些差异会直接体现在 e2e 和 quality 里，报告中要注明。
- AoT Wan 用默认的 device bf16 VAE（建议）。代价是多约 1.5 h 编译，而且这个 VAE 从没在 VCS=2 下编译过。

**D9. 测量工具（不在产品代码里）**
- 需要写或改的东西：
  - native Qwen 和 LTX-2 的噪声 / latents 捕获，通过扩展 harness probe（sitecustomize）实现。
  - HV 和 Wan 的 CPU 参考脚本：main 的 `scripts/cpu_fp32_reference.py` 只支持 qwen_image 和 ltx_2。
  - LTX-2 参考输出改成无损（现在是 mp4）。
  - 让 `quality_vs_reference.py` 能读 `.pt` 的 adapter。
- 都放在 repo 之外的 bench 目录。请确认这符合"不动 code"。

**D10. native 数据复用**
- 方案 A（推荐）：用 HEAD 上的 quality 运行做 hash 核对：
  - Qwen：png b1906d53 / latents 4da4c89e
  - Wan：latents 3cafa271
  - HV：latents b62d0891
- 方案 B：当天全部重测，额外约 1.5 h 设备时间。

## 4. 需要跑的东西

通用设置：
- AoT：venv `/home/ubuntu/venvs/difflet-neuron`，`PYTHONPATH=/home/ubuntu/aot-main`，每个 cell 用新 root `R`，并设 `DIFFLET_SHARED_WEIGHTS_DIR=$R/_shared_weights`、`NEURON_COMPILE_CACHE_URL=$R/neuron-compile-cache`、`HF_HUB_OFFLINE=1`。
- native：venv `/home/ubuntu/venvs/native-312`，`PYTHONPATH=<probe>:<hunyuan worktree>`，`DIFFLET_COMPILE_CACHE=$R`。
- 每次 generate 都显式传 `--steps 50` 和 guidance：AoT 的 HV 和 Qwen 默认只跑 4 steps，LTX-2 默认 40 steps、guidance 3.5。
- compile 前统一做一次 drop cache，再把 checkpoint 读两遍进 page cache。
- quality 运行都用 `--output out.pt`（Qwen 用 png），不要用 mp4（H.264 噪声底约 33-38 dB，HVN:255）。

**HunyuanVideo 1.0**（`--model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa6... --height 320 --width 512 --num-frames 61 --tp-degree 4 --steps 50 --guidance-scale 6.0 --seed 42`）
1. AoT `aot_main_tp4`：preread（40 GB，约 5 min）→ `compile --backend trainium`（约 75-85 min (inferred)）→ cold（约 10.5 min）→ gen0 + gen1-3（各约 2.3 min，step 用 probe 的 `_scheduler_step` 时间差算）→ gen_q（`--output out.pt`，约 2.3 min）。设备约 1.8-2 h，磁盘 60-90 GB。不需要设 VCS 或 bf16 参数。harness 从 `/home/ubuntu/bench-2026-10-26/s4/scripts/hvbench_s4.py` 复制，把 WT 改为 aot-main，先跑 probe 自检。
2. native `native_q`：在 root `bench-2026-10-26/roots/s4_tp4` 上跑 1 次 generate，`--output out.pt`，核对 latents b62d0891。约 2.2 min。
3. CPU 参考 A（native 噪声：`Generator().manual_seed(42)` 抽 fp32，形状 (1,16,16,40,64)）：diffusers `HunyuanVideoPipeline` 全 fp32，先编码文本、释放 text encoder，再载入 DiT，最后用 tiled VAE 做 fp32 decode 并存成 `.pt`。约 3-3.8 h。
4. CPU 参考 B（AoT 噪声）：在 AoT venv（torch 2.9.1）里执行 `torch.manual_seed(42); torch.randn(1,16,16,40,64,dtype=bf16)`，然后 `latents=x.float()`。约 3-3.8 h。AoT 不保存初始 latents，所以只能比帧指标，而且这个噪声复现无法直接核对 (inferred)。

**Qwen-Image**（`--model-id Qwen/Qwen-Image --revision 75e0b4be... --height 1024 --width 1024 --steps 50 --guidance-scale 4.0`）
1. AoT `aot_main_tp4 --cp-degree 1`：preread（53.7 GiB，约 7.5 min）→ compile（约 1240 s）→ cold（约 505 s）→ gen0 + gen1-3（各约 73 s）。设 `DIFFLET_DUMP_LATENTS_DIR=<run>` 和 `DIFFLET_STEP_TIMES_FILE=<run>/steps.jsonl`，gen1 同时作为 quality 候选样本。设备约 50 min，磁盘约 53 GB。
2. native `native_tp4_q`：跑 1 次 generate，probe 包住 `QwenImageOrchestrator._denoise`，dump initial.pt 和 final.pt，核对 png b1906d53 / latents 4da4c89e。约 3 min。
3. CPU 参考 D1（AoT）和 D2（native）：`aot-main/scripts/cpu_fp32_reference.py --model qwen_image --initial-latents-dir <run> --steps 50`。各约 1.35 h。D1 可以先用 `CB/runs/qwen/bf16/warm1/initial.pt` 启动，跑完后再和新 dump 逐字节比对。
4. `quality_vs_reference.py` 计算 PSNR、SSIM、latent rel-L2 和 cosine。

**Wan2.1 14B**（`--model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498c... --height 480 --width 832 --num-frames 9 --steps 50 --guidance-scale 5.0`）
1. AoT `aot_main_tp4_twocore`：所有步骤 export `NEURON_RT_VIRTUAL_CORE_SIZE=2`，不加 `--host-vae`（device VAE 是 bf16）。preread（约 11 min）→ compile（transformer 约 445 s + VAE 约 5500 s (inferred，按 Wan2.2 同配置 VAE 的 5488.5 s 估)，约 1.7 h）→ cold（约 7 min）→ gen0-3（各约 1.6 min）→ Q-run（`out.pt`，约 2 min）。记录 DiT HLO hash，和 MODULE_5ae08501 对比。设备约 2.1 h，磁盘约 40 GB。可选 A2：同一 root 加 `--host-vae` 跑 cold + warm（约 17 min），便于和 10-25 的数据衔接。
2. native：在 root `bench-2026-10-25/roots/wan_native_tp4` 上跑 1 次 generate，`--output out.pt`，核对 latents 3cafa271。约 3 min。
3. CPU 参考（1 个，两个后端共用）：基于 `origin/feat/torch-native-backend:tests/manual/wan_e2e_reference.py` 改写，Wan2.1、fp32、g5.0、negative prompt 为 `""`，`latents=torch.randn((1,16,3,60,104),generator=Generator().manual_seed(42))`。约 2-2.5 h，外加约 10 min 冷读 74.9 GiB。
4. 指标：native vs CPU、AoT vs CPU，再加 native vs AoT 直接对比（只有 Wan 能这样比，因为噪声相同）。

**LTX-2**（`--model-id Lightricks/LTX-2 --revision 47da56e2... --height 480 --width 704 --num-frames 49 --steps 50 --guidance-scale 1.0`）
1. AoT `ltx_aot_tp4_twocore`：export `NEURON_RT_VIRTUAL_CORE_SIZE=2`，`--cp-degree 1`，dump 和 step-times 环境变量同 Qwen。preread（约 88 GB，约 12 min）→ compile（10-17 min）→ cold（约 13 min）→ gen0 + gen1-3（各约 70 s）。可选：跑一次 20 steps，和 10-24 的 hash（mp4 edf71542，final_video b426bc92）对比，同时核对 initial_video 4458e4cf。设备约 1 h，磁盘约 39 GB。
2. native `ltx_native_tp4`：shadow HF cache，`DIFFLET_NEURON_PG_TIMEOUT_S=7200`，probe 包住 `LTX2Orchestrator._scheduler_step` 和 `_denoise`（dump initial/final video 和 audio）。先在 CPU 上跑 `pytest tests/unit/backends tests/unit/cli -k ltx`。preread（约 10 min）→ compile（约 5 min）→ cold（约 10 min）→ gen0-3（各约 90 s），然后扫 NEFF 确认 lnc0/lnc1。设备约 45 min，磁盘约 24 GB。
3. CPU 参考 3a（AoT）和 3b（native）：用 `cpu_fp32_reference.py --model ltx_2 --steps 50`，改成无损的 `.pt` 输出。各约 1.3 h。

**汇总（均为估算）**

| 项 | 设备 | CPU（参考） | 新磁盘 |
|---|---|---|---|
| HV | 约 1.9 h | 6-7.5 h | 60-90 GB |
| Qwen | 约 0.9 h | 约 2.7 h | 约 53 GB |
| Wan | 约 2.2 h（+0.3 h 可选 A2） | 2-2.5 h | 约 40 GB |
| LTX-2 | 约 1.8 h | 约 2.6 h | 约 63 GB |
| **合计** | **约 6.8 h**（可选：+1.5 h native 当天重测，+0.3 h A2） | **约 13.5-15.5 h**（20-step 降级方案约 5.5-7 h） | **约 216-246 GB（超出可用的 215 GB，见 D7）** |

## 5. 推荐执行顺序

1. **准备（不占设备，约 1 h）**
   - 拿到你对 D1-D10 的决定，特别是 D5 cold 定义、D6 CPU 范围和 D7 磁盘清理。
   - 重新 fetch，创建 `/home/ubuntu/aot-main`（540e5f0），确认修复相关的 commit 都在。
   - 复制 harness（hvbench、qbench、wbench）到 `/home/ubuntu/bench-2026-10-27`，把 WT 指向 aot-main，并在两个 venv 里跑 probe 自检。
   - 建 LTX-2 shadow cache，做一次 CPU `resolve_model_path` 验证。
   - 跑 LTX-2 的 CPU 单测。
   - 在 CPU 上复现 AoT 的噪声：Qwen、LTX-2 比对 10-24 的 dump，HV 噪声 B 生成并保存。
2. **设备块 1：native hash / quality 运行**（HV、Qwen、Wan，约 20 min）。先确认 native 数据可复用；如果 hash 不一致，就在后面加跑对应模型的完整 native cell（每个 25-30 min）。
3. **设备块 2：Qwen AoT**（约 50 min）。最短、工具最全，用来先验证 main worktree 和 harness 都正常。
4. **设备块 3：LTX-2 AoT → LTX-2 native**（约 1.8 h，同一模型的两个后端连着跑，减少 page cache 切换）。
5. **设备块 4：HV AoT**（约 2 h）。
6. **设备块 5：Wan AoT**（约 2.1 h，VAE 编译最长）。每个 AoT cell 结束后按 D7 处理 root。
7. **CPU 块（通宵，设备空闲）**，顺序：
   - Qwen D1 → D2
   - LTX-2 3a → 3b
   - Wan
   - HV A → B
   - 每个参考跑之前都核对它的 initial latents 是否和设备 dump 一致。参考之间不能并行（内存不够）。
   - 如果白天还要做设备计时，计时前要重新 preread（CPU 参考会把 page cache 清掉）。
8. **指标与报告（CPU，几分钟）**
   - 统一用 main 的 `quality_vs_reference.py` 定义：RGB uint8，每帧 PSNR 的 mean 和 worst，SSIM 按 Wang 2004（11x11 窗口，σ=1.5），再加 latent rel-L2 和 cosine（AoT HV 只有帧指标）。
   - 汇成 4×2 总表，每个数值注明 commit、日期和日志路径，并逐条标注各数值里包含的 dtype / placement 偏差（见 D2、D8）。
## Addendum 2026-10-26: native weight store in cold e2e (user question, decision pending)

The user asked whether reading the native weight store belongs to warm rather than cold e2e. The store is ordinary files on disk (`<DIFFLET_COMPILE_CACHE>/neuron/weights`); cold vs warm is decided by the page cache, so after `drop_caches` a store read is a cold disk read (fewer bytes: pre-cut bf16). It is, however, an artifact written by an earlier run, like AoT's pre-sharded bf16 files, which AoT cannot turn off without code. Two definitions:

- **A (symmetric):** both backends use their compile-time weight artifacts (native store, AoT pre-shards).
- **B (stricter for native):** native cold runs with `DIFFLET_NEURON_WEIGHT_STORE=0` (reads the original checkpoint); AoT unchanged.

Only Wan2.1 (DiT: store 217 s vs fp32 checkpoint 435 s) and LTX-2 (Gemma-3: store 178.5 s vs 358 s) differ; HunyuanVideo and Qwen-Image native write no store (bf16 checkpoints, "no gain"). Recommendation: measure both for Wan2.1 and LTX-2 native (A as the headline, B as a supplement, about +10 min each), unless the user wants B only.

**Decided by the user 2026-10-26: measure both.** For Wan2.1 and LTX-2 native, cold e2e is measured under A (weight store in use; the headline, symmetric with AoT's pre-shards) and under B (`DIFFLET_NEURON_WEIGHT_STORE=0`, original checkpoint). HunyuanVideo and Qwen-Image native have no store, so one cold number each.

## Decisions 2026-10-26 (user)

1. Cold e2e: A first (native with its weight store) and AoT for all four models; B (native without the store) only after A and AoT have results for every model. The user wants the A vs AoT comparison first.
2. Cold e2e definition (D5) approved: drop_caches, fresh process and fresh work dir, all 50 steps, text encoder + DiT + VAE weights read from disk inside the timed run, no stage outputs reused; compiled artifacts (NEFF / NKI cache, native weight store under A, AoT pre-shards) may exist.
3. bf16 exceptions on native (Wan umT5, Wan VAE, Qwen VAE): pending; the user asked for the trade-offs (answered in chat). Until decided, AoT-only work proceeds.
4. LTX-2 native Gemma index: cause confirmed (both text_encoder indexes complete since the 2026-10-24 re-download; `build_weight_map` picks `diffusion_pytorch_model.safetensors.index.json`, 0 of its 1124 keys are `language_model.model.*`). Proposed: shadow HF cache for this campaign (no code change), F12 later; awaiting the user's confirmation.
5. CPU references: all 50 steps; Qwen-Image and LTX-2 first. Order: Qwen + LTX-2 device, then their CPU references, then HunyuanVideo + Wan device, then their CPU references (the user wants CPU results early).
6. Disk: option A approved: delete the pre-fix AoT roots (`bench-2026-10-25/roots/qwen_aot_tp4`, the two Wan AoT roots, HunyuanVideo `aot_tp4`) and their stale `_shared_weights` entries.
7. Measurement tooling lives outside the repo (bench dir), approved.
8. F11 deferred.

## Decisions 2026-10-26, second round (user)

- **3 (bf16 exceptions):** keep fp32 (native Wan umT5, Wan VAE, Qwen VAE) for the device runs. After all device runs (A + AoT, four models) are done, change them to bf16 and re-test, then run the CPU references. Scope to confirm: the two VAEs (call-site dtype or a `--vae-dtype` flag) and umT5 (behind F13's accuracy gate).
- **4 (LTX-2 Gemma index):** option 2, fix the code (F12) before the LTX-2 native runs. Done now on CPU, in parallel with the phase-1 AoT device runs (unit tests run only while the device is idle).
- **Revised order:** phase 1 (running: prep, Qwen-Image AoT, LTX-2 AoT) and F12 in parallel → phase 2 device: native Qwen-Image hash/quality run, LTX-2 native full cell (A), HunyuanVideo AoT, Wan2.1 AoT, native HunyuanVideo/Wan hash/quality runs → phase 3: bf16 changes and native re-tests (Qwen-Image, Wan2.1) → phase 4: CPU references (7 × 50 steps, Qwen-Image and LTX-2 first) → phase 5: PSNR/SSIM and the report → phase 6: B (native cold without the weight store, Wan2.1 and LTX-2).

## Decisions 2026-10-26, third round (user)

- Phase 3 (bf16 changes) includes umT5 (behind F13's accuracy gate: per-prompt bf16 vs fp32 incl. the empty negative prompt, cosine >= 0.995, then a 50-step e2e; keep fp32 if it fails) and moves after the quality metrics and the report.
- Final order: phase 1 (AoT Qwen-Image, LTX-2; running) with F12 in parallel, then phase 2 device (Qwen native hash/quality, LTX-2 native A, HunyuanVideo AoT, Wan2.1 AoT, HunyuanVideo/Wan native hash/quality), then CPU references (7 x 50 steps), then PSNR/SSIM and the report, then B (native cold without the store, Wan2.1 and LTX-2; before phase 3 so it uses the same dtypes as A), then phase 3 (VAE and umT5 bf16, native re-tests of Qwen-Image and Wan2.1, reported as a supplement). Phase-3 code may be prepared during phase 2 (not wired into the default path).
- Standing instruction: proceed without asking; on a problem, apply the default recommendation and log the decision on the plan page (https://claude.ai/artifact/N1nuQ2V4rhnJLhtE2e4qVs); add results to the page as each batch finishes.

### Phase 3 device validation runbook (code prepared 2026-10-27; run after the CPU reference ends, about 22:20 UTC)

**Code (feat/hunyuan-neuron, opt-in only; every default stays fp32 and every default command line, stage key and manifest is unchanged):**
- `e809595` `WanVAEDecoderModel(fp32_output_head=True)`: for bf16 input, `conv_out` and the clamp run in fp32 and the frames come back fp32. The neuron device's written-out VAE attention computes its scores and softmax in fp32 for bf16 input. The RMS norms already normalise in fp32. This is the HunyuanVideo F2 precedent (`2e04372`, statistics in fp32) applied to Wan's norms. The Trainium model is unchanged.
- `c706e39` application switches: `TorchNeuronWanApplication(vae_dtype=, text_encoder_dtype=)` and `TorchNeuronQwenImageApplication(vae_dtype=)`, all fp32 by default. The VAE adapters un-normalise in fp32 and round only the decoder input. The weight store keys on the load dtype: bf16 umT5 writes an 11.36 GB entry (2.84 GB per rank) and fp32 umT5 writes none.
- `1832960` CLI switches, `--backend neuron` only:
  - `--vae-dtype bf16|fp32`: Wan2.1 and Qwen-Image vae stage, default fp32. Refused with `--host-vae`.
  - `--text-encoder-dtype bf16|fp32`: Wan umT5, device or `--host-text-encoder`, default fp32.
  - A bf16 component adds one field to its neuron stage key: `vae_dtype` on the vae stage, `text_encoder_dtype` on the transformer stage.
  - Both flags reach the stage processes and the DP workers.
- Check scripts (bench dir, not in the repo): `/home/ubuntu/bench-2026-10-26-4m/scripts/f13_umt5_bf16_check.py` and `.../vae_bf16_check.py`. Both were dry-run on the CPU on the unit tests' tiny random Wan (`--tiny --device cpu`). Nothing real was loaded.

**Common setup.** `B=/home/ubuntu/bench-2026-10-26-4m`, `W=/home/ubuntu/Difflet/.claude/worktrees/hunyuan-neuron`, `PY=/home/ubuntu/venvs/native-312/bin`.
- **Preconditions:**
  - the CPU reference has exited (`pgrep -f cpu_ref_` is empty and its `rc` file is 0);
  - `free -g` shows at least 100 GiB available;
  - `fuser /dev/neuron0` is empty;
  - at least 80 GB of disk is free.
- **Device order:** step 1 → 2 → 4 → 5. Steps 3 and 6 run on the CPU. Run the device steps one at a time as `systemd-run --user` units, not under a Monitor (phase-1 lesson).
- **Phase 6 (B) first:** if B (native cold without the store) has not run yet, run it before step 4. It must use today's fp32 dtypes, so leave both flags off.

**Step 1: F13, umT5 bf16 vs fp32 (device, TP4, about 15 min).**
- What runs: fp32, then bf16, through the native prompt path. That is the composed application with `text_encoder_device="neuron"` and `WanOrchestrator.encode_prompt`, without the DiT.
- Prompts: `"a cinematic shot of a red fox running through a snowy forest"` and the empty negative prompt `""`.
- The weight store is off for this check, so each dtype reads the 22.7 GB fp32 checkpoint.
```
mkdir -p $B/phase3/f13 && cd $W
HF_HUB_OFFLINE=1 PYTHONPATH=$W DIFFLET_COMPILE_CACHE=$B/roots/p3_scratch \
  $PY/torchrun --standalone --nproc-per-node 4 $B/scripts/f13_umt5_bf16_check.py \
  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --out $B/phase3/f13 --save-embeds 2>&1 | tee $B/phase3/f13/run.log
```
- **Pass:** every prompt has cosine ≥ 0.995 over its valid tokens, and the embeddings are identical on all 4 ranks (exit 0).
- **Reported:** max abs error, rel-L2, the worst per-token cosine, the same metrics after the bf16 cast the DiT sees, load seconds and per-rank bytes.
- **If it fails:** keep umT5 in fp32. Leave `--text-encoder-dtype` off in step 5 and record the numbers.

**Step 2: VAE bf16 vs fp32 decode of the same latents (device, one core).**
- The latents are the phase-2 q1 dumps.
- The script runs the CLI vae stage's exact path at fp32 and then at bf16. It loads without a warm-up, as generate does, and decodes once at each dtype.
- The first bf16 decode compiles its eager per-op kernels into `p3_scratch`. That takes minutes: the fp32 Wan vae launch compile took 169.7 s.
```
mkdir -p $B/phase3/vae_wan $B/phase3/vae_qwen && cd $W
HF_HUB_OFFLINE=1 PYTHONPATH=$W DIFFLET_COMPILE_CACHE=$B/roots/p3_scratch NEURON_RT_VISIBLE_CORES=0 \
  $PY/torchrun --standalone --nproc-per-node 1 $B/scripts/vae_bf16_check.py --model wan \
  --latents $B/wan2_1/native_tp4_hashcheck/q1/final.pt --repeat --save-frames --out $B/phase3/vae_wan
HF_HUB_OFFLINE=1 PYTHONPATH=$W DIFFLET_COMPILE_CACHE=$B/roots/p3_scratch NEURON_RT_VISIBLE_CORES=0 \
  $PY/torchrun --standalone --nproc-per-node 1 $B/scripts/vae_bf16_check.py --model qwen \
  --latents $B/qwen_image/native_tp4_hashcheck/q1/work/latents.pt --repeat --save-frames --out $B/phase3/vae_qwen
```
- **Pass:** bf16 vs fp32 device decode ≥ 50 dB mean PSNR (RGB uint8, per frame, peak 255, the campaign's `quality_metrics.py` rule). This is the bar HunyuanVideo's bf16 decoder met: 56.07 dB.
- **Also reported:**
  - the worst frame and SSIM;
  - both decodes against diffusers' fp32 host decode (the fp32 device decode should be near-identical to it);
  - decode seconds (first and repeat) and peak HBM.
- **If a model fails:** keep its VAE in fp32 (no `--vae-dtype` in its re-measure) and record the numbers.

**Step 3: decide the flags.**
- Wan2.1: `--vae-dtype bf16` if step 2 Wan passed; `--text-encoder-dtype bf16` if step 1 passed.
- Qwen-Image: `--vae-dtype bf16` if step 2 Qwen passed.
- If no switch passed for a model, its re-measure is skipped.

**Step 4: Qwen-Image native re-measure** (fresh root; device about 45 min; same harness and definitions as phase 2, decision D5).
- Cell `native_tp4_p3`: compile (cold compile), cold A, then warm gen0-3 (gen0 discarded; DiT step from gen1-3).
- Then q1 for the quality check, as the phase-2 hash check ran it.
```
cd $B && /usr/bin/python3 scripts/bench4m.py dry --model qwen_image --cell native_tp4_p3 --run gen0 \
  --mode generate --venv native -- --backend neuron --tp-degree 4 --vae-dtype bf16   # parse check
/usr/bin/python3 scripts/bench4m.py cell --model qwen_image --cell native_tp4_p3 --venv native \
  --runs compile,cold,gen0,gen1,gen2,gen3 --preread -- --backend neuron --tp-degree 4 --vae-dtype bf16
/usr/bin/python3 scripts/bench4m.py run --model qwen_image --cell native_tp4_p3 --run q1 --mode generate \
  --venv native --env HV1_DUMP_LATENTS_DIR=$B/qwen_image/native_tp4_p3/q1 \
  -- --backend neuron --tp-degree 4 --vae-dtype bf16
```
- Expected: the DiT step is unchanged (359.0 ms) and q1's latents equal the phase-2 hash check's (`4da4c89e`). Only the vae stage changes: the warm vae stage took 19.2 s in fp32.

**Step 5: Wan2.1 native re-measure** (fresh root; device about 50 min).
```
cd $B && /usr/bin/python3 scripts/bench4m.py cell --model wan2_1 --cell native_tp4_p3 --venv native \
  --runs compile,cold,gen0,gen1,gen2,gen3,q1 --preread \
  -- --backend neuron --tp-degree 4 --vae-dtype bf16 --text-encoder-dtype bf16
```
- **Cold A (D5):** the compile writes the root's weight store. That is the DiT's bf16 entry and, with bf16 umT5, an 11.36 GB umT5 entry. The cold run reads both.
- **Logs to check:**
  - `[weight store] ... NeuronWanTextEncoderApplication: hit 2.84 GB` on every rank;
  - the stage keys carry `text_encoder_dtype` and `vae_dtype`.
- **Projected (F13):** cold umT5 load about 86 s instead of 172.5 s; cold e2e about 456 s instead of 542.7 s.
- **Unchanged:** the DiT step, 1063.6 ms (the encoder runs only twice per request).

**Step 6: quality (CPU, minutes).**
- Both q1 runs against the existing CPU fp32 references, with the part-1/part-2 tooling: `quality_metrics.py`, plus `frames_map.py` for Wan's raw decode.
  - Wan: `--latents final $B/wan2_1/cpu_ref/ref_50/final.pt <q1>/final.pt --ref-frames $B/wan2_1/cpu_ref/ref_50/out.pt --cand-frames <mapped q1 out.pt>`.
  - Qwen-Image: against `$B/qwen_image/cpu_ref/ref_native_noise_50`.
- Also compare each run against its fp32 phase-2 q1 directly. For Wan with bf16 umT5 that is the "50-step e2e against today's fp32 device baseline" half of F13.
- **Decision recorded 2026-10-27 (conservative default; the plan never defined that half's pass rule):** bf16 umT5 passes the e2e if Wan's q1 is no more than 0.5 dB PSNR and 0.01 SSIM below the fp32 native q1, both against the CPU reference (22.66 dB / 0.885). Otherwise umT5 stays fp32, even when step 1 passed.

**Reporting and defaults.**
- Results go to the plan page and to `benchmark/trn2/four_model_native_vs_aot_2026-10-26.md` as a supplement, beside the fp32 numbers, which stay the headline.
- Making any of these bf16 paths the default is a separate change. It follows the default-change rule `74fc979` used (accuracy plus runs at TP2, TP4 and TP2xCP2) and needs the user's approval. Phase 7 (a compiled-by-region Wan VAE) builds on the bf16 decoder validated here.

## Added 2026-10-26 (user): last step, not urgent

- **Phase 7: compiled-by-region VAE for Wan2.1 and Qwen-Image native.** After phase 3, give the Wan VAE decoder (also used by Qwen-Image) the same treatment HunyuanVideo got in F1a (`1c10dfc`): compile the decoder region by region (fixed-size tiles, one set of small graphs) instead of eager per-op execution, in bf16 once phase 3 has validated bf16 for these VAEs. Why: HunyuanVideo's compiled-by-region decoder decodes in 16.0 s vs 74 s eager fp32 and compiles in about 135 s vs 1752 s eager, at the same speed as AoT's whole-graph compile (17.1 s) without AoT's hour-long VAE compile (Wan AoT VAE: 5468 s). Gates: decoded frames >= 50 dB vs the fp32 host decode, CPU tests of the tiling against diffusers, device decode timing; then re-measure Wan2.1 and Qwen-Image native warm/cold e2e as a supplement. **Done 2026-10-28** (`796e81b`, `338d73e`, `2cefb43`; opt-in `--vae-exec-mode compile`; `benchmark/trn2/four_model_native_vs_aot_2026-10-26.md`, "Last step: region-compiled Wan/Qwen VAE").
  - Frame by frame with explicit caches, whole frames (not tiled), 46 / 24 graphs.
  - Accuracy: bf16 55.91 / 55.87 dB vs the fp32 host decode; fp32 97.5 / 99.3 dB.
  - VAE compile: 179 → 140 s (Wan), 94 → 62 s (Qwen-Image).
  - Warm e2e: 129.1 s vs 127.9 s (Wan), 81.2 s vs 78.4 s (Qwen-Image); cold: 464.4 s vs 454.7 s, 517.9 s vs 519.2 s.
  - Why it loses at these shapes: each fresh vae process re-traces the regions with Dynamo.
  - Where it wins: from about 45 frames (81 frames: first decode 17.2 vs 21.3 s).
  - Kept opt-in.

## Added 2026-10-27 (user): deferred to a new machine

- **CPU bf16 vs CPU fp32 baseline (all four models).** Why: PSNR/SSIM against the CPU fp32 reference measure distance, not absolute quality, so the device numbers (16-21 dB on Qwen-Image and LTX-2) need a yardstick: the same pipeline in bf16 on the CPU from the same injected initial latents, compared with the fp32 reference, shows what bf16 precision alone costs. A device run at that level adds no error of its own. Scope: Qwen-Image x2 (native and AoT noise, about 3.5 h), LTX-2 x2 (about 3.6 h), Wan2.1 x1 (about 6 h), HunyuanVideo x2 (about 11 h); about 24 h of CPU in total, one at a time (60-82 GB RAM each). Inputs: the device q1 initial dumps and the fp32 references under `/home/ubuntu/bench-2026-10-26-4m/<model>/cpu_ref/`, the scripts in `/home/ubuntu/bench-2026-10-26-4m/scripts/` (`cpu_ref_*.py` with a bf16 dtype option, `quality_metrics.py`). Copy them to the new machine; the device is not needed. Not scheduled in this campaign.
