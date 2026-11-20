# Attention experiments on trn3: const-max (bf16) and static block-sparse

Branch `exp/attn-constmax-sparse`, from `feat/mxfp8-video-models` (`d6f8c0dc`). These are
experiments, not features: every hook is opt-in behind an `DIFFLET_EXP_*` env var, and with
the variable unset nothing changes.

**Why.** Several results from `feat/mxfp8-sage-attention` motivate these experiments. The
evidence is in `docs/verification/2026-11-13-dit-block-profile.md` on that branch, sections 9-12
and 15.

- MXFP8 attention loses on trn3. Shipped `attention_cte` is VectorE-bound: the MM1 PSUM evict
  with scale and row max, plus exp. TensorE is not the bottleneck.
- Two levers do not depend on matmul speed:
  1. **Const-max.** nkilib's `attention_const_max` (bf16) removes the VectorE evict and the DMA
     P transposes. At kernel level it ran 0.617 / 0.651 / 0.701x the rebalanced cte on
     flux_1024 / ltx2_video / wan_81f, with TensorE at 99.4% of span.
  2. **Block-sparse attention.** Skipping K/V tiles removes MM1, the evict, exp and MM2 work
     together, so the gain does not depend on which engine binds.

## E1: const-max, model level, FLUX.1-dev 1024x1024 first

- **Interface.**
  - Opt-in env `DIFFLET_EXP_ATTN_CONSTMAX=1`.
  - Unbounded, non-causal, bf16, D = 128 self/joint calls of the neuron backend go to an
    `nki_op` wrapper of nkilib `experimental/attention/attention_const_max.py` (2.14).
  - Everything else stays on today's path.
- **softmax_max per layer.**
  - Use the kernel docstring's rule: `0.88 * sigma * sqrt(2 ln Sk)`, with sigma from the
    QK-RMSNorm weights.
  - FLUX passes the `hard_bound - softmax_max <= ~80` check on all 57 layers.
  - The table is computed once on CPU and committed.
- **Gate checks.** Do these before any timing:
  - The kernel-level probe output is finite at the FLUX shape, and rel L2 against fp32 is
    recorded.
  - A 10-step smoke run produces finite latents with no noise.
- **Runs.** Three arms: bf16 (shipped cte), bf16 + const-max, and `--quant mxfp8` +
  const-max. All use seed 42, 50 steps, the `bench.sh` flags, and the same device and CPU set.
- **Metrics.**
  - Warm DiT step: median ms of two requests.
  - Quality against the bf16 arm and against the CPU fp32 reference
    (`/home/ubuntu/fp8-bench/fp32_ref/flux`): PSNR, SSIM, LPIPS and latent cosine.
  - Also compare with the bf16 floor (FLUX tp2 vs tp4).
- **Then, if FLUX holds quality.** Run LTX-2 video. Its 3 layers that fail the bound fall back
  to cte, through a per-layer allowlist.

## E2: static block-sparse, kernel level, wan_81f first

- **Kernel.**
  - Bring `difflet/backends/neuron/ops_impl/mx_attention_kernel.py` over from
    `feat/mxfp8-sage-attention`. It is attention_cte forked and pruned, and with its MX flags
    off it is bit-identical to shipped attention_cte.
  - Add a static K/V-tile skip mask: per Q group, the list of 512-key tiles to compute. It is a
    compile-time constant, with no indirection.
- **Correctness.** With the mask at full density the output must equal the unmasked kernel bit
  for bit. With a mask, the output must equal the masked fp32 reference.
- **Speed.** Measure at about 25%, 50% and 75% tile density on `wan_81f` (10 x 32,760 x 128;
  21 latent frames of 1,560 tokens). Use two mask families:
  - temporal windows (each query frame attends key frames within +-w);
  - STA-like 3D tile windows.
  
  Use the probe method: 3 processes, median and spread, the same device and CPU set.
- **Then, if time scales close to density.** Run model-level Wan 2.1 480x832x81 with the
  temporal-window mask at one density. Report step time and quality against bf16 and against
  its floor (22.75 dB).

## Rules

- Shared host:
  - Before every device run, check both the device (`fuser /dev/neuron<d>`) and the CPUs
    (`mpstat -P ALL`). Use only idle devices.
  - Pin `NEURON_RT_VISIBLE_CORES`, `taskset` and `NEURON_NUMA_PIN=0`.
  - Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for CPU-only jobs.
- Storage: caches and pytest `--basetemp` go on `/mnt/nvme`. Never delete a cache without the
  user.
- Git: push every commit to `origin/exp/attn-constmax-sparse`. Never force-push, never push to
  main.
- Results go to `docs/verification/2026-11-18-attn-constmax-sparse.md`.
