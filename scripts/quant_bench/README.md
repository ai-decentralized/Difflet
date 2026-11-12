# The before/after quantization bench

bf16 against MXFP8 per model: compile into a fresh cache, e2e cold (twice), e2e warm (two
requests in one process), the DiT step, the peak HBM per rank and the device quality. Section 11
of `docs/verification/2026-11-07-mxfp8-video-models.md` ran it at the phase G shapes. The
official-settings rerun (`docs/plans/2026-11-08-mxfp8-official-settings-rerun.md`, section 12)
runs it at the publishers' settings.

| file | what |
|---|---|
| `bench.sh` | the runs: `bench.sh --cores <list> --cpus <list> <key>...`, bf16 then MXFP8 per key on one lane |
| `report.py` | timings -> `report.json`, `report.md` |
| `quality.py` | md5s per arm, MXFP8 vs bf16, and with `--floor` the floor and the screen -> `quality.json` |
| `hbm_sampler.py` | `neuron-monitor -c nm.json` -> `hbm_samples.txt` (the lanes' cores only) |
| `stamp.py` | a timestamp on every log line |
| `requirements-native-fp8.txt` | the venv's own 51 packages |

Keys: section 11's `flux qwen_image ltx_2 wan21 wan22 hunyuan_video hunyuan_video_15` and the
official `wan21_official wan22_official hv15_official`. HunyuanVideo-1.5 has no MXFP8 default,
so both of its MXFP8 arms export the former default (G1 + G3) as `DIFFLET_QUANT_TARGETS`, for
their three commands only.

## Host

Any trn3.48xlarge with the Neuron SDK venv `/opt/aws_neuronx_venv_torch_neuronx_2_14_0`
(torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631, nki 0.7.0):

```bash
# The venv: a child of the SDK venv that sees its packages (a site.addsitedir .pth, set before
# the install so pip never pulls another torch) with its own 51 pins.
/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python -m venv ~/venvs/native-fp8
echo "import site; site.addsitedir('/opt/aws_neuronx_venv_torch_neuronx_2_14_0/lib/python3.12/site-packages')" \
  > ~/venvs/native-fp8/lib/python3.12/site-packages/_sdk_venv.pth
~/venvs/native-fp8/bin/pip install --no-deps -r scripts/quant_bench/requirements-native-fp8.txt

# The hub snapshots on the NVMe RAID ($HF_HUB_CACHE, default /mnt/nvme/difflet-bench/hub), pinned:
HF_HUB_CACHE=/mnt/nvme/difflet-bench/hub ~/venvs/native-fp8/bin/huggingface-cli download \
  Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68
# ... Wan-AI/Wan2.2-T2V-A14B-Diffusers --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7
# ... hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf
```

A model already in another hub cache can be linked rather than copied: a `models--<org>--<name>`
directory holding `blobs` and `snapshots` links to the existing ones and a `refs/main` file with the
pinned commit. Offline, a snapshot without `refs/main` does not resolve by repo id.

## Runs

Before every launch, check with `neuron-ls` that the lane's devices are idle. Device d is logical
cores 4d..4d+3 at LNC=2. Devices 0-7 are on NUMA node 1 (CPUs 48-95,144-191), devices 8-15 on
NUMA node 0 (CPUs 0-47,96-143).

```bash
cd <checkout>
export BENCH_ROOT=/mnt/nvme/difflet-bench/results PY=~/venvs/native-fp8/bin/python
mkdir -p $BENCH_ROOT

# Determinism gate (section 11's Wan 2.1 bf16 at 480x832x9: md5 450d04ef...). It runs both arms;
# the gate reads the bf16 one.
scripts/quant_bench/bench.sh --cores 0-3 --cpus 48-71,144-167 wan21

# HBM sampler on the lanes' cores, then the three lanes (D4), one model each.
nohup sh -c "stdbuf -oL neuron-monitor -c scripts/quant_bench/nm.json | \
  $PY scripts/quant_bench/hbm_sampler.py $BENCH_ROOT/hbm_samples.txt --cores 0-3,16-19,32-35" &
nohup scripts/quant_bench/bench.sh --cores 0-3 --cpus 48-71,144-167 wan21_official > $BENCH_ROOT/W1.log 2>&1 &
nohup scripts/quant_bench/bench.sh --cores 16-19 --cpus 72-95,168-191 wan22_official > $BENCH_ROOT/W2.log 2>&1 &
nohup scripts/quant_bench/bench.sh --cores 32-35 --cpus 0-47,96-143 hv15_official > $BENCH_ROOT/H.log 2>&1 &

# The tables.
$PY scripts/quant_bench/report.py --root $BENCH_ROOT
$PY scripts/quant_bench/quality.py --root $BENCH_ROOT \
  --floor wan21_official=<tp8 mp4> wan22_official=<tp8 mp4> hv15_official=<tp4 x CFG2 mp4>
```

The floors (D5) are one bf16 `generate` each at another TP degree, with the key's settings and a
fresh cache: Wan 2.1 and Wan 2.2 at `--tp-degree 8` (cores 0-7, 16-23), HunyuanVideo-1.5 at
`--tp-degree 4 --cfg-parallel` (cores 32-39).
