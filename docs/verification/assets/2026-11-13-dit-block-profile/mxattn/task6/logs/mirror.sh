#!/usr/bin/env bash
# Task 6: NVMe mirrors of the root compile caches (NEFF/HLO copied, stage manifests copied; the
# quantized dirs, bf16 copies and weight stores symlinked, read only by hits). Task 5b's
# qwen qk+pv smoke NEFFs merged into the fp8 copy (same source tree, keyed by HLO).
set -eu
T=/mnt/nvme/difflet-mxattn/task6
date
for s in hunyuan_video_generate ltx_2_generate wan2_1_t2v_14b_diffusers_transformer wan2_1_t2v_14b_diffusers_vae wan_transformer wan_vae; do
  cp -a /home/ubuntu/difflet-cache-video/$s $T/cache/video/
done
for s in qwen_image_dit qwen_image_vae; do cp -a /home/ubuntu/difflet-cache-fp8/$s $T/cache/fp8/; done
for s in neff neff_local hlo; do cp -a /home/ubuntu/difflet-cache-fp8/neuron/$s $T/cache/fp8/neuron/; done
cp -an /mnt/nvme/difflet-mxattn/task5b/cache_smoke_qk+pv/neuron/neff/. $T/cache/fp8/neuron/neff/
cp -an /mnt/nvme/difflet-mxattn/task5b/cache_smoke_qk+pv/neuron/hlo/. $T/cache/fp8/neuron/hlo/
for s in neff_local hlo neff; do cp -a /home/ubuntu/difflet-cache-video/neuron/$s $T/cache/video/neuron/; done
du -sh $T/cache/video/neuron/* $T/cache/fp8/neuron/*
date
echo MIRROR_DONE
