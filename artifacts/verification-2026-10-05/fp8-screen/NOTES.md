# fp8 step-time screen, round 2 — 2026-10-05 (idle host, rebuilt env)

trn2.3xlarge (re-imaged; env rebuilt at `<repo>/.venv`, neuronx-cc 2.26.6360, nki 0.5.0), tp4,
Wan 2.1 T2V-14B truncated to 2 real blocks, 480x832x9 (4680 tokens, text 512), static scales
calibrated on the probe's own inputs, 10 timed forwards after 2 warmups. Driver:
`scripts/ptq_fp8_screen.sh <name> [--only arm] KEY=VALUE...`, queue: `job_screen.sh`,
results: `screen/<name>.json` + `.log`. Nothing else ran on the host (the 10-05 round-1
screen had a CPU job alongside and read ~2 ms higher on every arm).

## Results (2-block forward, ms, mean of 10)

| config | switches | bf16 | fp8 static | fp8 vs base fp8 | numerics |
|---|---|---:|---:|---:|---|
| base (shipped) | — | 35.69 | 32.34 | — | cos 0.99991 / 0.99970 vs CPU |
| best_a | VC2 + `--vectorize-strided-dma` | **30.23** | **27.32** | **−15.5 %** | same |
| best_a_bf16q | + `DIFFLET_FP8_BF16_QUANT=1` | — | 27.32 | −15.5 % | identical fp8 output (HLO differs: MODULE_0d2ce1… vs 036641…) |
| best_a_bf16q_dq | + `DIFFLET_FP8_BF16_DEQUANT=1` | — | 27.20 | −15.9 % | same |
| bf16_dequant | `DIFFLET_FP8_BF16_DEQUANT=1` alone | — | 32.31 | −0.1 % | same |
| pad128 | `DIFFLET_WAN_PAD_TOKENS=128` (4680 → 4736) | 40.72 | 37.46 | **+15.8 %** | cos 0.99991 / 0.99970 (bounds correct) |
| best_a_bf16q_pad128 | levers + pad 128 | 34.52 | 31.61 | +15.7 % vs best_a | same |
| pad16 | `DIFFLET_WAN_PAD_TOKENS=16` (4688) | — | compile error | — | attention_cte bounded path: `NCC_IBIR243 access pattern out of bounds` at S % 128 != 0 |
| flag_mm_reorder | `--enable-internal-postsched-mm-accum-reorder` | — | 32.44 | +0.3 % | cos 0.99970 |
| flag_mm_remat / flag_lnc_single / flag_fp8_nosat | libwalrus strings passed on the neuronx-cc command line | — | `NCC_EARG002` unrecognized argument | — | not reachable through the 2.26 CLI |

## Findings

1. **The only lever is VC2 + strided DMA: −15 % on both arms.** It is not fp8-specific
   (bf16 35.69 → 30.23, fp8 32.34 → 27.32). Wan's production path does not set
   `NEURON_RT_VIRTUAL_CORE_SIZE=2` today (FLUX / Qwen-Image / HunyuanVideo do), so this is a
   free win for Wan and LTX-2 independent of quantization.
2. **bf16-domain quantize is neutral on an idle host.** Round 1's −8 % (28.27 vs 30.66 ms)
   was measured with a CPU job running; here the two arms are 27.32 vs 27.32 ms with
   bit-identical fp8 outputs (the bf16 multiply does not change any fp8 code). The switch
   changes the HLO but not the schedule. Same for bf16 dequant (−0.1 % / −0.4 %): the compiler
   already folds the fp32 dequant multiply into the output cast.
3. **Sequence-level padding to 128 is +15 % slower, not faster.** The double-row hypothesis
   (4680 = 12 x 390 rows not 16-aligned → single-row fp8 dots) does not translate into a win
   through padding: bf16 is penalised just as much as fp8, so the cost is not in the dots. The
   candidate costs are attention_cte's bounded path (range select + q pre-scale, replacing the
   unmasked kernel) and the +1.2 % tokens; `pad128_nobounds` (timing-only, wrong numerics)
   separates them — see below. Padding to 16 is not possible: the bounded kernel needs S % 128.
4. **neuronx-cc 2.26 exposes no extra fp8 / matmul flag.** The walrus-level options found in
   `libwalrus.so` are rejected by the CLI; the one accepted (`postsched-mm-accum-reorder`) is neutral.
5. **The 2-block probe over-rewards fp8.** fp8 is 9–10 % faster than bf16 on the probe in every
   configuration, while the full 40-block step is 2.4 % slower (10-03). With 2 blocks the weight
   DMA (halved by fp8) is exposed; at full depth it overlaps compute and only the quantize chain
   remains. The probe ranks levers; only the full A/B (`../fp8-ab/`) decides.

## pad128 without attention bounds (timing only)

`DIFFLET_WAN_PAD_NOBOUNDS=1` (padded keys attended, wrong numerics): bf16 40.34 ms, fp8
37.06 ms vs 40.72 / 37.46 with the bounds. The bounded attention path costs ~0.4 ms; the other
~5 ms is the padded shape itself: the compiler tiles 4736 rows (37 x 128) worse than 4680
(12 x 390). So 128-alignment is the wrong target; `job_screen2.sh` tries 512 / 256 multiples
(5120 = 10 x 512, +9.4 % tokens), the tile width the trace saw on the one layer pair that got
the double-row 2x.
