15a	| FLUX.1-dev | 221.4 (0.09%) | 194.4 (0.10%) | 0.878 | 0.827 (11b, host A) | 2.0% | both / bf16 < linear / bf16 - J: 0.807 -> not faster | +0.00% |
15b	| FLUX.1-dev | bf16 | blk1 (dual) x19 | 4.44 | 3.15 | 1.67 | 0.98 | 0.06 | 0.56 | 0.58 / 0.87 | tensor 70.9% |
15b	| FLUX.1-dev | bf16 | blk2 (single) x38 | 3.56 | 2.74 | 1.48 | 0.72 | 0.05 | 0.41 | 0.35 / 0.83 | tensor 76.8% |
15b	| FLUX.1-dev | both | blk1 (dual) x19 | 3.63 | 2.22 | 2.18 | 1.18 | 0.13 | 0.72 | 1.02 / 0.74 | tensor 61.2% |
15b	| FLUX.1-dev | both | blk2 (single) x38 | 3.10 | 1.84 | 2.02 | 0.93 | 0.12 | 0.74 | 0.84 / 0.70 | vector 65.4% |
15b-phase	| FLUX.1-dev | bf16 | T 18.9 | V 29.5, G 0.7 | V 15.6 | T 19.2, V 1.4 | - | - | - | T 0.2, V 0.3, G 0.7, H 6.6 | 86.8 | - |
15b-phase	| FLUX.1-dev | both | T 16.3 | V 30.6, G 3.8 | V 15.7 | T 8.8, V 1.3 | T 5.1, V 2.1, S 3.0, G 0.4, H 0.9 | V 0.6, G 1.1, H 0.2 | T 22.7, V 11.5, S 17.4 | T 0.4, V 0.4, S 0.1, G 0.8, H 5.5 | 129.8 | 1.4% |
15c	| FLUX.1-dev | 26.58 / 0.947 / 0.034 | 22.59 / 0.908 / 0.098 (11c, A) | 29.4 dB on this image (6a) | 24.41 / 0.916 / 0.077; bf16 here 23.36, evidence bf16 23.36 / linear 21.30 | - |
15d	| FLUX.1-dev | `67da9e98` (= evidence `67da9e98`, 11c, A); two requests equal | `c18e7bc5`; timed 2 and gen2 equal | no latents | no latents kept |
