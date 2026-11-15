15a	| Qwen-Image | 304.0 (0.10%) | 257.9 (0.12%) | 0.848 | 0.791 (11b, host A) | 2.0% | both / bf16 < linear / bf16 - J: 0.771 -> not faster | +0.38% |
15b	| Qwen-Image | bf16 | blk1 (block) x60 | 5.05 | 3.55 | 2.05 | 1.15 | 0.20 | 0.62 | 0.60 / 1.07 | tensor 70.3% |
15b	| Qwen-Image | both | blk1 (block) x60 | 4.43 | 2.41 | 2.66 | 1.42 | 0.11 | 1.00 | 1.06 / 0.97 | vector 60.0% |
15b-phase	| Qwen-Image | bf16 | T 27.6 | V 41.2, G 0.8 | V 20.1 | T 26.4, V 1.6 | - | - | - | T 0.4, V 0.4, G 0.9, H 9.2 | 117.4 | - |
15b-phase	| Qwen-Image | both | T 21.7 | V 41.2, G 2.1 | V 20.2 | T 12.5, V 1.6 | T 6.9, V 2.5, S 3.3, G 0.4, H 1.0 | V 0.6, G 1.4, H 0.2 | T 29.3, V 13.4, S 21.6 | T 0.6, V 0.6, S 0.1, G 1.0, H 8.3 | 167.1 | 1.3% |
15c	| Qwen-Image | 29.14 / 0.964 / 0.060 / cos 0.9865, rel L2 0.164 | 38.26 / 0.992 / 0.008 (11c, A) | 39-46 dB | 29.08 / 0.965 / 0.061 / cos 0.9868, rel L2 0.162; bf16 here 36.34, evidence bf16 36.34 / linear 37.36 | bounded calls: K smoothing averages over all keys, padding included (BR36) |
15d	| Qwen-Image | `5478687f` (= evidence `5478687f`, 11c, A); two requests equal | `583e3fce`; timed 2 and gen2 equal | bit-identical | finite |
