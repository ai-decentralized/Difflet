15a	| HunyuanVideo | 18,488.3 (0.03%) | 25,259.8 (0.00%) | 1.366 | 0.972 (7f, host A) | 2.0% | both / bf16 < linear / bf16 - J: 0.952 -> not faster | -0.03% |
15b	| HunyuanVideo | bf16 | blk1 (dual) x20 | 458.23 | 168.14 | 193.53 | 24.97 | 7.55 | 244.94 | 11.47 / 115.37 | hbm_dma 53.5% |
15b	| HunyuanVideo | bf16 | blk2 (single) x40 | 231.37 | 157.27 | 182.84 | 20.04 | 5.74 | 57.21 | 12.72 / 112.98 | vector 79.0% |
15b	| HunyuanVideo | both | blk1 (dual) x20 | 703.58 | 218.00 | 232.14 | 93.56 | 7.53 | 448.79 | 82.55 / 116.79 | hbm_dma 63.8% |
15b	| HunyuanVideo | both | blk2 (single) x40 | 278.21 | 204.47 | 222.66 | 85.72 | 5.63 | 58.57 | 81.70 / 114.71 | vector 80.0% |
15b-phase	| HunyuanVideo | bf16 | T 2655.9 | V 7272.5, S 321.0, G 106.2 | V 3067.5 | T 4954.4, V 194.8 | - | - | - | T 1125.8, V 130.7, S 52.9, G 117.4, H 3076.9 | 21571.4 | - |
15b-phase	| HunyuanVideo | both | T 3388.6 | V 7252.4, S 285.5, G 134.5 | V 3087.2 | T 2391.3, V 232.2 | T 569.8, V 210.8, S 344.6, H 68.9 | V 11.2, G 22.5, H 5.8 | T 6070.0, V 2347.0, S 3888.5 | T 0.9, V 2.5, S 0.2, G 135.1, H 3001.6 | 30091.3 | 0.1% |
15c	| HunyuanVideo | 20.50 (13.34) / 0.818 / 0.243 / cos 0.9588, rel L2 0.288 | 26.46 (20.07) / 0.907 / 0.128 (7f, A) | none at this shape (320x512x61: 36.64 dB) | - | bounded calls: K smoothing averages over all keys, padding included (BR36) |
15d	| HunyuanVideo | `6336311a` (!= evidence `258cd73e`, 7f, A); two requests equal | `05948f8e`; timed 2 and gen2 equal | bit-identical | finite |
