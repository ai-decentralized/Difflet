15a	| LTX-2 | 1,286.0 (0.23%) | 991.2 (0.03%) | 0.771 | 0.752 (this session) | 2.0% | both / bf16 < linear / bf16 - J: 0.732 -> not faster | -0.34% |
15b	| LTX-2 | bf16 | blk1 (block) x96 | 13.55 | 8.84 | 4.21 | 1.78 | 0.20 | 1.75 | 1.34 / 2.19 | tensor 65.3% |
15b	| LTX-2 | both | blk1 (block) x96 | 10.40 | 6.28 | 5.58 | 2.53 | 0.32 | 2.50 | 2.37 / 2.28 | tensor 60.4% |
15b-phase	| LTX-2 | bf16 | T 74.7 | V 111.7, G 2.1 | V 60.7 | T 84.5, V 4.0 | - | - | - | T 0.5, V 0.9, G 2.2, H 19.8 | 474.5 | - |
15b-phase	| LTX-2 | both | T 57.1 | V 115.3, G 5.9 | V 59.6 | T 35.6, V 4.0 | T 12.9, V 5.9, S 8.4, H 2.4 | V 1.4, G 3.2, H 0.5 | T 87.6, V 39.6, S 66.3 | T 1.3, V 1.0, S 0.2, G 2.4, H 13.8 | 613.3 | 0.8% |
15c	| LTX-2 | 21.76 (20.00) / 0.805 / 0.094 / cos 0.9886, rel L2 0.151 | 22.47 / 0.830 / 0.078 (11c, A) | 25.9 dB / 0.909 (3a) | 17.01 (15.51) / 0.494 / 0.193 / cos 0.9432, rel L2 0.344; bf16 here 16.55, evidence bf16 16.55 / linear 16.85 | LTX-2's routed call (video attn1) is unbounded: BR36 does not apply |
15d	| LTX-2 | `0702ffd2` (= evidence `0702ffd2`, 11c, A); two requests equal | `da882704`; timed 2 and gen2 equal | bit-identical | finite |
