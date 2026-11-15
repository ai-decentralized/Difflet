15a	| HunyuanVideo-1.5 | 9,918.5 (0.01%) | 12,092.0 (0.00%) | 1.219 | - | 2.0% | attn / bf16 < 1 - J: 0.980 -> not faster | +0.03% |
15b	| HunyuanVideo-1.5 | bf16 | blk1 (block) x54 | 182.56 | 117.43 | 142.90 | 15.50 | 4.83 | 41.30 | 6.80 / 87.75 | vector 78.3% |
15b	| HunyuanVideo-1.5 | attn | blk1 (block) x54 | 222.96 | 158.11 | 171.02 | 68.96 | 5.49 | 41.79 | 59.76 / 88.28 | vector 76.7% |
15b-phase	| HunyuanVideo-1.5 | bf16 | T 2807.7 | V 5039.0, S 220.0, G 73.5 | V 2123.1 | T 4523.7, V 138.9 | - | - | - | T 0.5, V 36.0, S 17.7, G 78.9, H 2015.1 | 15549.4 | - |
15b-phase	| HunyuanVideo-1.5 | attn | T 2875.4 | V 5028.8, S 219.9, G 99.3 | V 2166.2 | T 1757.3, V 164.8 | T 452.5, V 153.6, S 240.7, G 0.5, H 63.9 | V 6.9, G 15.5, H 2.3 | T 4250.3, V 1685.3, S 2699.0, G 37.2 | T 0.8, V 1.5, S 0.1, G 97.6, H 1998.5 | 21637.6 | 0.1% |
15c	| HunyuanVideo-1.5 | 12.99 (12.51) / 0.213 / 0.499 / cos 0.7876, rel L2 0.665 | 12.65 (12.20) / 0.183 / 0.504 (12c, B) | 16.94 (15.94) / 0.362 / 0.326, tp4 x CFG2 | - | below the bf16 floor's resolution, not a quality reading; bounded calls: K smoothing averages over all keys, padding included (BR36) |
15d	| HunyuanVideo-1.5 | `30903b0f` (!= evidence `d3ef0f1c`, 12c, B, 96 threads); two requests equal | `ec701ae5`; timed 2 and gen2 equal | bit-identical | finite |
