15a	| Wan 2.2 T2V A14B | 10,120.5 (0.03%) | 11,270.0 (0.01%) | 1.114 | 0.981 (12b, host B) | 2.0% | both / bf16 < linear / bf16 - J: 0.961 -> not faster | -0.30% |
15b	| Wan 2.2 T2V A14B | bf16 | blk1 (block) x80 | 126.25 | 91.37 | 80.54 | 17.03 | 5.99 | 25.17 | 9.56 / 46.61 | tensor 72.4% |
15b	| Wan 2.2 T2V A14B | both | blk1 (block) x80 | 140.18 | 91.02 | 102.34 | 41.54 | 2.03 | 30.91 | 35.78 / 47.98 | vector 73.0% |
15b-phase	| Wan 2.2 T2V A14B | bf16 | T 2726.6 | V 3360.7, S 130.2, G 394.1 | V 1614.7 | T 3649.6, V 90.6 | - | - | - | T 2.1, V 34.0, S 185.1, G 393.9, H 1246.0 | 12218.9 | - |
15b-phase	| Wan 2.2 T2V A14B | both | T 1769.5 | V 3413.7, S 129.8, G 63.9 | V 1642.4 | T 915.5, V 108.1 | T 346.2, V 103.4, S 168.1, G 2.6, H 44.5 | V 8.6, G 18.3, H 2.7 | T 2611.1, V 1114.6, S 1948.0, G 14.3 | T 3.3, V 3.8, S 0.2, G 69.8, H 1270.9 | 14699.9 | 0.2% |
15c	| Wan 2.2 T2V A14B | 11.83 (10.88) / 0.215 / 0.439 / cos 0.8775, rel L2 0.496 | 14.29 (13.27) / 0.398 / 0.275 (12c, B) | 14.83 (13.74) / 0.439 / 0.254, tp8 | - | below the bf16 floor's resolution, not a quality reading |
15d	| Wan 2.2 T2V A14B | `5b443ec6` (!= evidence `ee61e12e`, 12c, B); two requests equal | `3192daae`; timed 2 and gen2 equal | bit-identical | finite |
