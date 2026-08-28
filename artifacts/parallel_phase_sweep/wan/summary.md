# Parallel phase sweep — wan (Wan-AI/Wan2.2-T2V-A14B-Diffusers, 20 steps, guidance 4.0)

| config | compile s (cold) | load cold s | load warm s | e2e cold s | e2e warm s (median) | step ms (median, n) | finite | err |
|---|---:|---:|---:|---:|---:|---:|---|---|
| dp2tp2 | cache | 656.321 | None | 409.155 | 51.7 ±2.6 | derived | None |  |
| dp2tp2sp | cache | 666.226 | None | 371.886 | 55.6 ±1.4 | derived | None |  |
| tp4 | 6262.8 | 328.141 | 23.001 | 391.718 | 63.8 ±0.7 | 581.1 (n=39) | True |  |
| tp4sp | 987.1 | 328.556 | 23.486 | 388.22 | 65.2 ±1.0 | 612.0 (n=39) | True |  |
| tp2cp2 | 1387.0 | 630.218 | 26.28 | 691.76 | 69.2 ±0.9 | 625.6 (n=39) | True |  |
| tp2cfg | 1442.8 | 634.284 | 29.153 | 693.162 | 70.0 ±0.5 | 1123.6 (n=19) | True |  |
| tp2cfgsp | 1430.2 | 632.424 | 29.356 | 698.114 | 72.1 ±0.9 | 1237.2 (n=19) | True |  |
| tp2 | 1203.8 | 327.212 | 25.4 | 406.289 | 87.2 ±0.2 | 1116.3 (n=39) | True |  |
| tp2 | 1203.8 | 327.212 | 25.4 | 406.289 | 87.2 ±0.2 | 1116.3 (n=39) | True |  |
| tp2 | 1203.8 | 327.212 | 25.4 | 406.289 | 87.2 ±0.2 | 1116.3 (n=39) | True |  |
| tp2sp | 1164.9 | 325.616 | 25.841 | 409.523 | 92.7 ±0.7 | 1233.7 (n=39) | True |  |
| tp2cp2ring | None | None | None | None | — | derived | None | compile |
| tp2cp2ulysses | 1341.6 | 638.712 | None | 701.471 | — | derived | None | generate |
