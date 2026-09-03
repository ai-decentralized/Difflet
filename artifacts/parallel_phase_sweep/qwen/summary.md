# Parallel phase sweep — qwen (Qwen/Qwen-Image, 20 steps, guidance 4.0)

| config | compile s (cold) | load cold s | load warm s | e2e cold s | e2e warm s (median) | step ms (median, n) | finite | err |
|---|---:|---:|---:|---:|---:|---:|---|---|
| tp4sp | 1464.4 | 443.938 | 31.19 | 490.452 | 59.6 ±0.9 | 367.7 (n=19) | True |  |
| tp4 | 1204.9 | 447.691 | 31.126 | 494.391 | 60.8 ±1.0 | 421.7 (n=19) | True |  |
