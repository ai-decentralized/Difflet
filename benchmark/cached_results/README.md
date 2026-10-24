# Cached benchmark results

One JSON per model (`<slug>.json`), written by `benchmark/bench.py` and
`benchmark/neuron_cold_warm.py` after every completed run and read back before the next one,
so a new instance reuses a measurement instead of spending hours repeating it. Commit this
folder after a run.

Each file holds one entry per config × platform (instance type, accelerator, SDK). An entry
is reused only on the same config, instance type and SDK, and only for 30 days after it was
measured. See `benchmark/result_cache.py` and the "Reusing stored results" section of
`benchmark/README.md`.

```bash
python -m benchmark.result_cache    # every entry: fresh / expired
```
