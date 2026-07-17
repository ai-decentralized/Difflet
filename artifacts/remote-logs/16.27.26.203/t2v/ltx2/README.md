# LTX-2 Trn2 Videos Serving evidence

Captured from `16.27.26.203` on 2026-07-17. See
[`08_ltx2_trn2_serving_validation.md`](../../../../../docs/design/t2v_serving/08_ltx2_trn2_serving_validation.md)
for the interpreted results and measurement caveats.

| File | Purpose |
| --- | --- |
| `serve.log` | Complete cold download/compile/load/smoke/API/shutdown log |
| `resources.jsonl` | 302 raw five-second process/host records; 276 valid Neuron payloads plus 26 preserved early compile monitor timeouts |
| `sync-1.mp4` | First successful synchronous API output |
| `sync-1.headers` | Synchronous API response headers |
| `async-create.json` | Initial asynchronous job response |
| `async-complete.json` | Completed asynchronous job metadata |
| `async-content.mp4` | Downloaded asynchronous content output |
| `async-content.headers` | Content response headers |
| `queue-a.json` | Job allowed to run during queued-delete testing |
| `queue-b.json` | Job deleted while queued |
| `phase` | Last sampler phase (`shutdown`) |
| `sampler.pid`, `sampler.log` | Sampler process evidence; log is empty because sampling succeeded |

The MP4 files contain generated validation media. They are intentionally small
two-step outputs and are retained only as reproducible API/media evidence.
