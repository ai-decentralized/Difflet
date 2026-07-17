# Wan 2.1 Trn2 resident Serving evidence

The service process in `serve.log` was already running when API validation
started and was intentionally not stopped by this run. See the
[Wan 2.1 validation report](../../../../../docs/design/t2v_serving/09_wan21_trn2_serving_validation_zh.md).

`resources.jsonl` is the original startup sampler. The separate
`wan21_api/resources.jsonl` file records API phases without changing this
sampler's phase labels.

