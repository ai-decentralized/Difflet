# Flux Benchmark Images

Common configuration:

- Model: `black-forest-labs/FLUX.1-dev`
- Runtime: AWS Trn2, TP=4, CP=1, world size=4
- Shape: 1024x1024
- Steps: 28
- Guidance scale: 3.5
- Prompt: `A cinematic red panda astronaut standing on a lunar ridge, detailed fur, realistic lighting`

| File | Mode | Seed | End-to-end time |
| --- | --- | ---: | ---: |
| `flux1-dev_cli_tp4_cp1_1024x1024_steps28_guidance3.5_seed101.png` | CLI, fresh process | 101 | 48.81 s |
| `flux1-dev_cli_tp4_cp1_1024x1024_steps28_guidance3.5_seed102.png` | CLI, fresh process | 102 | 44.66 s |
| `flux1-dev_cli_tp4_cp1_1024x1024_steps28_guidance3.5_seed103.png` | CLI, fresh process | 103 | 45.67 s |
| `flux1-dev_serving_tp4_cp1_1024x1024_steps28_guidance3.5_seed201.png` | Resident serving | 201 | 10.87 s |
| `flux1-dev_serving_tp4_cp1_1024x1024_steps28_guidance3.5_seed202.png` | Resident serving | 202 | 10.39 s |
| `flux1-dev_serving_tp4_cp1_1024x1024_steps28_guidance3.5_seed203.png` | Resident serving | 203 | 11.60 s |

CLI time includes process startup, Neuron initialization, weight loading, inference, and local PNG write. Serving time uses an already-loaded worker and includes inference, PNG serialization, R2 upload, and URL signing.


> Status of Difflet Serving
  >
  > Serving is now running end-to-end. During validation, I found a few runtime constraints that differ from our previous assumptions.
  >
  > Qwen-Image
  >
  > - The CLI runs the three stages in separate processes: Text TP4/W4, DiT TP4/W4, and VAE TP1/W1.
  > - Serving requires all three stages to remain resident. Mixing W4 and W1 in the same process causes the Neuron Runtime to crash when loading the VAE.
  > - The current serving solution uses TP4/W4 for the VAE, so all three stages use the same world_size=4.
  > - This setup has passed model loading, smoke tests, and real API requests.
  > - We can test VAE TP1/W4 later, although it would still occupy W4 and would not be equivalent to a single-core VAE.
  >
  > Flux
  >
  > - Flux is loaded as one complete pipeline rather than separate external stages.
  > - Its internal topology is CLIP TP1/W4, T5 TP4/W4, DiT TP4/W4, and VAE TP1/W4.
  > - Components may use different TP values, but they must share the same world_size=4.
  > - Three serving requests took approximately 32.86 seconds, compared with 139.14 seconds for three independent CLI runs.
  >
  > Serving design
  >
  > - Currently, each worker loads one model with one H/W profile. Multiple H/W profiles have not been tested yet.
  > - Switching the H/W profile requires reloading the model and rerunning smoke validation. Precompiled artifacts reduce compilation time but do not avoid the HBM reload
  >   cost.
  >
  > - Startup validation still needs to be improved, including checks for component TP/world size, artifacts, and loading order.
  > - The worker/profile lifecycle and multi-profile serving design also need further refinement.
  >
  > Next steps
  >
  > - Add serving startup validation.
  > - Test Qwen-Image VAE TP1/W4.
  > - Evaluate multiple H/W profiles and improve the model reload/profile-switching workflow.