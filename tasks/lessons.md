# Lessons

- A staged CLI topology is not automatically a valid resident-serving topology. Before co-loading stages in one Neuron process, verify that their NxD `world_size` values match or compile a serving-specific artifact with a uniform world size; process-level core reservation alone does not make mixed TP safe.
- Do not turn optional, model-specific calibration metadata into a generic serving startup schema. Transport raw options to the selected adapter and let it parse only fields that model uses; ignore calibration paths when acceleration is disabled, but fail startup when explicitly enabled calibration is invalid.
- Optional serving acceleration config must be resolved only when its enable flag is explicit, then validated and frozen before worker spawn; replacement workers must not reopen the user's mutable source file.
- Runtime allocation contracts are incomplete until the spawned child applies exact core, virtual-core, logical-NC, and distributed environment values before model imports; tests must observe those values inside a real spawned worker.
