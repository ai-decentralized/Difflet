# Lessons

- A staged CLI topology is not automatically a valid resident-serving topology. Before co-loading stages in one Neuron process, verify that their NxD `world_size` values match or compile a serving-specific artifact with a uniform world size; process-level core reservation alone does not make mixed TP safe.
