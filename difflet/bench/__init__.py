"""``difflet bench``: one benchmark entry point for the PyTorch-native neuron backend.

Design: docs/plans/2026-11-16-difflet-bench-design.md. A recipe (``recipes.yaml``) is a named set of
the engine's own ``difflet generate`` flags; bench translates it to argv mechanically, lets the real
CLI parser validate and route it, runs the engine in child processes, and appends one result JSON
with provenance per run. Nothing here imports torch at module level.
"""
