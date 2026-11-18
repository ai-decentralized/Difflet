"""The MiniMax-H3 AdaLN table cache: two replicas building one table must both get it."""

import argparse
import threading
import time
from types import SimpleNamespace

import torch

from difflet.backends.neuron.core import distributed
from difflet.cli.orchestrators.minimax_h3 import MiniMaxH3Orchestrator
from difflet.models.minimax_h3 import pipeline


def test_two_replicas_building_one_adaln_table_both_get_it(tmp_path, monkeypatch):
    table = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    sched = SimpleNamespace(set_timesteps=lambda n, device: None, timesteps=torch.zeros(4))
    monkeypatch.setattr(pipeline, "load_minimax_h3_schedulers", lambda model_dir: (sched, sched))
    monkeypatch.setattr(pipeline, "step_timestep_slots", lambda v, a, keyframes: torch.zeros(4, 2))
    monkeypatch.setattr(pipeline, "build_adaln_modulation_table", lambda *a, **k: table.clone())
    monkeypatch.setattr(distributed, "rank0_call", lambda fn, **kwargs: fn())
    real_save = torch.save

    def slow_save(obj, f):
        real_save(obj, f)
        time.sleep(0.3)  # the other replica starts its own build meanwhile

    monkeypatch.setattr(torch, "save", slow_save)
    orch = object.__new__(MiniMaxH3Orchestrator)
    orch._neuron_device = None
    args = argparse.Namespace(cache_dir=str(tmp_path))
    config = SimpleNamespace(num_keyframes=0, num_layers=1, freq_dim=4, time_embed_hidden_dim=4,
                             time_embed_dim=4, hidden_size=4)
    got, errors = [], []

    def build(delay):
        time.sleep(delay)
        try:
            got.append(orch._neuron_adaln_table(args, str(tmp_path / "model"), config, 4))
        except Exception as exc:  # surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=build, args=(d,)) for d in (0.0, 0.1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []
    assert len(got) == 2 and all(torch.equal(t, table) for t in got)
    assert sorted(p.name for p in (tmp_path / "minimax_h3_neuron").iterdir()) == [
        "adaln_table_steps4.pt"]  # no temp file left behind
