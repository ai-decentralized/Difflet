"""A Trainium AoT compile killed midway must not leave something that counts as compiled.

model.pt + neuron_config.json is what has_compiled_artifacts accepts, so NeuronApplicationBase.compile
(and the HV1.5 VAE's own compile) must drop the old config first, write model.pt atomically and write
the config last. The scenarios run in a subprocess that stubs the Neuron SDK, so the stub never
reaches this pytest process.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # the difflet the subprocess imports

_SCRIPT = textwrap.dedent(r'''
    import importlib.abc, importlib.machinery, json, os, sys, types
    from types import SimpleNamespace
    from unittest import mock

    class _Mod(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            value = mock.MagicMock(name=f"{self.__name__}.{name}")
            setattr(self, name, value)
            return value

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname.split(".")[0] in ("neuronx_distributed", "torch_neuronx", "torch_xla",
                                          "libneuronxla", "neuronxcc", "nki", "nkilib"):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None
        def create_module(self, spec):
            m = _Mod(spec.name); m.__path__ = []; return m
        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _Finder())

    import torch
    from difflet.backends.trainium.core import application_base as ab
    from difflet.backends.trainium.core.config import CONFIG_FILE, InferenceConfig
    from difflet.backends.trainium.hunyuan_video import vae15

    root = sys.argv[1]
    traced = torch.jit.trace(torch.nn.Linear(2, 2), torch.zeros(1, 2))
    real_jit_save = torch.jit.save
    events = []

    class Cfg:
        def save(self, path):
            events.append("config")
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, CONFIG_FILE), "w") as f:
                f.write("{}")

    def base_app():
        app = object.__new__(ab.NeuronApplicationBase)
        app.neuron_config = SimpleNamespace()
        app.config = Cfg()
        app.get_builder = lambda debug: SimpleNamespace(trace=lambda **kw: traced)
        app._save_configs_to_compiler_workdir = lambda: None
        return app

    def listing(d):
        return sorted(os.listdir(d)) if os.path.isdir(d) else []

    def half_written_save(obj, f):
        with open(f, "wb") as h:
            h.write(b"PK-half")
        raise RuntimeError("killed in torch.jit.save")

    out = {}
    ab.set_compile_env_vars = lambda *a, **k: None

    # 1. Killed while sharding, over a directory that a previous compile left a config in.
    d = os.path.join(root, "shard_killed") + "/"
    os.makedirs(d); open(d + CONFIG_FILE, "w").write("{\"stale\": true}")
    app = base_app()
    def shard_killed(*a, **k):
        raise RuntimeError("killed during shard_weights")
    app.shard_weights = shard_killed
    try:
        app.compile(d)
    except RuntimeError:
        pass
    out["shard_killed"] = listing(d)

    # 2. Killed in the middle of writing model.pt.
    d = os.path.join(root, "jit_killed") + "/"
    app = base_app(); app.shard_weights = lambda *a, **k: events.append("shard")
    torch.jit.save = half_written_save
    try:
        app.compile(d)
    except RuntimeError:
        pass
    torch.jit.save = real_jit_save
    out["jit_killed"] = listing(d)

    # 3. A compile that finishes: the config is written after the shards.
    d = os.path.join(root, "ok") + "/"
    events.clear()
    app = base_app(); app.shard_weights = lambda *a, **k: events.append("shard")
    app.compile(d)
    out["ok"] = listing(d)
    out["ok_events"] = list(events)
    out["ok_loads"] = isinstance(torch.jit.load(d + "model.pt"), torch.jit.ScriptModule)

    # 4. The HV1.5 VAE decoder's own compile: killed in torch.jit.save, then a finished one.
    def vae_app():
        app = object.__new__(vae15.NeuronHunyuanVideo15VAEDecoderApplication)
        app.config = Cfg()
        app.dtype = torch.float32
        app._model_cls = lambda cfg: torch.nn.Linear(2, 2)
        app.checkpoint_loader_fn = lambda: {}
        app.models = [SimpleNamespace(input_generator=lambda: [[torch.zeros(1, 2)]])]
        app.get_compiler_args = lambda: ""
        return app
    sys.modules["torch_neuronx"].trace = lambda model, example, compiler_args=None: traced
    d = os.path.join(root, "vae_killed")
    os.makedirs(d); open(os.path.join(d, CONFIG_FILE), "w").write("{}")
    torch.jit.save = half_written_save
    try:
        vae_app().compile(d)
    except RuntimeError:
        pass
    torch.jit.save = real_jit_save
    out["vae_killed"] = listing(d)
    d = os.path.join(root, "vae_ok")
    vae_app().compile(d)
    out["vae_ok"] = listing(d)

    # 6. A dry run over a finished artifact leaves it as it is (still compiled).
    app = base_app(); app.shard_weights = lambda *a, **k: None
    app.compile(os.path.join(root, "ok") + "/", dry_run=True)
    vae_app().compile(os.path.join(root, "vae_ok"), dry_run=True)
    out["dry_run"] = [listing(os.path.join(root, "ok")), listing(os.path.join(root, "vae_ok"))]

    # 5. The config file itself is written atomically.
    p = os.path.join(root, "cfg", CONFIG_FILE)
    os.makedirs(os.path.dirname(p))
    InferenceConfig.to_json_file(SimpleNamespace(to_json_string=lambda: '{"tp_degree": 4}'), p)
    out["cfg"] = [listing(os.path.dirname(p)), open(p).read()]
    print("RESULT " + json.dumps(out))
''')


def _run(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path)],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO), "DIFFLET_BACKEND": "cpu"},
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, proc.stdout + proc.stderr
    return json.loads(line[-1][len("RESULT "):])


def test_compile_killed_midway_leaves_nothing_that_counts_as_compiled(tmp_path):
    out = _run(tmp_path)
    # Killed while sharding: the stale config is gone, so model.pt alone is not "compiled".
    assert out["shard_killed"] == ["model.pt"]
    # Killed inside torch.jit.save: no model.pt (not even a truncated one) and no temp file.
    assert out["jit_killed"] == []
    # Finished: both files, the config written after the shards, model.pt loads.
    assert out["ok"] == ["model.pt", "neuron_config.json"]
    assert out["ok_events"] == ["shard", "config"]
    assert out["ok_loads"] is True
    # The HV1.5 VAE decoder's compile follows the same order.
    assert out["vae_killed"] == []
    assert out["vae_ok"] == ["model.pt", "neuron_config.json"]
    # A dry run (trace only) does not unmark a finished artifact.
    assert out["dry_run"] == [["model.pt", "neuron_config.json"]] * 2
    # neuron_config.json is replaced in one step: no temp file left beside it.
    assert out["cfg"] == [["neuron_config.json"], '{"tp_degree": 4}\n']
