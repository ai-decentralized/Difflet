import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from difflet.backends.trainium.core.multi_component_application import (
    ComponentSpec,
    MultiComponentApplication,
)
from difflet.backends.trainium.core.world_check import NeuronWorldMismatchError


class DummyComponent:
    def __init__(
        self,
        *,
        name: str,
        tp_degree: int,
        world_size: int,
        events: list[str] | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.config = SimpleNamespace(
            neuron_config=SimpleNamespace(
                tp_degree=tp_degree,
                world_size=world_size,
                start_rank_id=None,
                local_ranks_size=None,
            )
        )
        self.compile_calls = []
        self.load_calls = []

    def compile(self, compiled_model_path: str, debug: bool = False) -> None:
        self.compile_calls.append(
            {
                "path": compiled_model_path,
                "debug": debug,
                "workdir": os.environ.get("BASE_COMPILE_WORK_DIR"),
            }
        )
        if self.events is not None:
            self.events.append(f"compile:{self.name}")
        path = Path(compiled_model_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.pt").write_text("compiled\n", encoding="utf-8")
        self.write_config(path)

    def write_config(self, path: Path) -> None:
        data = {
            "neuron_config": {
                "tp_degree": self.config.neuron_config.tp_degree,
                "world_size": self.config.neuron_config.world_size,
                "start_rank_id": None,
                "local_ranks_size": None,
            }
        }
        (path / "neuron_config.json").write_text(
            json.dumps(data) + "\n",
            encoding="utf-8",
        )

    def load(
        self,
        compiled_model_path: str,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        if self.events is not None:
            self.events.append(f"load:{self.name}")
        self.load_calls.append(
            {
                "path": compiled_model_path,
                "start_rank_id": start_rank_id,
                "local_ranks_size": local_ranks_size,
                "skip_warmup": skip_warmup,
            }
        )


class DummyMultiComponentApplication(MultiComponentApplication):
    def __init__(self, specs: list[ComponentSpec]) -> None:
        super().__init__()
        self._specs = specs

    def components(self) -> list[ComponentSpec]:
        return self._specs


def test_compile_uses_declaration_order_and_restores_workdir(tmp_path, monkeypatch):
    events: list[str] = []
    first = DummyComponent(name="first", tp_degree=1, world_size=1, events=events)
    second = DummyComponent(name="second", tp_degree=4, world_size=4, events=events)
    app = DummyMultiComponentApplication(
        [
            ComponentSpec("first", first),
            ComponentSpec("second", second),
        ]
    )
    monkeypatch.setenv("BASE_COMPILE_WORK_DIR", str(tmp_path / "work"))
    original_workdir = os.environ["BASE_COMPILE_WORK_DIR"]

    app.compile(str(tmp_path / "compiled"), debug=True)

    assert events == ["compile:first", "compile:second"]
    assert first.compile_calls[0]["path"] == str(tmp_path / "compiled" / "first")
    assert second.compile_calls[0]["path"] == str(tmp_path / "compiled" / "second")
    assert first.compile_calls[0]["workdir"] == str(tmp_path / "work" / "first")
    assert second.compile_calls[0]["workdir"] == str(tmp_path / "work" / "second")
    assert first.compile_calls[0]["debug"] is True
    assert second.compile_calls[0]["debug"] is True
    assert os.environ["BASE_COMPILE_WORK_DIR"] == original_workdir
    assert app.has_compiled_artifacts(str(tmp_path / "compiled"))


def test_load_sorts_by_priority_then_world_size_and_clamps_single_core(tmp_path):
    events: list[str] = []
    small = DummyComponent(name="small", tp_degree=1, world_size=1, events=events)
    big = DummyComponent(name="big", tp_degree=4, world_size=4, events=events)
    # Same process world as `big` (mixed TP is fine; mixed world is rejected —
    # see test_load_rejects_mixed_worlds_before_touching_any_component).
    explicit = DummyComponent(name="explicit", tp_degree=2, world_size=4, events=events)
    app = DummyMultiComponentApplication(
        [
            ComponentSpec("small", small),
            ComponentSpec("big", big),
            ComponentSpec("explicit", explicit, load_priority=-10),
        ]
    )

    app.load(
        str(tmp_path / "compiled"),
        start_rank_id=3,
        local_ranks_size=1,
        skip_warmup=True,
    )

    assert events == ["load:explicit", "load:big", "load:small"]
    assert explicit.load_calls[0]["path"].endswith("/explicit")
    assert big.load_calls[0]["path"].endswith("/big")
    assert small.load_calls[0]["path"].endswith("/small")
    assert explicit.load_calls[0]["start_rank_id"] == 3
    assert big.load_calls[0]["start_rank_id"] == 3
    assert small.load_calls[0]["start_rank_id"] == 0
    assert small.load_calls[0]["local_ranks_size"] == 1
    assert small.load_calls[0]["skip_warmup"] is True


def test_load_rejects_mixed_worlds_before_touching_any_component(tmp_path):
    # The HunyuanVideo tp2cp2 signature (DiT world 4 + VAE world 2 in one
    # process) SIGSEGVed the Neuron runtime at weight init on device
    # (2026-08-30). It must now fail before the first component loads.
    events: list[str] = []
    dit = DummyComponent(name="transformer", tp_degree=2, world_size=4, events=events)
    vae = DummyComponent(name="vae_decoder", tp_degree=1, world_size=2, events=events)
    app = DummyMultiComponentApplication(
        [ComponentSpec("transformer", dit), ComponentSpec("vae_decoder", vae)]
    )

    with pytest.raises(NeuronWorldMismatchError) as excinfo:
        app.load(str(tmp_path / "compiled"), skip_warmup=True)

    assert events == []  # zero device time spent
    assert dit.load_calls == [] and vae.load_calls == []
    assert "transformer=w4" in str(excinfo.value)
    assert "vae_decoder=w2" in str(excinfo.value)


def test_load_accepts_one_world_with_mixed_tp_and_a_standalone_component(tmp_path):
    # Flux/HunyuanVideo resident topology: tp4/w4 + tp1/w4 co-resident, plus a
    # world-1 standalone component clamped to rank 0.
    events: list[str] = []
    t5 = DummyComponent(name="t5", tp_degree=4, world_size=4, events=events)
    clip = DummyComponent(name="clip", tp_degree=1, world_size=4, events=events)
    vae = DummyComponent(name="vae", tp_degree=1, world_size=1, events=events)
    app = DummyMultiComponentApplication(
        [ComponentSpec("clip", clip), ComponentSpec("t5", t5), ComponentSpec("vae", vae)]
    )

    app.load(str(tmp_path / "compiled"), start_rank_id=0, local_ranks_size=4, skip_warmup=True)

    assert events == ["load:clip", "load:t5", "load:vae"]
    assert clip.load_calls[0]["local_ranks_size"] == 4
    assert t5.load_calls[0]["local_ranks_size"] == 4
    assert vae.load_calls[0]["local_ranks_size"] == 1
    assert vae.load_calls[0]["start_rank_id"] == 0


def test_select_filters_components_and_rejects_unknown(tmp_path):
    first = DummyComponent(name="first", tp_degree=1, world_size=1)
    second = DummyComponent(name="second", tp_degree=4, world_size=4)
    app = DummyMultiComponentApplication(
        [
            ComponentSpec("first", first),
            ComponentSpec("second", second),
        ]
    )

    app.compile(str(tmp_path / "compiled"), select=["second"])

    assert first.compile_calls == []
    assert len(second.compile_calls) == 1
    assert app.has_compiled_artifacts(str(tmp_path / "compiled"), select=["second"])
    assert not app.has_compiled_artifacts(str(tmp_path / "compiled"))
    with pytest.raises(ValueError, match="Unknown component"):
        app.load(str(tmp_path / "compiled"), select=["missing"])


def test_has_compiled_artifacts_checks_neuron_config(tmp_path):
    component = DummyComponent(name="component", tp_degree=4, world_size=4)
    app = DummyMultiComponentApplication([ComponentSpec("component", component)])

    app.compile(str(tmp_path / "compiled"))
    assert app.has_compiled_artifacts(str(tmp_path / "compiled"))

    config_path = tmp_path / "compiled" / "component" / "neuron_config.json"
    config_path.write_text(
        json.dumps(
            {
                "neuron_config": {
                    "tp_degree": 1,
                    "world_size": 1,
                    "start_rank_id": None,
                    "local_ranks_size": None,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert not app.has_compiled_artifacts(str(tmp_path / "compiled"))
