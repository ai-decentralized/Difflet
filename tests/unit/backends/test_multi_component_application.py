import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from difflet.backends.trainium.core.multi_component_application import (
    ComponentSpec,
    MultiComponentApplication,
)


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
    explicit = DummyComponent(name="explicit", tp_degree=2, world_size=2, events=events)
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
