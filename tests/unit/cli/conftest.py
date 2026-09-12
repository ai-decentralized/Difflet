"""The CLI suite describes ``difflet compile/generate/run`` on Trainium.

``main()`` refuses those commands on any other backend before building an
orchestrator (619d98b), and a venv with torch_xla auto-detects ``tpu``, so
pin the backend here; the one test that exercises the refusal sets ``tpu``
itself, after this fixture.
"""

import pytest


@pytest.fixture(autouse=True)
def _cli_tests_assume_trainium(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")
