"""models/ltx_2/tpu_application.py: Dynamo's guards are skipped only for an input
signature that already ran, never for a new one, and not after reset or with the
switch off."""

from __future__ import annotations

import contextlib

import torch

from difflet.models.ltx_2 import tpu_application as ta

def _setup(monkeypatch):
    stances = []

    @contextlib.contextmanager
    def fake_stance(*args, **kwargs):
        stances.append(kwargs.get("skip_guard_eval_unsafe", False))
        yield

    monkeypatch.setattr(torch.compiler, "set_stance", fake_stance)
    monkeypatch.delenv(ta.SKIP_GUARDS_ENV, raising=False)
    app = object.__new__(ta.TpuLTX2Application)

    class Transformer:
        def forward(self, *inputs):
            return inputs[0] + 1, inputs[1]

    return app, Transformer(), stances

def _inputs(rows=4):
    return [torch.zeros(1, rows, 8), torch.zeros(1, 2, 8)]

def test_skips_only_a_signature_that_ran(monkeypatch):
    app, tr, stances = _setup(monkeypatch)
    video, _ = app._dit_call(tr, _inputs())
    assert stances == [] and torch.equal(video, torch.ones(1, 4, 8))  # first: guards run
    app._dit_call(tr, _inputs())
    assert stances == [True]                                          # same signature: skipped
    app._dit_call(tr, _inputs(rows=6))
    assert stances == [True]                                          # new shape: guards run
    app._dit_call(tr, _inputs(rows=6))
    assert stances == [True, True]

def test_reset_and_switch_off(monkeypatch):
    app, tr, stances = _setup(monkeypatch)
    app._dit_call(tr, _inputs())
    app.reset_guard_skip()
    app._dit_call(tr, _inputs())
    assert stances == []                                              # after reset: guards run
    monkeypatch.setenv(ta.SKIP_GUARDS_ENV, "0")
    app._dit_call(tr, _inputs())
    assert stances == []                                              # switched off
