from __future__ import annotations

import argparse
from abc import ABC, abstractmethod


def cp_mode_token(args: argparse.Namespace) -> str:
    """Staged-artifact cache-dir token for ``--cp-mode``.

    The staged compiled-artifact directories are keyed on tp/cp/cfg/sp and the shape,
    but NOT on cp_mode — so two compiles differing only in cp_mode would collide in
    ~/.cache/difflet and silently reuse each other's artifact, even though their
    compile-cache hashes differ. This token disambiguates them.

    Empty at the ``gather_kv`` default, so every pre-existing cache dir keeps its
    current name and stays valid.
    """
    mode = getattr(args, "cp_mode", "gather_kv") or "gather_kv"
    return "" if mode == "gather_kv" else mode


class ModelOrchestrator(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @abstractmethod
    def download(self) -> None: ...

    @abstractmethod
    def compile(self) -> None: ...

    @abstractmethod
    def generate(self) -> None: ...

    def run(self) -> None:
        self.download()
        self.compile()
        self.generate()

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _run_stage_internal"
        )
