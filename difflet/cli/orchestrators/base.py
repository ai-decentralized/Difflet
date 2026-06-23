from __future__ import annotations

import argparse
from abc import ABC, abstractmethod


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
