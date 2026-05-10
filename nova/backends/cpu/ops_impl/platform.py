"""CPU platform helpers for import-time numerical tests."""

from enum import Enum


class hardware(Enum):
    CPU = "cpu"
    TRN1 = "trn1"
    TRN2 = "trn2"


def get_platform_target():
    return hardware.CPU


__all__ = ["get_platform_target", "hardware"]

