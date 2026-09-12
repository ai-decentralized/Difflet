"""Compile-time attention selection, inherited by compiler/stage subprocesses."""

import os
from contextlib import contextmanager

ATTENTION_IMPLS = ("megakernel", "sdpa")
_ENV = "DIFFLET_ATTENTION_IMPL"


def get_attention_impl() -> str:
    impl = os.environ.get(_ENV, "megakernel")
    if impl not in ATTENTION_IMPLS:
        raise ValueError(f"unknown attention implementation: {impl!r}")
    return impl


def attention_cache_inputs(impl: str) -> dict:
    # Preserve existing optimized artifacts; SDPA must have its own identity.
    return {} if impl == "megakernel" else {"attention_impl": impl}


@contextmanager
def attention_implementation(impl: str):
    if impl not in ATTENTION_IMPLS:
        raise ValueError(f"unknown attention implementation: {impl!r}")
    previous = os.environ.get(_ENV)
    os.environ[_ENV] = impl
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = previous
