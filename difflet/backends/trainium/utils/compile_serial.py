"""Serialize NxD bucket compiler jobs within a component compile."""

from contextlib import contextmanager
from functools import wraps
from threading import Lock


@contextmanager
def serial_bucket_compilation():
    """Bound host memory by running one bucket compiler at a time.

    NxD 0.19's ModelBuilder does not expose a compilation worker limit.
    Like compile_retry, this adapter is scoped to Difflet's sequential
    component compile flow; concurrent compile scopes in one process are
    unsupported. Compiler jobs within the scope can use multiple threads.
    """
    from neuronx_distributed.trace import model_builder

    original = model_builder.neuron_xla_compile
    if getattr(original, "_difflet_serial_compile", False):
        yield
        return
    lock = Lock()

    @wraps(original)
    def serialized(*args, **kwargs):
        with lock:
            return original(*args, **kwargs)

    serialized._difflet_serial_compile = True
    model_builder.neuron_xla_compile = serialized
    try:
        yield
    finally:
        model_builder.neuron_xla_compile = original
