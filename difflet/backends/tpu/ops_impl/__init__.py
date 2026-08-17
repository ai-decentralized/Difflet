"""TPU implementations of the ``difflet.ops`` surface.

Empty until Phase 2 of docs/plans/2026-08-16-tpu-backend-support.md.
``difflet.ops`` dispatch imports ``difflet.backends.tpu.ops_impl.<module>``
lazily, so a missing module raises only when the corresponding op is first
used — never at backend registration time.
"""
