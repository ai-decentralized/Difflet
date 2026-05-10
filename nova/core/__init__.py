"""Compatibility shims for Trainium core.

The Trainium implementation moved to ``nova.backends.trainium.core`` during
Phase B. Import new code from the backend path; this package remains only for
older call sites during the migration window.
"""
