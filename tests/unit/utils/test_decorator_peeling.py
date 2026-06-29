"""Unit tests for difflet.utils.decorator_peeling.peel_decorations."""

import functools

from difflet.utils.decorator_peeling import peel_decorations


def test_undecorated_function_is_returned_unchanged():
    def plain():
        return 1

    assert peel_decorations(plain) is plain


def test_single_wrap_is_peeled():
    def base():
        return 42

    @functools.wraps(base)
    def wrapper(*args, **kwargs):
        return base(*args, **kwargs)

    # functools.wraps sets __wrapped__.
    assert peel_decorations(wrapper) is base


def test_multiple_wraps_are_peeled_to_innermost():
    def base():
        return "inner"

    @functools.wraps(base)
    def mid(*args, **kwargs):
        return base(*args, **kwargs)

    @functools.wraps(mid)
    def outer(*args, **kwargs):
        return mid(*args, **kwargs)

    assert peel_decorations(outer) is base
