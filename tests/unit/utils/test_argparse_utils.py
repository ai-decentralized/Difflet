"""Unit tests for difflet.utils.argparse_utils.StringOrIntegers."""

import argparse

import pytest

from difflet.utils.argparse_utils import AUTO, StringOrIntegers


def _make_action():
    return StringOrIntegers(option_strings=["--x"], dest="x", nargs="+")


def test_init_defaults():
    action = _make_action()
    assert action.string_value == AUTO
    assert action.string_value_found is False


def test_parses_list_of_integers():
    action = _make_action()
    ns = argparse.Namespace()
    action(None, ns, ["1", "2", "3"])
    assert ns.x == [1, 2, 3]


def test_auto_single_use_sets_string():
    action = _make_action()
    ns = argparse.Namespace()
    action(None, ns, ["AUTO"])  # case-insensitive
    assert ns.x == AUTO
    assert action.string_value_found is True


def test_auto_with_other_arguments_raises():
    action = _make_action()
    ns = argparse.Namespace()
    with pytest.raises(argparse.ArgumentTypeError):
        action(None, ns, ["auto", "1"])


def test_auto_used_twice_in_same_call_raises():
    action = _make_action()
    ns = argparse.Namespace()
    with pytest.raises(argparse.ArgumentTypeError):
        action(None, ns, ["auto", "auto"])


def test_auto_after_already_found_raises():
    action = _make_action()
    action.string_value_found = True  # simulate previously-seen AUTO
    ns = argparse.Namespace()
    with pytest.raises(argparse.ArgumentTypeError):
        action(None, ns, ["auto"])


def test_non_integer_value_raises():
    action = _make_action()
    ns = argparse.Namespace()
    with pytest.raises(argparse.ArgumentTypeError):
        action(None, ns, ["notanumber"])


def test_integration_with_parser_for_integers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", nargs="+", action=StringOrIntegers)
    args = parser.parse_args(["--x", "4", "5"])
    assert args.x == [4, 5]
