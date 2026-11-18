"""A child process's command line (a DP worker, a stage launch), generated from the child's parser.

A hand-written list of flags loses every option added to the parser later and not to the list
(--sp, --shapes, --taef1, ... were lost that way). Walking the parser instead forwards each option
the parent holds a value for, spelled the way the child declares it.
"""

from __future__ import annotations

import argparse
import functools


def forward_argv(args: argparse.Namespace, parser: argparse.ArgumentParser, *,
                 exclude=frozenset(), defaults=None, overrides=None) -> list[str]:
    """The options of ``parser`` whose value in ``args`` differs from the parser's default.

    ``defaults`` fill only options ``args`` holds None for (a model's default: an explicit 0
    stays 0); ``overrides`` are values the parent resolved and are always spelled out. An option
    of a kind this function does not know raises instead of being dropped."""
    overrides = overrides or {}
    values = dict(vars(args))
    for dest, value in (defaults or {}).items():
        if values.get(dest) is None:
            values[dest] = value
    values.update(overrides)
    argv: list[str] = []
    for action in parser._actions:
        dest = action.dest
        if not action.option_strings or isinstance(action, argparse._HelpAction):
            continue
        if dest in exclude or values.get(dest) is None:
            continue
        value, flag = values[dest], action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                raise ValueError(f"{flag} is a switch; got {value!r}")
            if value:
                argv.append(flag)
            continue
        if type(action) is not argparse._StoreAction or action.nargs is not None:
            raise ValueError(f"{flag}: {type(action).__name__} is not forwarded; extend forward_argv")
        if dest not in overrides and type(value) is type(action.default) and value == action.default:
            continue
        text = str(value)
        # "--flag=value" for a value argparse would otherwise read as an option ("-blurry").
        argv += [f"{flag}={text}"] if text.startswith("-") else [flag, text]
    return argv


@functools.lru_cache(maxsize=1)
def generate_parser() -> argparse.ArgumentParser:
    """The real ``difflet generate`` parser: what a DP worker parses."""
    from difflet.cli.main import _build_parser

    sub = next(a for a in _build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices["generate"]
