"""A DP worker gets every `difflet generate` option the user set, except what the router owns.

The worker argv used to be a hand-written list: an option added to the parser and not to the list
was silently lost in --dp runs (--shapes, --taef1, --vae-in-process, --vae-cores, --first-frame,
--last-frame were). This walks the real parser, so a new option is covered without editing it.
"""

import argparse

import pytest

from difflet.cli.argv import generate_parser
from difflet.cli.dp.router import worker_cli_args

#: What the router sets per worker itself, or must never hand one (a worker is dp=1, reads its
#: requests from the manifest, and neither re-checks capacity nor force-rebuilds).
NOT_FORWARDED = {"dp", "mode", "prompt", "output", "requests", "requests_dir", "worker_index",
                 "dp_schedule", "work_dir", "total_cores", "force"}


def _sample(action):
    """A valid value for ``action`` that differs from its default."""
    if isinstance(action, argparse._StoreTrueAction):
        return True
    if action.choices:
        return next(c for c in reversed(list(action.choices)) if c != action.default)
    if action.type is int:
        return 3
    if action.type is float:
        return 0.5
    return "-blurry"  # argparse reads "-word" as an option unless sent as --flag=-word


def _options():
    return [a for a in generate_parser()._actions
            if a.option_strings and not isinstance(a, argparse._HelpAction) and a.dest not in NOT_FORWARDED]


@pytest.mark.parametrize("action", _options(), ids=lambda a: a.dest)
def test_every_generate_option_reaches_the_worker(action):
    parser = generate_parser()
    args = parser.parse_args(["--model-id", "org/model"])
    setattr(args, action.dest, _sample(action))

    back = parser.parse_args(worker_cli_args(args))

    assert getattr(back, action.dest) == getattr(args, action.dest)


@pytest.mark.parametrize("dest", sorted(NOT_FORWARDED))
def test_what_the_router_owns_never_reaches_the_worker(dest):
    parser = generate_parser()
    action = next(a for a in parser._actions if a.dest == dest)
    args = parser.parse_args(["--model-id", "org/model"])
    setattr(args, dest, _sample(action))

    assert getattr(parser.parse_args(worker_cli_args(args)), dest) == parser.get_default(dest)
