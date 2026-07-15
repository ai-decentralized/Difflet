from difflet.cli.main import _build_parser


def test_cp_mode_flag_defaults_to_gather_kv():
    parser = _build_parser()
    args = parser.parse_args(["compile", "--model-id", "x"])
    assert args.cp_mode == "gather_kv"


def test_cp_mode_flag_accepts_ring():
    parser = _build_parser()
    args = parser.parse_args(["generate", "--model-id", "x", "--cp-degree", "2", "--cp-mode", "ring", "--prompt", "test", "--output", "/tmp/out.mp4"])
    assert args.cp_mode == "ring"


def test_cp_mode_flag_accepts_ulysses():
    parser = _build_parser()
    args = parser.parse_args([
        "generate", "--model-id", "x", "--cp-degree", "2", "--cp-mode", "ulysses",
        "--prompt", "test", "--output", "/tmp/out.mp4",
    ])
    assert args.cp_mode == "ulysses"


def test_cp_mode_flag_rejects_unknown():
    import pytest

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "--model-id", "x", "--cp-mode", "bogus"])


def test_cli_and_stage_parsers_offer_the_same_cp_modes():
    # The two parsers used to hand-copy the mode list; both now derive it from
    # CP_MODES, so a new mode can never reach one parser and not the other.
    from difflet.cli.stage import _build_stage_parser
    from difflet.pipeline.parallel_config import CP_MODES

    def cp_mode_choices(parser):
        for action in parser._actions:
            if action.dest == "cp_mode":
                return tuple(action.choices)
        raise AssertionError("no --cp-mode action")

    # main's cp_mode lives on the subparsers, not the top-level parser.
    main_sub = _build_parser()._subparsers._group_actions[0].choices["generate"]
    assert cp_mode_choices(main_sub) == CP_MODES
    assert cp_mode_choices(_build_stage_parser()) == CP_MODES
