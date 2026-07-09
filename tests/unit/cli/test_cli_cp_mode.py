from difflet.cli.main import _build_parser


def test_cp_mode_flag_defaults_to_gather_kv():
    parser = _build_parser()
    args = parser.parse_args(["compile", "--model-id", "x"])
    assert args.cp_mode == "gather_kv"


def test_cp_mode_flag_accepts_ring():
    parser = _build_parser()
    args = parser.parse_args(["generate", "--model-id", "x", "--cp-degree", "2", "--cp-mode", "ring", "--prompt", "test", "--output", "/tmp/out.mp4"])
    assert args.cp_mode == "ring"


def test_cp_mode_flag_rejects_unknown():
    import pytest

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "--model-id", "x", "--cp-mode", "bogus"])
