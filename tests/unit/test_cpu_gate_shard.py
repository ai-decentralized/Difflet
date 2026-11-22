"""CI sharding (tests/_shard.py, wired in tests/conftest.py and scripts/cpu-gate.sh).

Every tests/unit file must land in exactly one shard, the same one on every worker and
every job, or a shard silently drops or repeats tests.
"""

import importlib.util
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parents[1]


def _load_shard():
    spec = importlib.util.spec_from_file_location("_shard_under_test", _TESTS_DIR / "_shard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shard = _load_shard()


@pytest.mark.parametrize("spec, parsed", [(None, None), ("", None), (" ", None),
                                          ("1/1", (1, 1)), ("2/3", (2, 3)), (" 3/3 ", (3, 3))])
def test_parse_spec(spec, parsed):
    assert shard.parse_spec(spec) == parsed


@pytest.mark.parametrize("spec", ["0/3", "4/3", "1/0", "3", "a/b", "1/3/2", "-1/3"])
def test_parse_spec_refuses_what_is_not_k_of_n(spec):
    with pytest.raises(ValueError, match="k/n"):
        shard.parse_spec(spec)


def test_files_are_dealt_longest_first_to_the_least_loaded_shard():
    durations = {"a.py": 10.0, "b.py": 9.0, "c.py": 1.0, "d.py": 1.0}
    # a -> 0 (10), b -> 1 (9), c -> 1 (10), d: tie at 10 -> the lowest shard, 0.
    assert shard.assign(["d.py", "c.py", "b.py", "a.py"], durations, 2) == [
        ["a.py", "d.py"], ["b.py", "c.py"]]


def test_an_unlisted_file_weighs_the_default():
    durations = {"heavy.py": shard.DEFAULT_SECONDS * 2}
    assert shard.assign(["heavy.py", "new1.py", "new2.py"], durations, 2) == [
        ["heavy.py"], ["new1.py", "new2.py"]]


def test_the_shipped_shards_partition_tests_unit():
    files = shard.unit_test_files()
    shards = [shard.shard_files(f"{k}/3") for k in (1, 2, 3)]
    assert all(shards)
    assert sorted(f for s in shards for f in s) == files  # each file once, none lost
    assert shard.shard_files(None) is None and shard.shard_files("") is None


def test_durations_from_logs_sums_each_file(tmp_path):
    log = tmp_path / "cpu-gate-unit.log"
    log.write_text(
        "============================= slowest durations ==============================\n"
        "12.50s call     tests/unit/a/test_x.py::test_one[tp2 cp2]\n"
        "0.25s setup    tests/unit/a/test_x.py::test_one[tp2 cp2]\n"
        "1.00s teardown tests/unit/b/test_y.py::Test::test_two\n"
        "3.00s call     tests/unit/b/test_y.py\n"
        "(20 durations < 0.005s hidden.  Use -vv to show these durations.)\n"
        "FAILED tests/unit/a/test_x.py::test_one - AssertionError\n")
    assert shard.durations_from_logs([str(log)]) == {
        "tests/unit/a/test_x.py": 12.75, "tests/unit/b/test_y.py": 4.0}
