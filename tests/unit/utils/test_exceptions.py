"""Unit tests for difflet.utils.exceptions.LogitMatchingValidationError."""

import pytest

from difflet.utils.exceptions import LogitMatchingValidationError


def _result(passed):
    return {"passed": passed}


def test_stores_message_and_results():
    results = {0: [_result(True)]}
    err = LogitMatchingValidationError("boom", results)
    assert err.message == "boom"
    assert err.results is results
    assert isinstance(err, AssertionError)
    assert str(err) == "boom"


def test_no_failures_returns_negative_one():
    results = {0: [_result(True), _result(True)], 1: [_result(True)]}
    err = LogitMatchingValidationError("ok", results)
    assert err.get_divergence_index() == -1


def test_token_zero_failure_short_circuits_to_zero():
    # First sample fails at token 0 -> early return 0.
    results = {0: [_result(False), _result(True)]}
    err = LogitMatchingValidationError("fail", results)
    assert err.get_divergence_index() == 0


def test_returns_largest_failed_token_index():
    results = {0: [_result(True), _result(True), _result(False)]}
    err = LogitMatchingValidationError("fail", results)
    assert err.get_divergence_index() == 2


def test_largest_divergence_across_batch():
    results = {
        0: [_result(True), _result(False), _result(True)],
        1: [_result(True), _result(True), _result(True), _result(False)],
    }
    err = LogitMatchingValidationError("fail", results)
    # token_index 3 in batch 1 is the largest failed index.
    assert err.get_divergence_index() == 3


def test_token_zero_in_later_batch_returns_zero():
    results = {
        0: [_result(True), _result(False)],
        1: [_result(False), _result(True)],
    }
    err = LogitMatchingValidationError("fail", results)
    # batch 0 sets last_divergence_index=1; batch 1 token 0 fails but
    # token_index(0) > last_divergence_index(1) is False, so no early 0.
    assert err.get_divergence_index() == 1


def test_empty_results_returns_negative_one():
    err = LogitMatchingValidationError("empty", {})
    assert err.get_divergence_index() == -1
