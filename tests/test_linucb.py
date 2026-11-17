# Filename: tests/test_linucb.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify LinUCB scores, eligibility, reward updates,
#          and invalid-input handling.

import numpy as np
import pytest

from src.orchestrator.linucb import LinUCB


def test_initial_score_matches_closed_form():
    policy = LinUCB(
        arms=["small", "large"],
        context_dimension=2,
        alpha=0.5,
        regularization=2.0,
    )

    # ||x|| = 5; A = 2I.
    scores = policy.scores([3.0, 4.0])
    expected = 0.5 * np.sqrt(25.0 / 2.0)

    assert scores["small"] == pytest.approx(expected)
    assert scores["large"] == pytest.approx(expected)


def test_selection_respects_eligibility():
    policy = LinUCB(
        arms=["text", "vision"],
        context_dimension=2,
    )

    assert policy.select(
        [1.0, 0.0],
        eligible_arms=["vision"],
    ) == "vision"

    assert policy.update_counts == {"text": 0, "vision": 0}


def test_reward_changes_only_selected_arm():
    policy = LinUCB(
        arms=["small", "large"],
        context_dimension=2,
        alpha=0.0,
    )

    context = [1.0, 0.0]

    policy.update("large", context, reward=1.0)

    # A_large = diag(2, 1), b_large = [1, 0].
    assert policy.scores(context)["large"] == pytest.approx(0.5)
    assert policy.scores(context)["small"] == pytest.approx(0.0)

    assert policy.select(context) == "large"
    assert policy.update_counts == {"small": 0, "large": 1}

    np.testing.assert_array_equal(policy.A["small"], np.eye(2))
    np.testing.assert_array_equal(policy.b["small"], np.zeros(2))


def test_invalid_inputs_are_rejected_without_updates():
    policy = LinUCB(
        arms=["small"],
        context_dimension=2,
    )

    with pytest.raises(ValueError):
        policy.select([1.0])

    with pytest.raises(ValueError):
        policy.select([1.0, np.nan])

    with pytest.raises(ValueError):
        policy.select([1.0, 0.0], eligible_arms=[])

    with pytest.raises(ValueError):
        policy.select([1.0, 0.0], eligible_arms=["unknown"])

    with pytest.raises(ValueError):
        policy.update("small", [1.0, 0.0], reward=1.5)

    assert policy.update_counts["small"] == 0
    np.testing.assert_array_equal(policy.A["small"], np.eye(2))