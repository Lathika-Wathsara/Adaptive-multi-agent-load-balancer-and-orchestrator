# Filename: tests/test_feedback.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify explicit feedback, original-context preservation,
#          duplicate prevention, and disabled-learning behavior.

from types import SimpleNamespace

import numpy as np
import pytest

from src.orchestrator.feedback import FeedbackManager
from src.orchestrator.linucb import LinUCB


def make_policy():
    return LinUCB(
        arms=["small", "large"],
        context_dimension=2,
        alpha=0.0,
    )


def make_decision():
    return SimpleNamespace(
        decision_id="decision001",
        model="large",
        context=[1.0, 0.0],
    )


def test_feedback_uses_recorded_context():
    policy = make_policy()
    manager = FeedbackManager(policy, learning_enabled=True)
    decision = make_decision()

    manager.register(decision)

    # Later mutation must not change the registered context.
    decision.context[0] = 99.0

    manager.apply("decision001", reward=0.8)

    np.testing.assert_allclose(
        policy.A["large"],
        [[2.0, 0.0], [0.0, 1.0]],
    )
    np.testing.assert_allclose(
        policy.b["large"],
        [0.8, 0.0],
    )

    assert policy.update_counts["large"] == 1
    assert policy.update_counts["small"] == 0
    assert manager.pending_count == 0


def test_duplicate_feedback_is_rejected():
    policy = make_policy()
    manager = FeedbackManager(policy, learning_enabled=True)

    manager.register(make_decision())
    manager.apply("decision001", reward=0.8)

    with pytest.raises(ValueError, match="already applied"):
        manager.apply("decision001", reward=0.8)

    assert policy.update_counts["large"] == 1


def test_disabled_learning_does_not_update():
    policy = make_policy()
    manager = FeedbackManager(policy, learning_enabled=False)
    manager.register(make_decision())

    with pytest.raises(RuntimeError, match="disabled"):
        manager.apply("decision001", reward=0.8)

    assert policy.update_counts["large"] == 0


def test_invalid_feedback_preserves_pending_decision():
    policy = make_policy()
    manager = FeedbackManager(policy, learning_enabled=True)
    manager.register(make_decision())

    with pytest.raises(ValueError):
        manager.apply("unknown", reward=0.8)

    with pytest.raises(ValueError):
        manager.apply("decision001", reward=2.0)

    assert manager.pending_count == 1
    assert policy.update_counts["large"] == 0

    # Corrected feedback can still be submitted.
    manager.apply("decision001", reward=0.8)
    assert policy.update_counts["large"] == 1