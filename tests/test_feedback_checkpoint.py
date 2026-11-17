# Filename: tests/test_feedback_checkpoint.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify delayed feedback and duplicate prevention
#          after restoring a combined learning checkpoint.

from types import SimpleNamespace

import pytest

from src.orchestrator.feedback import FeedbackManager
from src.orchestrator.linucb import LinUCB


def make_manager():
    return FeedbackManager(
        LinUCB(
            arms=["small", "large"],
            context_dimension=16,
        ),
        learning_enabled=True,
    )


def decision(decision_id):
    return SimpleNamespace(
        decision_id=decision_id,
        model="large",
        context=[1.0] + [0.0] * 15,
    )


def test_pending_feedback_survives_restart(tmp_path):
    path = tmp_path / "learning.json"

    manager = make_manager()
    manager.register(decision("pending001"))
    manager.save_checkpoint(path)

    restored = FeedbackManager.load_checkpoint(
        path,
        learning_enabled=True,
    )

    assert restored.pending_count == 1

    restored.apply("pending001", reward=0.8)

    assert restored.policy.update_counts["large"] == 1
    assert restored.pending_count == 0


def test_duplicate_remains_blocked_after_restart(tmp_path):
    path = tmp_path / "learning.json"

    manager = make_manager()
    manager.register(decision("completed001"))
    manager.apply("completed001", reward=0.8)
    manager.save_checkpoint(path)

    restored = FeedbackManager.load_checkpoint(
        path,
        learning_enabled=True,
    )

    with pytest.raises(ValueError, match="already applied"):
        restored.apply("completed001", reward=0.8)

    assert restored.policy.update_counts["large"] == 1


def test_restored_learning_is_disabled_by_default(tmp_path):
    path = tmp_path / "learning.json"

    manager = make_manager()
    manager.register(decision("pending001"))
    manager.save_checkpoint(path)

    restored = FeedbackManager.load_checkpoint(path)

    with pytest.raises(RuntimeError, match="disabled"):
        restored.apply("pending001", reward=0.8)