# Filename: tests/test_policy_state.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify policy round trips, continued learning,
#          and rejection of incompatible or corrupt checkpoints.

import json

import numpy as np
import pytest

from src.orchestrator.linucb import LinUCB
from src.orchestrator.policy_state import load_policy, save_policy


def make_policy():
    return LinUCB(
        arms=["small", "large"],
        context_dimension=16,
        alpha=0.5,
    )


def test_round_trip_preserves_scores_and_learning(tmp_path):
    original = make_policy()
    context = [1.0] + [0.0] * 15

    original.update("large", context, reward=0.8)

    path = tmp_path / "policy.json"
    save_policy(original, path)
    restored = load_policy(path)

    assert restored.scores(context) == pytest.approx(
        original.scores(context)
    )
    assert restored.update_counts == original.update_counts
    assert restored.alpha == original.alpha

    # Learning can continue from the restored state.
    original.update("small", context, reward=0.4)
    restored.update("small", context, reward=0.4)

    for arm in original.arms:
        np.testing.assert_allclose(
            restored.A[arm], original.A[arm]
        )
        np.testing.assert_allclose(
            restored.b[arm], original.b[arm]
        )


def test_context_mismatch_is_rejected(tmp_path):
    path = tmp_path / "policy.json"
    save_policy(make_policy(), path)

    payload = json.loads(path.read_text())
    payload["context"]["version"] = 999
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="context definition"):
        load_policy(path)


def test_invalid_matrix_is_rejected(tmp_path):
    path = tmp_path / "policy.json"
    save_policy(make_policy(), path)

    payload = json.loads(path.read_text())
    payload["A"]["small"][0][0] = -1.0
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        load_policy(path)