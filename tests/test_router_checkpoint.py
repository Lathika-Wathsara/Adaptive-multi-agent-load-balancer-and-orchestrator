# Filename: tests/test_router_checkpoint.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify router checkpoint restoration, durable feedback,
#          duplicate prevention, and rollback after a save failure.

import yaml
import pytest

from src.common.schemas import PaperDoc
from src.orchestrator.feedback import FeedbackManager
from src.orchestrator.router import (
    DEFAULT_CONFIG_PATH,
    ModelRouter,
)


def training_config(tmp_path):
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Enable learning only in this temporary test configuration.
    config["routing"]["learning_enabled"] = True

    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_router_feedback_survives_restart(tmp_path):
    config_path = training_config(tmp_path)
    checkpoint = tmp_path / "learning.json"

    router = ModelRouter(config_path, checkpoint)
    decision = router.route(
        PaperDoc(id="paper001"),
        task_type="general",
    )

    # Registration itself must survive restarting.
    restored = ModelRouter(config_path, checkpoint)
    assert restored.feedback.pending_count == 1

    restored.apply_feedback(decision.decision_id, reward=0.8)

    restarted = ModelRouter(config_path, checkpoint)

    assert restarted.feedback.pending_count == 0
    assert restarted.policy.update_counts[decision.model] == 1

    with pytest.raises(ValueError, match="already applied"):
        restarted.apply_feedback(decision.decision_id, reward=0.8)


def test_failed_save_preserves_live_policy(tmp_path, monkeypatch):
    config_path = training_config(tmp_path)
    checkpoint = tmp_path / "learning.json"

    router = ModelRouter(config_path, checkpoint)
    decision = router.route(
        PaperDoc(id="paper001"),
        task_type="general",
    )

    original_checkpoint = checkpoint.read_bytes()

    def fail_save(self, path):
        raise OSError("Simulated storage failure")

    monkeypatch.setattr(
        FeedbackManager,
        "save_checkpoint",
        fail_save,
    )

    with pytest.raises(OSError, match="storage failure"):
        router.apply_feedback(decision.decision_id, reward=0.8)

    assert router.policy.update_counts[decision.model] == 0
    assert router.feedback.pending_count == 1
    assert checkpoint.read_bytes() == original_checkpoint