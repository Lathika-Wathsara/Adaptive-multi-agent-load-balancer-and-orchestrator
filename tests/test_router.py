# Filename: tests/test_router.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify model eligibility, availability filtering,
#          decision context, and routing without learning.

import pytest

from src.common.schemas import PaperDoc
from src.orchestrator.router import (
    ModelRouter,
    NoAvailableModelError,
)


def make_paper():
    return PaperDoc(
        id="paper001",
        sections={"Methods": "We derive a channel model."},
        equations=["y = Hx + n"],
        figures=["figure1.png"],
    )


def test_routes_only_to_eligible_models():
    router = ModelRouter()
    paper = make_paper()

    for task_type in ("general", "math", "vision"):
        decision = router.route(paper, task_type)

        assert decision.model in router.eligibility[task_type]
        assert decision.paper_id == paper.id
        assert len(decision.context) == 16
        assert decision.context_version == 1
        assert decision.decision_id


def test_availability_filters_candidates():
    router = ModelRouter()

    decision = router.route(
        make_paper(),
        task_type="vision",
        available_models=["qwen3:4b", "gemma3:4b"],
    )

    assert decision.model == "gemma3:4b"
    assert decision.eligible_models == ("gemma3:4b",)


def test_unavailable_and_unknown_tasks():
    router = ModelRouter()
    paper = make_paper()

    with pytest.raises(NoAvailableModelError):
        router.route(
            paper,
            task_type="math",
            available_models=[],
        )

    with pytest.raises(NoAvailableModelError):
        router.route(
            paper,
            task_type="vision",
            available_models=["qwen3:4b"],
        )

    with pytest.raises(ValueError):
        router.route(paper, task_type="unknown")


def test_routing_does_not_update_policy():
    router = ModelRouter()
    paper = make_paper()

    first = router.route(paper, task_type="general")
    second = router.route(paper, task_type="general")

    assert first.decision_id != second.decision_id
    assert first.model == second.model

    assert all(
        count == 0
        for count in router.policy.update_counts.values()
    )