# Filename: tests/test_planner.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify specialist selection and review-plan dependencies.

import asyncio

import pytest

from src.common.schemas import PaperDoc
from src.orchestrator.planner import PaperPlanner


def test_full_paper_plan():
    paper = PaperDoc(
        id="paper001",
        sections={"Methods": "We derive a channel model."},
        equations=["y = Hx + n"],
        figures=["figure1.png"],
        references=["Reference 1"],
    )

    plan = asyncio.run(PaperPlanner(use_mock=True).plan(paper))

    assert plan.paper_id == paper.id
    assert len(plan.tasks) == 7
    assert plan.tasks[0].agent == "summarizer"
    assert plan.tasks[-1].agent == "final_reviewer"

    agents = {task.agent for task in plan.tasks}
    assert {
        "math_checker",
        "figure_checker",
        "ref_verifier",
        "methodology",
        "novelty",
    } <= agents

    ids = [task.task_id for task in plan.tasks]
    assert len(ids) == len(set(ids))

    for task in plan.tasks[1:-1]:
        assert task.depends_on == ["summary"]

    assert set(plan.tasks[-1].depends_on) == set(ids[:-1])


def test_missing_evidence_skips_optional_specialists():
    paper = PaperDoc(
        id="text_only",
        sections={"Abstract": "A conceptual discussion."},
    )

    plan = asyncio.run(PaperPlanner().plan(paper))

    assert [task.agent for task in plan.tasks] == [
        "summarizer",
        "methodology",
        "novelty",
        "final_reviewer",
    ]

    assert all(task.task_type == "general" for task in plan.tasks)


def test_empty_paper_id_is_rejected():
    with pytest.raises(ValueError, match="Paper ID"):
        asyncio.run(PaperPlanner().plan(PaperDoc(id=" ")))


def test_llm_planning_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError):
        PaperPlanner(use_mock=False)