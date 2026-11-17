# Filename: tests/test_context.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify context order, normalization, task encoding,
#          and invalid-input handling.

import pytest

from src.common.schemas import PaperDoc
from src.orchestrator.context import (
    CONTEXT_DIMENSION,
    CONTEXT_FEATURE_NAMES,
    ContextBuilder,
)


def test_context_values_and_order():
    paper = PaperDoc(
        id="paper001",
        sections={"Methods": "x" * 4000},
        equations=["(1) y = Hx + n", "z = x + y"],
        figures=["figure1.png"],
        references=["Reference 1"],
    )

    context = ContextBuilder().build(
        paper,
        task_type="math",
        difficulty=0.8,
        queue_utilization=0.25,
    )

    assert CONTEXT_DIMENSION == 16
    assert len(context) == 16

    values = dict(zip(CONTEXT_FEATURE_NAMES, context))

    assert values["bias"] == 1.0
    assert values["n_tokens_est"] == pytest.approx(1000 / 20000)
    assert values["n_equations"] == pytest.approx(2 / 100)
    assert values["n_numbered_equations"] == pytest.approx(1 / 100)
    assert values["eq_per_1k_tokens"] == pytest.approx(2 / 20)

    assert values["task_general"] == 0.0
    assert values["task_math"] == 1.0
    assert values["task_vision"] == 0.0

    assert values["difficulty"] == 0.8
    assert values["queue_utilization"] == 0.25


def test_empty_and_large_papers_produce_bounded_contexts():
    builder = ContextBuilder()

    papers = [
        PaperDoc(id="empty"),
        PaperDoc(
            id="large",
            sections={"Methods": "x" * 100000},
            equations=["x = y"] * 200,
        ),
    ]

    for paper in papers:
        context = builder.build(paper, task_type="general")
        assert len(context) == 16
        assert all(0.0 <= value <= 1.0 for value in context)

    large_context = builder.build(papers[1], task_type="general")
    values = dict(zip(CONTEXT_FEATURE_NAMES, large_context))

    assert values["n_tokens_est"] == 1.0
    assert values["n_equations"] == 1.0


def test_invalid_task_and_runtime_values():
    paper = PaperDoc(id="paper001")
    builder = ContextBuilder()

    with pytest.raises(ValueError):
        builder.build(paper, task_type="unknown")

    with pytest.raises(ValueError):
        builder.build(paper, task_type="math", difficulty=1.5)

    with pytest.raises(ValueError):
        builder.build(
            paper,
            task_type="vision",
            queue_utilization=float("nan"),
        )