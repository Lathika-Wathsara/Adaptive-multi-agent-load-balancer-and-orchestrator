# Filename: tests/test_real_agent_adapter.py
# Created: 2026-11-19
# Created by: Thulashan
# Purpose: Verify delegation to teammate agents, bounded execution,
#          metadata preservation, and unsupported-agent rejection.

import asyncio

import pytest

from src.common.schemas import AgentResult
from src.ingest.models import ParsedPaper
from src.orchestrator.real_agent_adapter import RealAgentAdapter
from src.agents import math_checker, vision_checker


def test_math_delegation_preserves_paper_and_limits(monkeypatch):
    paper = ParsedPaper(id="paper001")
    client = object()
    expected = AgentResult(raw_output="math result")
    received = {}

    async def fake_check_math(**kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(
        math_checker,
        "check_math",
        fake_check_math,
    )

    result = asyncio.run(
        RealAgentAdapter(max_equations=3).execute(
            paper=paper,
            agent="math_checker",
            model="deepseek-r1:7b",
            client=client,
            numbers=["1", "2"],
        )
    )

    assert result is expected
    assert received["paper"] is paper
    assert received["client"] is client
    assert received["numbers"] == ["1", "2"]
    assert received["max_equations"] == 3
    assert received["model"] == "deepseek-r1:7b"


def test_vision_delegation_preserves_image_root(monkeypatch, tmp_path):
    paper = ParsedPaper(id="paper001")
    expected = AgentResult(raw_output="vision result")
    received = {}

    async def fake_check_vision(**kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(
        vision_checker,
        "check_vision",
        fake_check_vision,
    )

    result = asyncio.run(
        RealAgentAdapter(max_items=2).execute(
            paper=paper,
            agent="figure_checker",
            model="qwen2.5vl:7b",
            client=object(),
            ids=["Figure 1"],
            image_root=tmp_path,
        )
    )

    assert result is expected
    assert received["paper"] is paper
    assert received["max_items"] == 2
    assert received["ids"] == ["Figure 1"]
    assert received["image_root"] == tmp_path


def test_unsupported_agent_is_rejected():
    with pytest.raises(NotImplementedError):
        asyncio.run(
            RealAgentAdapter().execute(
                paper=ParsedPaper(id="paper001"),
                agent="final_reviewer",
                model="qwen3:4b",
                client=object(),
            )
        )


def test_incompatible_model_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        asyncio.run(
            RealAgentAdapter().execute(
                paper=ParsedPaper(id="paper001"),
                agent="math_checker",
                model="qwen2.5vl:7b",
                client=object(),
            )
        )