# File: tests/test_notation.py
# Created: 2026-11-18
# Created by: Lathika
# Purpose: Tests for src/tools/notation.py and its use in the math checker prompt.

import asyncio

from src.agents import math_checker as mc
from src.common.schemas import ModelResponse
from src.ingest.models import EquationInfo, ParsedPaper
from src.tools import notation


def test_symbols_of_latex():
    assert notation.symbols(r"\mathbf{R}_u = \mathbf{R} + \mathbf{u}\mathbf{u}^{H}") == {"R_u", "R", "u"}
    assert notation.symbols(r"A(\psi) = \sum_{n=0}^{N-1} e^{jn\psi}") == {"psi", "n", "N", "e", "jn"}


def test_common_words_and_commands_are_not_symbols():
    assert notation.symbols(r"d \ge \left(|a| - |b|\right)^{2}") == {"d", "b"}


PAPER = ParsedPaper(
    id="p1",
    sections={
        "Introduction": "We study a link. Here $B$ denotes the interference matrix of the cell. "
                        "The weather was nice.",
        "System model": "Let $x$ be the received vector. The noise is white.",
        "Analysis": "We now combine the terms. The result follows from (2).",
    },
    equation_meta=[
        EquationInfo(index=0, number="2", latex="B = C D", source="tex"),
        EquationInfo(index=1, number="3", latex="y = 2 z", source="tex"),
        EquationInfo(index=2, number="30", latex="x = a + B", section="Analysis",
                     before="We now combine the terms.", after="The result follows from (2).",
                     source="tex"),
    ])


def test_definitions_come_from_far_away_sections():
    sentences = notation.paper_sentences(PAPER)
    found = notation.definitions_for("x = a + B", sentences)
    assert any("interference matrix" in s for s in found)
    assert any("received vector" in s for s in found)
    assert not any("weather" in s or "white" in s for s in found)     # no definition cue


def test_definitions_skip_the_equations_own_text():
    sentences = ["Here B denotes the matrix."]
    assert notation.definitions_for("x = B", sentences, skip="Here B denotes the matrix.") == []


def test_referenced_equations_skip_itself_and_the_previous_one():
    eqs = PAPER.equation_meta
    assert notation.referenced_equations("see (2) and (3) and (30) and (99)", eqs, "30", "3") == \
        [("2", "B = C D")]


class Recorder:
    def __init__(self):
        self.prompts = []

    async def call_model(self, model, messages, images=None):
        self.prompts.append(messages[1]["content"])
        return ModelResponse(text='{"verdict": "ok", "confidence": 0.9}', model_name=model, latency=0.0)


def test_late_equation_prompt_contains_the_early_definition():
    client = Recorder()
    asyncio.run(mc.check_math(PAPER, client, numbers=["30"]))
    prompt = client.prompts[0]
    assert "interference matrix" in prompt           # defined in the Introduction
    assert "(2) B = C D" in prompt                   # referred to by the text after the equation
    assert "Equation (30): x = a + B" in prompt