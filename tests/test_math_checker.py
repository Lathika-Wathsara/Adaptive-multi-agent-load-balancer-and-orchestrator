# File: tests/test_math_checker.py
# Created: 2026-11-18
# Created by: Lathika
# Purpose: Tests for src/agents/math_checker.py with a scripted fake model (no Ollama).
#          The numeric tests the fake model "writes" are real and run in the real
#          restricted subprocess, so the whole path from reply to finding is covered.

import asyncio
import json

from src.agents import math_checker as mc
from src.common.schemas import AgentResult, ModelResponse, PaperDoc
from src.ingest.models import EquationInfo, ParsedPaper

# Real numeric tests, as a model might write them.
TEST_SQUARE = """
import numpy as np
rng = np.random.default_rng(1)
a = rng.normal(size=4) + 1j * rng.normal(size=4)
b = rng.normal(size=4) + 1j * rng.normal(size=4)
left = abs(a - b) ** 2
right = abs(a) ** 2 + abs(b) ** 2 {sign} 2 * (a * b.conj()).real
print("PASS" if np.allclose(left, right) else f"FAIL {{left[0]}} vs {{right[0]}}")
"""
TEST_GOOD = TEST_SQUARE.format(sign="-")        # the paper's claim as written is true
TEST_BAD = TEST_SQUARE.format(sign="+")         # the paper's claim as written is false


def reply(verdict="ok", confidence=0.8, code="", error_type="none", reason="fine"):
    """A model reply: JSON, plus a code block when code is given."""
    data = {"restated": "restated", "verdict": verdict, "error_type": error_type,
            "reason": reason, "confidence": confidence}
    text = "<think>let me see {x}</think>\n" + json.dumps(data)
    return text + (f"\n```python\n{code}\n```" if code else "")


class FakeClient:
    """Answers each equation from a dict {equation number: reply text or list of texts}.
    The equation is found in the prompt ("Equation (2): ...")."""

    def __init__(self, replies, default=None, tests=None):
        self.tests = tests or {}                      # {equation number: reply to "write the test"}
        self.replies = {k: (list(v) if isinstance(v, list) else [v]) for k, v in replies.items()}
        self.default = default or reply("ok")
        self.models, self.prompts = [], []

    async def call_model(self, model, messages, images=None):
        self.models.append(model)
        prompt = messages[1]["content"]              # the first user message, even on a retry
        self.prompts.append(prompt)
        number = next(line.split("(")[1].split(")")[0] for line in prompt.splitlines()
                      if line.startswith("Equation ("))
        if "Write one short Python test" in prompt:
            self.test_calls = getattr(self, "test_calls", 0) + 1
            return ModelResponse(text=self.tests.get(number, "NONE"), model_name=model, latency=0.0)        
        queue = self.replies.get(number)
        text = (queue.pop(0) if len(queue) > 1 else queue[0]) if queue else self.default
        return ModelResponse(text=text, model_name=model, latency=0.0)


def paper(*items):
    """ParsedPaper from (number, latex) pairs; the text around each equation is made up."""
    eqs = [EquationInfo(index=i, number=n, latex=l, section="2 Method",
                        before=f"before {n}", after=f"after {n}", source="tex")
           for i, (n, l) in enumerate(items)]
    return ParsedPaper(id="p1", equation_meta=eqs)


PAPER = paper(("1", "d = |a - b|^2"), ("2", "d = |a|^2 + |b|^2 - 2 Re(a b^*)"),
              ("3", "d \\ge (|a| - |b|)^2"))


def run(client, doc=PAPER, **kw):
    return asyncio.run(mc.check_math(doc, client, **kw))


# ------------------------------------------------------------------ finding the steps
def test_plain_paperdoc_strings_are_parsed():
    doc = PaperDoc(id="x", equations=["(1) a = b", "(2.1) c <= d", "no number here = 5"])
    eqs = mc.get_equations(doc)
    assert [e.number for e in eqs] == ["1", "2.1", None]
    assert eqs[0].latex == "a = b"


def test_steps_skip_unnumbered_short_and_relationless_equations():
    doc = paper(("1", "max_x f(x)"), ("2", "y = Hx + n"), (None, "z = 1"), ("4", "a=b"),
                ("5", "q = r + s"))
    steps = mc.build_steps(doc)
    assert [s["number"] for s in steps] == ["2", "5"]      # "max f" has no relation, "a=b" is short


def test_step_carries_previous_equation_and_text():
    steps = mc.build_steps(PAPER)
    assert steps[0]["previous"] is None
    assert steps[1]["previous"] == ("1", "d = |a - b|^2")
    assert steps[1]["before"] == "before 2" and steps[1]["section"] == "2 Method"


def test_numbers_and_max_equations_limit_the_work():
    assert [s["number"] for s in mc.build_steps(PAPER, numbers=["2"])] == ["2"]
    assert len(mc.build_steps(PAPER, max_equations=2)) == 2
    client = FakeClient({})
    run(client, numbers=["3"])
    assert len(client.prompts) == 1


def test_prompt_shows_previous_equation_and_the_equation():
    client = FakeClient({})
    run(client, numbers=["2"])
    prompt = client.prompts[0]
    assert "Previous equation: (1) d = |a - b|^2" in prompt
    assert "Equation (2): d = |a|^2" in prompt
    assert "before 2" in prompt and "after 2" in prompt


# ---------------------------------------------------------------------- reading replies
def test_read_reply_ignores_think_block_and_splits_code():
    data, code = mc.read_reply(reply("error", code="print('PASS')"))
    assert data["verdict"] == "error" and code == "print('PASS')"


def test_read_reply_rejects_bad_verdict_and_bad_json():
    assert mc.read_reply('{"verdict": "maybe"}')[0] is None
    assert mc.read_reply("no json at all")[0] is None
    assert mc.read_reply("<think>{\"verdict\": \"ok\"}")[0] is None     # unfinished think block


def test_braces_inside_the_code_do_not_break_the_json():
    code = "d = {'a': 1}\nprint(f'PASS {d}')"
    data, got = mc.read_reply(reply("ok", code=code))
    assert data["verdict"] == "ok" and got == code


def test_latex_backslashes_inside_the_json_text_are_tolerated():
    text = r'{"restated": "uses \alpha and \mathbf{R}", "verdict": "error", "reason": "\sum is wrong", "confidence": 0.7}'
    data, _ = mc.read_reply(text)
    assert data is not None and data["verdict"] == "error"
    assert "alpha" in data["restated"] and "mathbf" in data["restated"]


# ------------------------------------------------------------------ decisions and findings
def test_clean_derivation_has_no_findings():
    client = FakeClient({"2": reply("ok", code=TEST_GOOD)})
    result = run(client)
    assert isinstance(result, AgentResult)
    assert result.findings == [] and result.needs_human is False


def test_error_confirmed_by_failing_test_is_high_and_confident():
    client = FakeClient({"2": reply("error", 0.7, code=TEST_BAD, error_type="sign",
                                    reason="The cross term should be negative.")})
    result = run(client)
    assert len(result.findings) == 1
    f = result.findings[0]
    assert (f.type, f.severity, f.confidence) == ("math_error", "high", mc.CONFIRMED_CONF)
    assert f.evidence.startswith("Equation (2):") and "FAIL" in f.evidence and "sign" in f.evidence
    assert result.needs_human is False


def test_error_without_a_test_is_medium_capped_and_needs_a_human():
    client = FakeClient({"2": reply("error", 0.99, error_type="sign")})
    result = run(client)
    f = result.findings[0]
    assert (f.type, f.severity) == ("math_error", "medium")
    assert f.confidence == mc.UNVERIFIED_CONF_MAX
    assert result.needs_human is True and "No numeric test" in f.evidence


def test_error_with_a_broken_test_counts_as_untested():
    client = FakeClient({"2": reply("error", 0.5, code="import os\nprint('PASS')")})
    f = run(client).findings[0]
    assert (f.type, f.severity, f.confidence) == ("math_error", "medium", 0.5)
    assert "ERROR" in f.evidence


def test_error_but_passing_test_is_a_disagreement_for_a_human():
    client = FakeClient({"2": reply("error", 0.9, code=TEST_GOOD)})
    result = run(client)
    f = result.findings[0]
    assert (f.type, f.severity, f.confidence) == ("math_unsure", "low", mc.UNSURE_CONF)
    assert result.needs_human is True


def test_ok_but_failing_test_is_a_disagreement_for_a_human():
    client = FakeClient({"2": reply("ok", 0.9, code=TEST_BAD)})
    result = run(client)
    assert [f.type for f in result.findings] == ["math_unsure"]
    assert result.needs_human is True


def test_unsure_and_unreadable_replies_are_unchecked_not_findings():
    client = FakeClient({"1": reply("unsure"), "2": "I do not know", "3": reply("ok")})
    result = run(client)
    assert result.findings == [] and result.needs_human is False
    records = json.loads(result.raw_output)
    assert [r["verdict"] for r in records] == ["unchecked", "unchecked", "ok"]
    assert "no readable answer" in records[1]["reason"]


def test_one_retry_on_an_unreadable_reply():
    client = FakeClient({"2": ["not json", reply("error", 0.9, code=TEST_BAD)]})
    result = run(client, numbers=["2"])
    assert [f.type for f in result.findings] == ["math_error"]
    assert len(client.prompts) == 2


def test_hanging_test_does_not_hang_the_checker():
    client = FakeClient({"2": reply("error", 0.8, code="while True:\n    pass")})
    result = run(client, numbers=["2"], timeout=2)
    f = result.findings[0]
    assert (f.severity, f.confidence) == ("medium", mc.UNVERIFIED_CONF_MAX)
    assert "timed out" in f.evidence


def test_model_knob_and_raw_output():
    client = FakeClient({"2": reply("error", 0.8, code=TEST_BAD)})
    result = run(client, model="qwen3:14b")
    assert set(client.models) == {"qwen3:14b"}
    records = json.loads(result.raw_output)
    assert [r["number"] for r in records] == ["1", "2", "3"]
    assert records[1]["test"]["status"] == "fail" and records[1]["decision"][0] == "math_error"
    assert 0 <= result.findings[0].confidence <= 1

# ---------------------------------------------------------- equations read from PDF text
def test_findings_on_pdf_equations_are_lowered_and_need_a_human():
    pdf_paper = ParsedPaper(id="p2", equation_meta=[
        EquationInfo(index=0, number="2", latex="d = |a|^2 + |b|^2 + 2 Re(a b^*)", source="pdf")])
    client = FakeClient({"2": reply("error", 0.9, code=TEST_BAD, error_type="sign")})
    result = run(client, doc=pdf_paper)
    f = result.findings[0]
    assert (f.type, f.severity) == ("math_error", "medium")
    assert f.confidence == mc.PDF_CONF_MAX and result.needs_human is True


# ------------------------------------------------------------- keeping the reply for debugging
def test_reply_end_is_kept_when_no_test_was_written_for_an_error():
    client = FakeClient({"2": reply("error", 0.9, reason="no test this time")})
    records = json.loads(run(client).raw_output)
    assert "no test this time" in records[1]["reply_end"]
    assert records[0]["reply_end"] == ""                 # an ok answer needs no explanation


def test_reply_end_is_kept_for_unreadable_replies():
    client = FakeClient({"2": "I cannot say"})
    records = json.loads(run(client).raw_output)
    assert records[1]["reply_end"] == "I cannot say"



# ------------------------------------------------- reading fenced replies, asking for the test
def test_json_in_a_fence_followed_by_a_python_block():
    text = ("```json\n" + json.dumps({"verdict": "error", "confidence": 0.8}) + "\n```\n\n"
            "```python\nprint('FAIL 1 2')\n```")
    data, code = mc.read_reply(text)
    assert data["verdict"] == "error" and code == "print('FAIL 1 2')"


def test_python_block_before_a_json_fence_and_unmarked_code():
    text = "```python\nd = {'a': 1}\nprint('PASS')\n```\n```json\n{\"verdict\": \"ok\"}\n```"
    data, code = mc.read_reply(text)
    assert data["verdict"] == "ok" and "print('PASS')" in code
    data, code = mc.read_reply('{"verdict": "ok"}\n```\nx = 1\nprint("PASS")\n```')
    assert data["verdict"] == "ok" and "print(" in code


def test_error_without_code_asks_for_the_test_and_a_failing_test_confirms_it():
    client = FakeClient({"2": reply("error", 0.7, error_type="sign")},
                        tests={"2": f"```python\n{TEST_BAD}\n```"})
    result = run(client)
    f = result.findings[0]
    assert (f.type, f.severity, f.confidence) == ("math_error", "high", mc.CONFIRMED_CONF)
    assert client.test_calls == 1 and result.needs_human is False


def test_test_writer_saying_none_leaves_the_error_unverified():
    client = FakeClient({"2": reply("error", 0.7)}, tests={"2": "NONE"})
    f = run(client).findings[0]
    assert (f.type, f.severity) == ("math_error", "medium")


def test_passing_test_from_the_test_writer_sends_the_error_to_a_human():
    client = FakeClient({"2": reply("error", 0.9)}, tests={"2": f"```python\n{TEST_GOOD}\n```"})
    result = run(client)
    assert [f.type for f in result.findings] == ["math_unsure"] and result.needs_human is True


def test_test_is_only_requested_for_errors_without_code():
    client = FakeClient({"1": reply("ok"), "2": reply("error", 0.8, code=TEST_BAD), "3": reply("ok")})
    run(client)
    assert getattr(client, "test_calls", 0) == 0