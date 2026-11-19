# File: src/agents/math_checker.py
# Created: 2026-11-18
# Created by: Lathika
# Purpose: Check each numbered equation of a paper with the math model plus a numeric test.
#          Updated 2026-11-18: definitions found elsewhere in the paper are added to each
#          prompt; findings on equations read from PDF text are demoted to a human check.

import asyncio
import json
import re

from src.common.schemas import AgentResult, Finding
from src.ingest.models import EquationInfo
from src.tools import notation, sympy_check

MATH_MODEL = "deepseek-r1:7b"       # the other allowed math model is "qwen3:14b" (model=...)
MAX_EQUATIONS = 60                  # most equations checked per paper (one model call each)
MIN_LATEX = 5                       # shorter equations are not worth a call
CONFIRMED_CONF = 0.90               # confidence when the model and its numeric test agree
UNVERIFIED_CONF_MAX = 0.60          # an error nobody tested never gets more than this
UNSURE_CONF = 0.30                  # model and test disagree
PDF_CONF_MAX = 0.50                 # equations read from PDF text are often garbled

VERDICTS = ("ok", "error", "unsure")
ERROR_TYPES = ("sign", "missing_term", "factor", "index", "other", "none")

_RELATION = re.compile(r"=|<|>|≤|≥|≈|\\le\b|\\ge\b|\\leq|\\geq|\\approx|\\equiv")
_NUMBERED = re.compile(r"^\((\S+?)\)\s+(.*)$", re.S)        # "(3) x = y" as stored in PaperDoc
_THINK = re.compile(r"<think>.*?</think>", re.S)
_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
_LONE_BACKSLASH = re.compile(r'\\(?!["\\/bfnrtu])')    # LaTeX such as \alpha is not valid JSON

SYSTEM = ("You are a careful mathematician checking one step of a derivation in a research "
          "paper. You work the step out yourself before you judge it.")


# ------------------------------------------------------------------ equations to check
def get_equations(paper):
    """The paper's equations as EquationInfo. A ParsedPaper already has them with their
    surrounding text; a plain PaperDoc only has strings like "(3) x = y"."""
    meta = getattr(paper, "equation_meta", None)
    if meta:
        return list(meta)
    out = []
    for text in paper.equations:
        found = _NUMBERED.match(text.strip())
        number, latex = (found.group(1), found.group(2)) if found else (None, text)
        out.append(EquationInfo(index=len(out), number=number, latex=latex))
    return out


def build_steps(paper, numbers=None, max_equations=MAX_EQUATIONS):
    """One step per numbered equation that states a relation (=, <, >, ...). Each step has the
    previous equation, definitions of its symbols found elsewhere in the paper, and the
    equations its surrounding text refers to. numbers: only these equation numbers."""
    equations = get_equations(paper)
    sentences = notation.paper_sentences(paper)
    steps, previous = [], None
    for eq in equations:
        if eq.number is None:
            continue
        wanted = numbers is None or eq.number in numbers
        if wanted and len(eq.latex) >= MIN_LATEX and _RELATION.search(eq.latex):
            around = eq.before + " " + eq.after
            steps.append({
                "number": eq.number, "latex": eq.latex, "section": eq.section,
                "before": eq.before, "after": eq.after, "source": eq.source,
                "previous": previous,
                "definitions": notation.definitions_for(eq.latex, sentences, skip=around),
                "referenced": notation.referenced_equations(
                    around, equations, eq.number, previous[0] if previous else None)})
        previous = (eq.number, eq.latex)
    return steps[:max_equations]


# ------------------------------------------------------------------------ model calls
def make_prompt(step):
    prev = "none"
    if step["previous"]:
        prev = f"({step['previous'][0]}) {step['previous'][1]}"
    defined = "".join(f"- {d}\n" for d in step["definitions"]) or "(none found)\n"
    referred = "".join(f"({n}) {latex}\n" for n, latex in step["referenced"]) or "(none)\n"
    return (
        f"Section: {step['section'] or 'unknown'}\n"
        f"Definitions of its symbols found elsewhere in the paper:\n{defined}"
        f"Equations the text refers to:\n{referred}"
        f"Text before the equation: {step['before'] or '(none)'}\n"
        f"Previous equation: {prev}\n"
        f"Equation ({step['number']}): {step['latex']}\n"
        f"Text after the equation: {step['after'] or '(none)'}\n\n"
        "Do this:\n"
        "1. Restate in one or two sentences what the step claims (what equals what, or what "
        "follows from the previous equation).\n"
        "2. Work it out yourself and decide whether it is correct as written. Look for sign "
        "errors, dropped or extra terms, wrong factors, wrong powers or indices. A definition "
        "or a modelling assumption cannot be wrong: answer ok. If the LaTeX is garbled, cut "
        "off, or needs context you do not have, answer unsure.\n"
        "3. If the step is an identity or inequality you can evaluate on numbers, write a short "
        "Python test using only numpy and sympy. It draws random values with a fixed seed, "
        "evaluates both sides exactly as the paper writes them (do not correct the paper's "
        "equation inside the test), and prints PASS if they agree within 1e-6, or FAIL "
        "followed by both values. If the step rewrites the previous equation, compare the two "
        "expressions. Skip the test for definitions.\n\n"
        "Reply with one JSON object (no LaTeX backslashes inside its text), then, only if you "
        "wrote a test, one python code block:\n"
        '{"restated": "...", "verdict": "ok" or "error" or "unsure", '
        '"error_type": "sign" or "missing_term" or "factor" or "index" or "other" or "none", '
        '"reason": "one or two sentences", "confidence": 0.0 to 1.0}'
    )


def read_reply(text):
    """Split a model reply into (JSON dict or None, test code or "")."""
    text = _THINK.sub("", text)
    if "<think>" in text:                       # thinking that never finished: no answer yet
        text = text[:text.index("<think>")]
    code = ""
    found = _CODE.search(text)
    if found:
        code = found.group(1).strip()
        text = text[:found.start()] + text[found.end():]
    start, end = text.find("{"), text.rfind("}")
    data = None
    if 0 <= start < end:
        raw = text[start:end + 1]
        for attempt in (raw, _LONE_BACKSLASH.sub(r"\\\\", raw)):
            try:
                data = json.loads(attempt)
                break
            except ValueError:
                data = None
    if not isinstance(data, dict) or str(data.get("verdict", "")).strip().lower() not in VERDICTS:
        data = None
    return data, code


async def ask_model(step, client, model):
    """Ask about one step; one retry if the reply cannot be read. Returns (data, code, last reply text)."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": make_prompt(step)}]
    for _ in range(2):
        reply = await client.call_model(model=model, messages=messages)
        data, code = read_reply(reply.text)
        if data is not None:
            return data, code, reply.text
        messages = messages + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": "I could not read that. Reply with the JSON object "
                                        "(verdict must be ok, error or unsure), then the code block if any."}]
    return None, "", reply.text


# ------------------------------------------------------------------------- the decision
def decide(verdict, confidence, test_status):
    """(finding type, severity, confidence, needs_human), or None when there is no finding.
    test_status: "pass", "fail", or "none" (no test, or the test could not run).
      error + test fail   -> math_error, high, confident
      error + no test     -> math_error, medium, a human should confirm
      error + test pass   -> math_unsure (model and test disagree), human
      ok    + test fail   -> math_unsure (model and test disagree), human
      anything else       -> no finding"""
    if verdict == "error":
        if test_status == "fail":
            return ("math_error", "high", CONFIRMED_CONF, False)
        if test_status == "pass":
            return ("math_unsure", "low", UNSURE_CONF, True)
        return ("math_error", "medium", min(confidence, UNVERIFIED_CONF_MAX), True)
    if verdict == "ok" and test_status == "fail":
        return ("math_unsure", "low", UNSURE_CONF, True)
    return None


def for_pdf_text(decision):
    """Equations read from PDF text are often garbled: lower the finding and ask a human."""
    kind, severity, confidence, _ = decision
    return (kind, "medium" if severity == "high" else severity, min(confidence, PDF_CONF_MAX), True)


async def check_step(step, client, model=MATH_MODEL, timeout=sympy_check.TIMEOUT):
    """Check one equation and return its record (a plain dict, saved in raw_output)."""
    data, code, last_reply = await ask_model(step, client, model)
    record = {"number": step["number"], "latex": step["latex"], "source": step["source"],
              "verdict": "unchecked", "error_type": "none", "reason": "", "restated": "",
              "confidence": 0.0, "test": {"status": "none", "detail": ""}, "decision": None,
              "reply_end": ""}
    if data is None:
        record["reason"] = "the model gave no readable answer"
        record["reply_end"] = last_reply[-600:]
        return record

    verdict = str(data["verdict"]).strip().lower()
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
    except (ValueError, TypeError):
        confidence = 0.5
    error_type = str(data.get("error_type", "none")).strip().lower()
    record.update(verdict=verdict, confidence=confidence,
                  error_type=error_type if error_type in ERROR_TYPES else "other",
                  reason=str(data.get("reason", "")).strip(),
                  restated=str(data.get("restated", "")).strip())
    if verdict == "unsure":
        record["verdict"] = "unchecked"           # nothing to report; kept in raw_output
        record["reply_end"] = last_reply[-600:]
        return record

    if code:
        record["test"] = await asyncio.to_thread(sympy_check.run_test, code, timeout)
    status = record["test"]["status"] if record["test"]["status"] in ("pass", "fail") else "none"
    record["decision"] = decide(verdict, confidence, status)
    if record["decision"] and step["source"] == "pdf":
        record["decision"] = for_pdf_text(record["decision"])
    if not code and verdict != "ok":
        record["reply_end"] = last_reply[-600:]   # why did the model write no test?
    return record


# ------------------------------------------------------------------------- the result
def to_agent_result(records):
    """Turn the per-equation records into the shared AgentResult."""
    findings, needs_human = [], False
    for r in records:
        if r["decision"] is None:
            continue
        kind, severity, confidence, human = r["decision"]
        test = r["test"]
        tested = f" Numeric test: {test['status'].upper()}. {test['detail']}" if test["status"] != "none" else \
            " No numeric test was run."
        findings.append(Finding(
            type=kind, severity=severity, confidence=confidence,
            evidence=(f"Equation ({r['number']}): {r['reason']} "
                      f"(error type: {r['error_type']}, model verdict: {r['verdict']}).{tested}")))
        needs_human = needs_human or human
    return AgentResult(findings=findings, needs_human=needs_human, raw_output=json.dumps(records))


async def check_math(paper, client, model=MATH_MODEL, numbers=None, max_equations=MAX_EQUATIONS,
                     timeout=sympy_check.TIMEOUT):
    """Check the derivation steps of a PaperDoc (or ParsedPaper) and return an AgentResult.

    client: anything with async call_model(model=..., messages=...) such as LLMClient.
    model: the math model (deepseek-r1:7b by default, qwen3:14b is the other allowed one).
    numbers: only check these equation numbers, e.g. ["26", "27"] (default: all).
    max_equations: most equations checked (one model call each, the slow part).
    """
    steps = build_steps(paper, numbers, max_equations)
    records = []
    for step in steps:                              # one at a time: one GPU
        records.append(await check_step(step, client, model, timeout))
    return to_agent_result(records)