# File: src/agents/vision_checker.py
# Created: 2026-11-19
# Created by: Lathika
# Purpose: Compare the numbers a paper's text states about a figure or table with what the figure
#          or table shows. The vision model reads the values; the decision rules are in code.

import base64
import json
import math
import re
from pathlib import Path

from rapidfuzz import fuzz

from src.agents.ref_verifier import normalize
from src.common.schemas import AgentResult, Finding

VISION_MODEL = "qwen2.5vl:7b"       # the other allowed vision model is "gemma3:4b" (model=...)
MAX_ITEMS = 30                      # most figures and tables checked per paper
MAX_CLAIMS = 4                      # most numeric statements checked per figure or table
QUOTE_MATCH = 85                    # how well a claim's quote must appear in the citing text
TOL_TABLE = 0.01                    # table cell vs text: relative difference allowed
TOL_PRINTED = 0.03                  # number printed on a figure vs text
TOL_ESTIMATED = 0.10                # value read off the axes: up to this it matches
BIG_ESTIMATED = 0.20                # above this it is a mismatch; in between only "unsure"
TABLE_CONF = 0.90
PRINTED_CONF = 0.85
ESTIMATED_CONF_MAX = 0.60           # a value estimated from the axes never gets more than this
UNSURE_CONF = 0.30
CROP_CONF_MAX = 0.50                # the crop is a blind guess (no graphic found by the parser)

REPO_ROOT = Path(__file__).resolve().parents[2]

SYSTEM = ("You check numbers in a research paper against its figures and tables. "
          "You report only what you can see. You never guess.")

_THINK = re.compile(r"<think>.*?</think>", re.S)
_LONE_BACKSLASH = re.compile(r'\\(?!["\\/bfnrtu])')    # LaTeX such as \alpha is not valid JSON
_NUMBER = re.compile(r"(?<![\w.])[-−–]?\d[\d,]*\.?\d*")


# ------------------------------------------------------------------------- small helpers
def numbers_in(text):
    """All numbers in a text, as floats ("1,000" is 1000, a leading minus sign is kept)."""
    found = []
    for match in _NUMBER.finditer(text):
        raw = match.group().replace(",", "").replace("−", "-").replace("–", "-")
        try:
            found.append(float(raw.rstrip(".")))
        except ValueError:
            pass
    return found


def to_number(value):
    """A number from a model's JSON value (5, 5.0, "91.3%"), or None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        found = numbers_in(str(value))
        number = found[0] if found else None
    return number if number is not None and math.isfinite(number) else None


def rescale(read, claimed):
    """The read value in the scale of the claim (a table may hold 0.913 where the text says 91.3%)."""
    return min((read, read * 100, read / 100), key=lambda v: abs(v - claimed))


def relative_difference(claimed, read):
    return abs(claimed - read) / max(abs(read), 1e-9) if (claimed or read) else 0.0


def read_json(text):
    """The first JSON object in a model reply, or None (think blocks and code fences are ignored)."""
    text = _THINK.sub("", text)
    start, end = text.find("{"), text.rfind("}")
    if not 0 <= start < end:
        return None
    raw = text[start:end + 1]
    for attempt in (raw, _LONE_BACKSLASH.sub(r"\\\\", raw)):
        try:
            data = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def load_image(path, root=REPO_ROOT):
    """The image file as a base64 string (what Ollama expects), or None if it is not there."""
    file = Path(path) if path else None
    if file is not None and not file.is_absolute():
        file = Path(root) / file
    if file is None or not file.is_file():
        return None
    return base64.b64encode(file.read_bytes()).decode("ascii")


# ------------------------------------------------------------------ what there is to check
def build_items(paper, ids=None, max_items=MAX_ITEMS):
    """The figures and tables of a ParsedPaper as plain dicts. A plain PaperDoc has no captions
    or citing paragraphs, so it gives nothing. ids: only these, e.g. ["Figure 2", "Table 1"]."""
    items = []
    for fig in getattr(paper, "figure_meta", None) or []:
        items.append({"id": fig.id, "kind": "figure", "caption": fig.caption, "path": fig.path,
                      "text": "", "paragraphs": list(fig.citing_paragraphs),
                      "fallback_crop": fig.fallback_crop})
    for tab in getattr(paper, "table_meta", None) or []:
        items.append({"id": tab.id, "kind": "table", "caption": tab.caption, "path": "",
                      "text": tab.text, "paragraphs": list(tab.citing_paragraphs),
                      "fallback_crop": False})
    if ids:
        items = [item for item in items if item["id"] in ids]
    return items[:max_items]


# ------------------------------------------------------------------------------ prompts
def claims_prompt(item):
    paragraphs = "".join(f"[{i + 1}] {p}\n" for i, p in enumerate(item["paragraphs"]))
    return (
        f"{item['id']} caption: {item['caption']}\n\n"
        f"Text of the paper that cites {item['id']}:\n{paragraphs}\n"
        f"List the numeric statements in this text that can be checked against {item['id']} "
        "(a value, a percentage, a ratio, a count, an improvement, a position). Skip "
        "statements about other figures or tables and numbers that are not in the figure "
        "(number of runs, sample sizes). For each give:\n"
        '"quote": the words of the text, copied exactly, at most one sentence;\n'
        '"question": a short question that says where to read it (for example "Accuracy of the '
        'proposed method"); name the curve, bar or row and the x value;\n'
        '"claimed": the number the text states; "unit": its unit, or "".\n'
        'Reply with one JSON object only: {"claims": [{"quote": "...", "question": "...", '
        '"claimed": 0.0, "unit": "..."}]}. If the text states no checkable number, reply '
        '{"claims": []}.')


def questions_list(claims):
    return "".join(f"{i + 1}. {c['question']}\n" for i, c in enumerate(claims))


def figure_read_prompt(item, claims):
    return (
        f"This image is {item['id']} of a research paper. Caption: {item['caption']}\n\n"
        f"Read these quantities from the image:\n{questions_list(claims)}\n"
        'For each give: "value": the number you read; "printed": true if that exact number is '
        "printed on the figure, false if you estimated it from the axes and gridlines; "
        '"note": a few words on what you looked at. Use null for value if you cannot find the '
        "quantity. Do not guess.\n"
        'Reply with one JSON object only: {"readings": [{"n": 1, "value": 0.0, "printed": true, '
        '"note": "..."}]}')


def table_read_prompt(item, claims):
    return (
        f"{item['id']} of a research paper. Caption: {item['caption']}\n"
        "Table text (one row per line, cells separated by ' | '):\n"
        f"{item['text']}\n\n"
        f"Find these values in the table:\n{questions_list(claims)}\n"
        'For each give: "cell": the cell copied exactly as written in the table; "row": the row '
        'label. Use null for cell if the table has no such value. Do not guess.\n'
        'Reply with one JSON object only: {"readings": [{"n": 1, "cell": "...", "row": "..."}]}')


async def ask_list(client, model, prompt, key, images=None):
    """Ask once more if the reply cannot be read. Returns the list under key, or None."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    for _ in range(2):
        reply = await client.call_model(model=model, messages=messages, images=images)
        data = read_json(reply.text)
        if data is not None and isinstance(data.get(key), list):
            return data[key]
        messages = messages + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": f'I could not read that. Reply with only the JSON object '
                                        f'with the list "{key}".'}]
    return None


# ----------------------------------------------------------------------- checking the claims
def valid_claims(raw, paragraphs):
    """(claims that can be trusted, claims dropped with the reason). A claim needs a quote that is
    really in the citing text and the claimed number must appear in that quote."""
    text = normalize(" ".join(paragraphs))
    good, dropped = [], []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        claim = {"quote": str(entry.get("quote", "")).strip(),
                 "question": str(entry.get("question", "")).strip(),
                 "claimed": to_number(entry.get("claimed")),
                 "unit": str(entry.get("unit", "")).strip()}
        why = ""
        if not claim["quote"] or not claim["question"] or claim["claimed"] is None:
            why = "incomplete"
        elif fuzz.partial_ratio(normalize(claim["quote"]), text) < QUOTE_MATCH:
            why = "quote not found in the text"
        elif not any(math.isclose(claim["claimed"], n, rel_tol=1e-6, abs_tol=1e-9)
                     for n in numbers_in(claim["quote"])):
            why = "claimed number not in the quote"
        if why:
            claim["dropped"] = why
            dropped.append(claim)
        else:
            good.append(claim)
    return good[:MAX_CLAIMS], dropped


def table_cell_ok(reading, table_text):
    """The cell the model copied must be a whole cell of the table, in a row with that label."""
    cell = str(reading.get("cell") or "").strip()
    if not cell:
        return False
    lines = [line for line in table_text.splitlines()
             if cell in [c.strip() for c in line.split("|")]]
    row = normalize(str(reading.get("row") or ""))
    return any(fuzz.partial_ratio(row, normalize(line)) >= 80 for line in lines) if row else bool(lines)


def readings_for(item, claims, raw):
    """One reading per claim: {"read": number or None, "how": "table" | "printed" | "estimated", "note"}."""
    out = [{"read": None, "how": "", "note": "not found"} for _ in claims]
    for entry in raw:
        number = to_number(entry.get("n")) if isinstance(entry, dict) else None
        if number is None or not 1 <= int(number) <= len(claims):
            continue
        slot = out[int(number) - 1]
        if item["kind"] == "table":
            cell = str(entry.get("cell") or "").strip()
            if not table_cell_ok(entry, item["text"]):
                slot["note"] = "the cell is not in the table text"
            elif not numbers_in(cell):
                slot["note"] = "the cell holds no number"
            else:
                slot.update(read=numbers_in(cell)[0], how="table",
                            note=f"cell '{cell}' in row '{entry.get('row')}'")
        else:
            value = to_number(entry.get("value"))
            if value is not None:
                slot.update(read=value, how="printed" if entry.get("printed") is True else "estimated",
                            note=str(entry.get("note") or "")[:100])
    return out


# ------------------------------------------------------------------------- the decision
def decide(kind, claimed, read, how):
    """(finding type, severity, confidence, needs_human), or None when text and figure agree.
    how: "table" (cell copied from the table text), "printed" or "estimated" (from the axes).
      table cell off by more than TOL_TABLE        -> table_mismatch, high
      printed number off by more than TOL_PRINTED  -> figure_mismatch, high
      estimated, off by more than BIG_ESTIMATED    -> figure_mismatch, medium, a human checks
      estimated, off by more than TOL_ESTIMATED    -> figure_unsure, low, a human checks"""
    diff = relative_difference(claimed, read)
    if how == "table":
        return ("table_mismatch", "high", TABLE_CONF, False) if diff > TOL_TABLE else None
    if how == "printed":
        return (f"{kind}_mismatch", "high", PRINTED_CONF, False) if diff > TOL_PRINTED else None
    if diff > BIG_ESTIMATED:
        return (f"{kind}_mismatch", "medium", ESTIMATED_CONF_MAX, True)
    if diff > TOL_ESTIMATED:
        return (f"{kind}_unsure", "low", UNSURE_CONF, True)
    return None


def for_blind_crop(decision):
    """The parser found no graphic and cropped blindly: lower the finding and ask a human."""
    kind, severity, confidence, _ = decision
    return (kind, "medium" if severity == "high" else severity, min(confidence, CROP_CONF_MAX), True)


def make_evidence(item, claim):
    unit = claim["unit"]
    if claim["how"] == "table":
        where = f"the table has {claim['read']:g} ({claim['note']})"
    else:
        how = "printed on the figure" if claim["how"] == "printed" else "estimated from the axes"
        where = f"the figure shows {claim['read']:g}, {how}, for '{claim['question']}'"
    crop = " The figure crop is a blind guess." if item["fallback_crop"] else ""
    return (f"{item['id']}: the text says {claim['claimed']:g}{unit} ('{claim['quote'][:140]}'), "
            f"{where}; relative difference {claim['difference']:.0%}.{crop}")


# ---------------------------------------------------------------------- one figure or table
def new_record(item):
    return {"id": item["id"], "kind": item["kind"], "status": "checked", "reason": "",
            "claims": [], "dropped": []}


async def check_item(item, client, model, image_root=REPO_ROOT):
    """Check one figure or table and return its record (a plain dict, saved in raw_output)."""
    record = new_record(item)
    if not item["paragraphs"]:
        record.update(status="skipped", reason="no paragraph cites it")
        return record
    image = None
    if item["kind"] == "figure":
        image = load_image(item["path"], image_root)
        if image is None:
            record.update(status="skipped", reason="the image file was not found")
            return record
    elif not item["text"].strip():
        record.update(status="skipped", reason="the table has no text")
        return record

    raw = await ask_list(client, model, claims_prompt(item), "claims")
    if raw is None:
        record.update(status="unchecked", reason="no readable list of claims")
        return record
    claims, record["dropped"] = valid_claims(raw, item["paragraphs"])
    if not claims:
        return record

    if item["kind"] == "figure":
        raw = await ask_list(client, model, figure_read_prompt(item, claims), "readings", [image])
    else:
        raw = await ask_list(client, model, table_read_prompt(item, claims), "readings")
    if raw is None:
        record.update(status="unchecked", reason="no readable values")
        return record

    for claim, got in zip(claims, readings_for(item, claims, raw)):
        claim.update(got, difference=None, decision=None, evidence="")
        if got["read"] is None:
            continue
        claim["read"] = rescale(got["read"], claim["claimed"])
        claim["difference"] = relative_difference(claim["claimed"], claim["read"])
        decision = decide(item["kind"], claim["claimed"], claim["read"], got["how"])
        if decision and item["fallback_crop"]:
            decision = for_blind_crop(decision)
        claim["decision"] = decision
        if decision:
            claim["evidence"] = make_evidence(item, claim)
    record["claims"] = claims
    return record


# ------------------------------------------------------------------------- the result
def to_agent_result(records):
    """Turn the per-item records into the shared AgentResult."""
    findings, needs_human = [], False
    for record in records:
        for claim in record["claims"]:
            if claim["decision"] is None:
                continue
            kind, severity, confidence, human = claim["decision"]
            findings.append(Finding(type=kind, severity=severity, confidence=confidence,
                                    evidence=claim["evidence"]))
            needs_human = needs_human or human
    return AgentResult(findings=findings, needs_human=needs_human, raw_output=json.dumps(records))


async def check_vision(paper, client, model=VISION_MODEL, ids=None, max_items=MAX_ITEMS,
                       image_root=REPO_ROOT):
    """Check the figures and tables of a ParsedPaper and return an AgentResult.

    client: anything with async call_model(model=..., messages=..., images=...) such as LLMClient.
    model: the vision model (qwen2.5vl:7b by default, gemma3:4b is the other allowed one).
    ids: only check these, e.g. ["Figure 2", "Table 1"] (default: all).
    image_root: figure paths that are not absolute are relative to this folder (the repo root).
    """
    records = []
    for item in build_items(paper, ids, max_items):      # one at a time: one GPU
        try:
            records.append(await check_item(item, client, model, image_root))
        except Exception as error:                       # a model timeout must not lose the other items
            record = new_record(item)
            record.update(status="unchecked", reason=f"the model call failed: {type(error).__name__}")
            records.append(record)
    return to_agent_result(records)
