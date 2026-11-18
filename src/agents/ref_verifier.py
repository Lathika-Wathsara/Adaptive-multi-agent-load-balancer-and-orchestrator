# Created: 2026-11-17
# Created by: Lathika
# Purpose: Check that every reference in a paper is a real paper. Each reference is
#          looked up through src/tools/scholar.py and fuzzy-matched on title, year
#          and authors. Result per reference: exists, not_found, mismatch or unchecked.

import json
import re
import unicodedata

from rapidfuzz import fuzz

from src.common.schemas import AgentResult, Finding
from src.tools import scholar

TITLE_SAME = 90    # title score at or above this: same paper
TITLE_MAYBE = 75   # between MAYBE and SAME: ambiguous, ask the judge if there is one
YEAR_SLACK = 2     # preprint year and journal year often differ a little


def normalize(text):
    """Lower case, no accents, only letters and digits separated by single spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def guess_title(ref):
    """Best guess of the title inside a reference string (used as the search query)."""
    ref = re.sub(r"^\s*(\[\d+\]|\d+[.)])\s*", "", ref)           # leading "[3]" or "3."
    quoted = re.search(r'[\u201c"](.{15,}?)[\u201d"]', ref)       # IEEE: "Title,"
    if quoted:
        return quoted.group(1).strip(" ,.")
    after_year = re.search(r"\((?:19|20)\d{2}[a-z]?\)\.?\s*(.+?)[.?]\s", ref)   # APA
    if after_year:
        return after_year.group(1).strip()
    chunks = [c for c in re.split(r"[.?]\s+", ref) if len(c.split()) >= 4]
    return max(chunks, key=len) if chunks else ref[:200]


def title_score(cand_title, ref_norm, guess_norm):
    """0-100: how well the candidate's title appears in the reference text."""
    cand_norm = normalize(cand_title)
    if not cand_norm:
        return 0
    if len(cand_norm) >= 20:
        return fuzz.partial_ratio(cand_norm, ref_norm)
    return fuzz.ratio(cand_norm, guess_norm)        # short titles: compare whole strings


def year_ok(ref, cand_year):
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", ref)]
    if not years or not cand_year:
        return True                                  # cannot judge
    return any(abs(y - cand_year) <= YEAR_SLACK for y in years)


def authors_ok(ref_norm, cand_authors):
    names = [normalize(a) for a in cand_authors[:3] if normalize(a)]
    if not names:
        return True
    padded = f" {ref_norm} "
    return any(f" {n} " in padded for n in names)    # at least one author appears


def verify_reference(ref, sources=None, judge=None):
    """Return {"ref", "status", "title_score", "match", "reason"} for one reference.

    sources: list of (name, search_function); default is the three real APIs.
    judge:   optional function (ref, candidate) -> bool for ambiguous title matches.
    """
    sources = scholar.SOURCES if sources is None else sources
    ref_norm = normalize(ref)
    guess = guess_title(ref)
    guess_norm = normalize(guess)

    best, best_score, best_good, failed = None, 0, False, 0
    for _name, search in sources:
        try:
            candidates = search(guess)
        except scholar.ScholarError:
            failed += 1
            continue
        for cand in candidates:
            score = title_score(cand["title"], ref_norm, guess_norm)
            # Same title found twice (e.g. a reprint): prefer the record whose year
            # and authors fit the reference.
            good = (score >= TITLE_SAME and year_ok(ref, cand["year"])
                    and authors_ok(ref_norm, cand["authors"]))
            if (good, score) > (best_good, best_score):
                best, best_score, best_good = cand, score, good
        if best_good:
            break                                    # good match: skip the other sources

    out = {"ref": ref, "status": "", "title_score": int(best_score),
           "match": best, "reason": ""}

    if best is None and failed == len(sources):
        out.update(status="unchecked", reason="all lookups failed (network or rate limit)")
    elif best is None or best_score < TITLE_MAYBE:
        out.update(status="not_found", reason="no paper with a similar title")
    elif best_score >= TITLE_SAME:
        problems = []
        if not year_ok(ref, best["year"]):
            problems.append(f"year differs (found {best['year']})")
        if not authors_ok(ref_norm, best["authors"]):
            problems.append("authors differ")
        if problems:
            out.update(status="mismatch", reason="; ".join(problems))
        else:
            out.update(status="exists")
    elif judge is not None:                          # ambiguous title
        if judge(ref, best):
            out.update(status="exists", reason="judge accepted the match")
        else:
            out.update(status="not_found", reason="judge rejected the closest match")
    else:
        out.update(status="mismatch", reason="title only partly matches")
    return out


def verify_references(references, sources=None, judge=None):
    return [verify_reference(r, sources, judge) for r in references]


def to_agent_result(checks):
    """Turn the per-reference checks into the shared AgentResult."""
    findings = []
    for c in checks:
        short = c["ref"][:90]
        url = c["match"]["url"] if c["match"] else ""
        if c["status"] == "not_found":
            findings.append(Finding(
                type="fake_reference", severity="high", confidence=0.8,
                evidence=f"{short} - {c['reason']}"))
        elif c["status"] == "mismatch":
            findings.append(Finding(
                type="reference_mismatch", severity="medium", confidence=0.6,
                evidence=f"{short} - {c['reason']}. Closest match: {url}"))
    summary = [{"ref": c["ref"][:60], "status": c["status"]} for c in checks]
    return AgentResult(
        findings=findings,
        needs_human=any(c["status"] == "not_found" for c in checks),
        raw_output=json.dumps(summary),
    )