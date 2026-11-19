# File: src/tools/notation.py
# Created: 2026-11-18
# Created by: Lathika
# Purpose: Find where the symbols of an equation are defined elsewhere in the paper.

import re

_CUE = re.compile(r"\b(?:denote[sd]?|represent[s]?|stand[s]? for|defined?|is the|are the|let)\b"
                  r"|∈|\\in\b", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_STYLE = re.compile(r"\\(?:mathbf|mathrm|mathcal|mathbb|boldsymbol|text|left|right|frac|sum|prod|log"
                    r"|ln|exp|sin|cos|tan|det|ge|le|geq|leq|cdot|times|sqrt|approx|neq|infty"
                    r"|forall|to|in|operatorname)(?![A-Za-z])")
_NUMBER_REF = re.compile(r"\((\d+(?:\.\d+)?[a-z]?)\)")
_NOT_SYMBOLS = {"frac", "sum", "prod", "log", "ln", "exp", "sin", "cos", "tan", "det", "tr",
                "Re", "Im", "max", "min", "ge", "le", "geq", "leq", "cdot", "times", "left",
                "right", "sqrt", "mathrm", "mathbf", "mathcal", "text", "operatorname", "approx",
                "neq", "infty", "in", "forall", "to", "H", "T"}
_TOO_COMMON = {"a", "A", "I"}              # also ordinary English words
MAX_SENTENCE = 250                         # characters kept per definition
MAX_DEFINITIONS = 5
MAX_REFERENCED = 3


def paper_sentences(paper):
    """Every sentence of the paper's sections that contains a definition cue."""
    out = []
    for text in paper.sections.values():
        out.extend(s.strip() for s in _SENTENCE_END.split(text) if _CUE.search(s))
    return out


def clean(text):
    """Text without LaTeX styling, braces, dollars and backslashes, for matching symbols."""
    text = _STYLE.sub("", text)
    text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)           # R_{u} -> R_u
    text = re.sub(r"\{([A-Za-z0-9])\}(?=_)", r"\1", text)    # {R}_u -> R_u
    return re.sub(r"[${}\\^]", " ", text)


def symbols(latex):
    """The variable-like tokens of an equation, e.g. {"d", "a", "R_u", "alpha"}."""
    tokens = re.findall(r"[A-Za-z]+(?:_[A-Za-z0-9]+)?|[A-Za-z]", clean(latex))
    return {t for t in tokens if t not in _NOT_SYMBOLS and t not in _TOO_COMMON}


def definitions_for(latex, sentences, skip="", limit=MAX_DEFINITIONS):
    """Up to `limit` sentences that define the most symbols of this equation. Sentences
    already shown as the equation's own surrounding text (skip) are left out."""
    wanted = symbols(latex)
    scored = []
    for sentence in sentences:
        if sentence in skip:
            continue
        words = set(re.findall(r"[A-Za-z]+(?:_[A-Za-z0-9]+)?", clean(sentence)))
        hits = len(wanted & words)
        if hits:
            scored.append((-hits, len(sentence), sentence))
    scored.sort()
    return [s[:MAX_SENTENCE] for _, _, s in scored[:limit]]


def referenced_equations(text, equations, own_number, previous_number=None, limit=MAX_REFERENCED):
    """Equations mentioned as "(4)" in the given text: [(number, latex)]. The equation
    itself and the previous one are skipped (they are shown anyway)."""
    by_number = {eq.number: eq.latex for eq in equations if eq.number}
    out = []
    for number in _NUMBER_REF.findall(text):
        if number in by_number and number not in (own_number, previous_number) \
                and number not in [n for n, _ in out]:
            out.append((number, by_number[number]))
    return out[:limit]