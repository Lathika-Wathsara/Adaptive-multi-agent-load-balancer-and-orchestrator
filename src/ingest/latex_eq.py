# Created: 2026-11-17
# Created by: Lathika
# Purpose: Extract display equations from an arXiv LaTeX source archive.
#          Reads .tex files safely in memory, expands \input/\include, and returns
#          each equation with its number, label, section and surrounding text.


import gzip
import io
import re
import tarfile
from typing import Dict, List, Optional, Tuple

from src.ingest.models import EquationInfo

MAX_MEMBER_BYTES = 5_000_000
NUMBERED_ENVS = {"equation", "align", "eqnarray", "gather", "multline", "flalign", "alignat"}
ROW_ENVS = {"align", "eqnarray", "gather", "flalign", "alignat"}   # one number per row

_DISPLAY = re.compile(
    r"\\begin\{(?P<env>equation|align|eqnarray|gather|multline|flalign|alignat|displaymath)"
    r"(?P<star>\*?)\}(?P<body>.*?)\\end\{(?P=env)(?P=star)\}"
    r"|(?<!\\)\\\[(?P<br>.*?)(?<!\\)\\\]"
    r"|(?<!\\)\$\$(?P<dd>.*?)(?<!\\)\$\$",
    re.S,
)
_SECTION = re.compile(
    r"\\(?P<kind>section|appendix)(?P<star>\*?)\s*(?:\[[^\]]*\])?\s*"
    r"(?:\{(?P<title>[^{}]*(?:\{[^{}]*\}[^{}]*)*)\})?"
)
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
_LABEL = re.compile(r"\\label\{([^}]*)\}")
_TAG = re.compile(r"\\tag\*?\{([^}]*)\}")
_NONUM = re.compile(r"\\(?:nonumber|notag)\b")


# ----------------------------------------------------------------------------- IO
def read_tex_files(archive_path: str) -> Dict[str, str]:
    """{member name: text} for .tex/.bbl files in a .tar(.gz) or single .gz source."""
    with open(archive_path, "rb") as fh:
        raw = fh.read()
    files: Dict[str, str] = {}
    try:
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:*")
    except tarfile.TarError:
        tf = None
    if tf is not None:
        for m in tf.getmembers():
            name = m.name
            while name.startswith("./"):
                name = name[2:]
            if not m.isfile() or m.size > MAX_MEMBER_BYTES:
                continue
            if name.startswith("/") or ".." in name.split("/"):
                continue
            if not name.lower().endswith((".tex", ".bbl")):
                continue
            fh2 = tf.extractfile(m)
            if fh2 is not None:
                files[name] = fh2.read().decode("utf-8", "ignore")
        return files
    try:
        data = gzip.decompress(raw)
    except OSError:
        data = raw
    text = data.decode("utf-8", "ignore")
    if "\\begin{document}" in text:
        files["main.tex"] = text
    return files


def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def _find_member(files: Dict[str, str], ref: str) -> Optional[str]:
    ref = ref.strip()
    while ref.startswith("./"):
        ref = ref[2:]
    for cand in (ref, ref + ".tex"):
        if cand in files:
            return cand
    base = ref.split("/")[-1]
    for name in files:
        if name == base or name == base + ".tex" or name.endswith("/" + base) or name.endswith("/" + base + ".tex"):
            return name
    return None


def flatten(files: Dict[str, str]) -> str:
    """Main .tex with \\input/\\include expanded and comments removed ('' if none)."""
    mains = [n for n, t in files.items() if n.endswith(".tex") and "\\begin{document}" in t]
    if not mains:
        return ""
    main = max(mains, key=lambda n: len(files[n]))

    def expand(name: str, depth: int, seen: frozenset) -> str:
        text = _strip_comments(files[name])
        if depth >= 6:
            return text

        def repl(m):
            target = _find_member(files, m.group(1))
            if target and target not in seen:
                return expand(target, depth + 1, seen | {target})
            return ""

        return _INPUT.sub(repl, text)

    return expand(main, 0, frozenset({main}))


# ------------------------------------------------------------------------ helpers
def _split_rows(body: str) -> List[str]:
    """Split on top-level \\\\ (not inside nested environments or braces)."""
    rows, start, i, env, brace, n = [], 0, 0, 0, 0, len(body)
    while i < n:
        c = body[i]
        if c == "\\":
            if body.startswith("\\\\", i):
                if env == 0 and brace == 0:
                    rows.append(body[start:i])
                    i += 2
                    m = re.match(r"\s*\[\s*-?[\d.]+\s*(?:pt|em|ex|mm|cm|in)\s*\]", body[i:])
                    if m:
                        i += m.end()
                    start = i
                    continue
                i += 2
                continue
            if body.startswith("\\begin{", i):
                env += 1
            elif body.startswith("\\end{", i):
                env -= 1
            i += 2
            continue
        if c == "{":
            brace += 1
        elif c == "}":
            brace -= 1
        i += 1
    rows.append(body[start:])
    return rows


def _plain(text: str) -> str:
    """Light cleanup of surrounding prose so a model can read it."""
    text = re.sub(r"\\(?:emph|textit|textbf|textsc|mathit)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{[^}]*\}", "[cite]", text)
    text = re.sub(r"\\(?:eq|auto|c|C)?ref\{[^}]*\}", "(ref)", text)
    text = re.sub(r"\\(?:begin|end)\{(?:document|abstract)\}", " ", text)
    text = re.sub(r"\\(?:sub)*section\*?\{([^{}]*)\}", r"\1. ", text)
    text = _LABEL.sub("", text)
    text = re.sub(r"\\([%&_#$])", r"\1", text).replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def _clean_row(row: str, family_env: bool) -> Tuple[str, Optional[str], Optional[str], bool]:
    """-> (latex, label, custom_tag, suppressed_number)"""
    label = None
    m = _LABEL.search(row)
    if m:
        label = m.group(1)
    tag = None
    m = _TAG.search(row)
    if m:
        tag = m.group(1)
    suppressed = bool(_NONUM.search(row))
    row = _LABEL.sub("", row)
    row = _TAG.sub("", row)
    row = _NONUM.sub("", row)
    if family_env and "\\begin{" not in row:
        row = row.replace("&", " ")
    return re.sub(r"\s+", " ", row).strip(), label, tag, suppressed


# --------------------------------------------------------------------------- main
def extract_equations(tex: str) -> List[EquationInfo]:
    """Display equations in order, with number, label, section and context."""
    numberwithin = bool(re.search(r"\\numberwithin\s*\{equation\}\s*\{section\}", tex))
    body_start = tex.find("\\begin{document}")
    base = body_start if body_start >= 0 else 0

    # structural events: sections and \appendix
    events = []
    for m in _SECTION.finditer(tex, base):
        if m.group("kind") == "appendix":
            events.append((m.start(), "appendix", ""))
        elif not m.group("star"):
            events.append((m.start(), "section", _plain(m.group("title") or "")))

    out: List[EquationInfo] = []
    counter = 0               # global equation counter
    sec_counter = 0           # section counter (for numberwithin)
    in_appendix = False
    sec_title = ""
    sec_eq_counter = 0
    ev_i = 0

    for m in _DISPLAY.finditer(tex, base):
        # advance structure state up to this equation
        while ev_i < len(events) and events[ev_i][0] < m.start():
            _, kind, title = events[ev_i]
            if kind == "appendix":
                in_appendix, sec_counter = True, 0
                sec_eq_counter = 0
            else:
                sec_counter += 1
                sec_title = title
                sec_eq_counter = 0
            ev_i += 1

        env = m.group("env") or "display"
        star = bool(m.group("star"))
        raw = m.group("body") if m.group("env") else (m.group("br") if m.group("br") is not None else m.group("dd"))
        raw = re.sub(r"^\s*\{\d+\}", "", raw) if env == "alignat" else raw

        numbered_env = env in NUMBERED_ENVS and not star
        if env in ROW_ENVS:
            rows = [r for r in _split_rows(raw) if r.strip()]
        else:
            rows = [raw]
            if env == "multline":
                rows = [" \\\\ ".join(r.strip() for r in _split_rows(raw) if r.strip())]

        before = _plain(tex[max(base, m.start() - 600): m.start()])[-450:]
        after = _plain(tex[m.end(): m.end() + 400])[:300]

        for row in rows:
            latex, label, tag, suppressed = _clean_row(row, env in ROW_ENVS or env == "multline")
            if not latex:
                continue
            number = None
            if tag is not None:
                number = tag
            elif numbered_env and not suppressed:
                if numberwithin:
                    sec_eq_counter += 1
                    prefix = chr(ord("A") + sec_counter - 1) if in_appendix and 1 <= sec_counter <= 26 else str(sec_counter)
                    number = f"{prefix}.{sec_eq_counter}"
                else:
                    counter += 1
                    number = str(counter)
            out.append(
                EquationInfo(
                    index=len(out), number=number, latex=latex, label=label,
                    section=sec_title, before=before, after=after, source="tex",
                )
            )
    return out


def equations_from_archive(archive_path: str) -> List[EquationInfo]:
    tex = flatten(read_tex_files(archive_path))
    return extract_equations(tex) if tex else []
