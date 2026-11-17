# Created: 2026-11-17
# Created by: Lathika
# Purpose: Parse a paper PDF with PyMuPDF into a ParsedPaper JSON: title,
#          sections, equations (LaTeX source first, PDF fallback), cropped figures
#          with captions, tables, and references. Also provides the batch CLI.
"""PDF (+ optional arXiv LaTeX source) -> ParsedPaper JSON .

    python3 -m src.ingest.parse_pdf --csv data/papers.csv --pdf-dir data/papers/pdf \
        --src-dir data/papers/src --out data/papers/parsed [--ids t01 e02]

Layout handled: one or two columns, numbered / canonical headings, "Figure N:" and
"Table N:" captions, numbered or hanging-indent reference lists. Equations come from
the LaTeX source when it parses, otherwise from numbered display lines in the PDF
(a weak fallback: spot-check them).
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF < 1.24.3
    import fitz

from src.ingest.features import compute_features
from src.ingest.models import (
    EquationInfo, FigureInfo, ParsedPaper, TableInfo, to_json,
)

_LIG = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\u00ad": ""}

CANON = {
    "abstract", "introduction", "background", "related work", "related works", "preliminaries",
    "method", "methods", "methodology", "approach", "model", "problem setup", "problem formulation",
    "problem setting", "experiments", "experimental results", "experimental setup", "results",
    "discussion", "conclusion", "conclusions", "conclusion and future work", "limitations",
    "analysis", "evaluation", "references", "bibliography", "appendix", "acknowledgments",
    "acknowledgements", "proofs", "broader impact", "notation", "algorithm", "theory",
}
_NUM_HEAD = re.compile(r"^\d+(?:\.\d+){0,3}\.?\s+[A-Za-z]")
_LET_HEAD = re.compile(r"^[A-Z](?:\.\d+){0,2}\.?\s+[A-Z]")        # appendix headings
_ROM_HEAD = re.compile(r"^[IVX]{1,4}\.\s+[A-Z]")
# Matches "Figure 2: ...", "Fig. 3. ...", and IEEE-style "TABLE I TITLE" (no punctuation).
_CAPTION = re.compile(r"^\s*(Figure|Fig\.|FIGURE|FIG\.|Table|TABLE)\s*(\d+|[IVX]+)(?:\s*[:.|\u2013\u2014\-]\s*|\s+(?=[A-Z]{2,}))")
_EQNUM = re.compile(r"^(?P<body>.*\S)\s+\((?P<num>\d{1,3}(?:\.\d{1,2})?)\)\s*$")
_MATHY = re.compile(r"[=<>≤≥∈∑∫∏±×≈→∂∇]")


@dataclass
class Blk:
    page: int
    bbox: Tuple[float, float, float, float]
    lines: List[Tuple[float, str]]      # (x0, text) per line
    size: float
    bold: bool

    @property
    def text(self) -> str:
        return "\n".join(t for _, t in self.lines)

    @property
    def nlines(self) -> int:
        return len(self.lines)


def _clean(s: str) -> str:
    for k, v in _LIG.items():
        s = s.replace(k, v)
    return s


def _para(text: str) -> str:
    """Join wrapped lines into one paragraph (undo hyphenation)."""
    text = re.sub(r"-\n(?=[a-z])", "", text)
    return re.sub(r"\s*\n\s*", " ", text).strip()


# ------------------------------------------------------------- page -> ordered blocks
def _page_blocks(page, pno: int) -> List[Blk]:
    H = page.rect.height
    out: List[Blk] = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines, sizes, bolds = [], [], []
        for ln in b["lines"]:
            d = ln.get("dir", (1, 0))
            if (round(d[0]), round(d[1])) != (1, 0):      # rotated text (arXiv stamp)
                continue
            txt = "".join(sp["text"] for sp in ln["spans"])
            if not txt.strip():
                continue
            lines.append((ln["bbox"][0], _clean(txt).rstrip()))
            for sp in ln["spans"]:
                n = len(sp["text"].strip())
                if n:
                    sizes.append((sp["size"], n))
                    bolds.append((bool(sp["flags"] & 16) or "bold" in sp["font"].lower(), n))
        if not lines:
            continue
        tot = sum(n for _, n in sizes)
        size = sum(s * n for s, n in sizes) / tot
        bold = sum(n for f, n in bolds if f) / tot > 0.6
        blk = Blk(pno, tuple(b["bbox"]), lines, size, bold)
        t = blk.text.strip()
        if blk.bbox[1] > 0.94 * H and len(t) <= 40:                       # footers
            continue
        if re.fullmatch(r"\d{1,4}", t) and (blk.bbox[1] > 0.80 * H or blk.bbox[3] < 0.12 * H):
            continue                                                      # bare page numbers
        if pno > 0 and blk.bbox[3] < 0.06 * H and len(t) <= 80:           # running heads
            continue
        out.append(blk)
    return out


def _order(blocks: List[Blk], W: float) -> List[Blk]:
    """Reading order: top-to-bottom; left column before right between separators.

    A separator is any block that is wide or straddles the page midline (title, authors,
    full-width figures/captions, centred single-column text).
    """
    mid, tol = W / 2, 0.03 * W
    res: List[Blk] = []
    region: List[Blk] = []

    def flush():
        left = [b for b in region if b.bbox[2] <= mid + tol]
        right = [b for b in region if b not in left]
        res.extend(sorted(left, key=lambda b: (b.bbox[1], b.bbox[0])))
        res.extend(sorted(right, key=lambda b: (b.bbox[1], b.bbox[0])))
        region.clear()

    for b in sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0])):
        straddles = b.bbox[0] < mid - tol and b.bbox[2] > mid + tol
        if (b.bbox[2] - b.bbox[0]) > 0.62 * W or straddles:
            flush()
            res.append(b)
        else:
            region.append(b)
    flush()
    return res


def _body_size(blocks: List[Blk]) -> float:
    c = collections.Counter()
    for b in blocks:
        c[round(b.size * 2) / 2] += len(b.text)
    return c.most_common(1)[0][0] if c else 10.0


# ------------------------------------------------------------------------- headings
def _norm_head(t: str) -> str:
    t = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)*\.?|[IVX]+\.)\s+", "", t.strip())
    return t.strip(" :.").lower()


def _is_heading(b: Blk, body: float, after_refs: bool) -> bool:
    t = b.text.strip().replace("\n", " ")
    if b.nlines > 2 or not (3 <= len(t) <= 110) or len(t.split()) > 12:
        return False
    if _CAPTION.match(t):
        return False
    bigger, heavy = b.size >= body + 0.8, b.bold and b.size >= body - 0.4
    if not (bigger or heavy or b.bold or t.isupper()):
        return False
    if _norm_head(t) in CANON:
        return bigger or heavy or b.bold or t.isupper()
    if _NUM_HEAD.match(t) or _ROM_HEAD.match(t):
        return not t.endswith(".") or len(t.split()) <= 6
    return bool(after_refs and _LET_HEAD.match(t))


# ----------------------------------------------------------------------- graphics
def _graphics(page) -> List["fitz.Rect"]:
    rects = []
    for img in page.get_images(full=True):
        try:
            rects += [fitz.Rect(r) for r in page.get_image_rects(img[0])]
        except Exception:
            pass
    try:
        rects += [fitz.Rect(r) for r in page.cluster_drawings()]
    except Exception:
        pass
    return [r for r in rects if r.width > 30 or r.height > 30]      # keep thin axis/rule lines


def _column_range(cap: "fitz.Rect", W: float, pb: Optional[List[Blk]] = None) -> Tuple[float, float]:
    """Horizontal extent a figure/table under this caption may occupy."""
    if cap.width > 0.62 * W or (cap.x0 < W / 2 - 0.03 * W and cap.x1 > W / 2 + 0.03 * W):
        return 0.0, W
    cx = (cap.x0 + cap.x1) / 2
    paras = [b for b in (pb or []) if _paragraph_like(b) and b.bbox[0] <= cx <= b.bbox[2]]
    if paras:                                      # the text column that contains the caption
        return min(b.bbox[0] for b in paras) - 2, max(b.bbox[2] for b in paras) + 2
    return max(0.0, cap.x0 - 6), min(W, cap.x1 + 6)


def _figure_rect(page, cap: "fitz.Rect", gfx: List["fitz.Rect"], pb: List[Blk]) -> Tuple["fitz.Rect", bool]:
    W, H = page.rect.width, page.rect.height
    x0, x1 = _column_range(cap, W, pb)

    def in_col(r):
        ov = min(r.x1, x1) - max(r.x0, x0)
        return ov > 0.5 * r.width

    cand = [r for r in gfx if r.y1 <= cap.y0 + 3 and r.y0 >= cap.y0 - 620 and in_col(r)]
    if not cand:
        top = max(0.05 * H, cap.y0 - 230)
        return fitz.Rect(x0, top, x1, cap.y0 - 2), True
    cand.sort(key=lambda r: -r.y1)
    grp = fitz.Rect(cand[0])
    if cap.y0 - grp.y1 > 80:                       # graphic too far above: likely not this caption's
        top = max(0.05 * H, cap.y0 - 230)
        return fitz.Rect(x0, top, x1, cap.y0 - 2), True
    changed = True
    while changed:
        changed = False
        for r in cand:
            if r.y1 >= grp.y0 - 30 and r.y0 < grp.y0 and not (r in grp):
                grp |= r
                changed = True
            elif r.y0 < grp.y1 and r.y1 > grp.y0 and not grp.contains(r):
                grp |= r
                changed = True
    grp = fitz.Rect(max(x0, grp.x0 - 16), max(0, grp.y0 - 16), min(x1, grp.x1 + 16), min(cap.y0 - 1, grp.y1 + 16))
    for b in pb:                                   # absorb axis labels / titles / legends touching the crop
        br = fitz.Rect(b.bbox)
        if not _paragraph_like(b) and not _CAPTION.match(b.text.strip()) and not (br & grp).is_empty:
            grp |= br
    grp = fitz.Rect(max(x0, grp.x0), max(0, grp.y0), min(x1, grp.x1), min(cap.y0 - 1, grp.y1))
    return grp, False


# ------------------------------------------------------------------------- tables
def _paragraph_like(b: Blk) -> bool:
    return len(re.findall(r"[A-Za-z]{3,}", b.text)) >= 12


def _table_rows(page, region: "fitz.Rect") -> Tuple[str, Optional["fitz.Rect"]]:
    """Rows of text inside region: words grouped by baseline, cells split on wide gaps."""
    words = [w for w in page.get_text("words", clip=region) if w[4].strip()]
    if not words:
        return "", None
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: List[List[tuple]] = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if rows and abs(yc - sum((v[1] + v[3]) / 2 for v in rows[-1]) / len(rows[-1])) <= 0.6 * (w[3] - w[1]):
            rows[-1].append(w)
        else:
            rows.append([w])
    lines = []
    for r in rows:
        r.sort(key=lambda w: w[0])
        cells, cur, prev = [], [r[0][4]], r[0]
        for w in r[1:]:
            if w[0] - prev[2] > 7:
                cells.append(" ".join(cur))
                cur = []
            cur.append(w[4])
            prev = w
        cells.append(" ".join(cur))
        lines.append(" | ".join(cells))
    bbox = fitz.Rect(min(w[0] for w in words), min(w[1] for w in words), max(w[2] for w in words), max(w[3] for w in words))
    return "\n".join(lines)[:4000], bbox


def _table_text(page, cap: "fitz.Rect", pb: List[Blk], body: float) -> Tuple[str, Optional["fitz.Rect"]]:
    W, H = page.rect.width, page.rect.height
    x0, x1 = _column_range(cap, W, pb)
    paras = [fitz.Rect(b.bbox) for b in pb
             if (_paragraph_like(b) or _is_heading(b, body, True) or _CAPTION.match(b.text.strip()))
             and fitz.Rect(b.bbox) != cap and b.bbox[0] < x1 and b.bbox[2] > x0]
    below_end = min([r.y0 for r in paras if r.y0 >= cap.y1 - 1] + [0.94 * H])
    above_start = max([r.y1 for r in paras if r.y1 <= cap.y0 + 1] + [0.05 * H])
    below = _table_rows(page, fitz.Rect(x0, cap.y1 + 1, x1, below_end))
    if below[0].count("\n") >= 1:
        return below
    above = _table_rows(page, fitz.Rect(x0, above_start, x1, cap.y0 - 1))
    return above if above[0].count("\n") >= 1 else below


# --------------------------------------------------------------------- references
def split_references(blocks: List[Blk]) -> List[str]:
    """Split the lines of a reference section into entries."""
    lines = [(x0, t.strip()) for b in blocks for x0, t in b.lines if t.strip()]
    starts = [i for i, (_, t) in enumerate(lines) if re.match(r"^\[\d{1,3}\]", t)]
    entries: List[str] = []
    if len(starts) >= 2:                                           # [1] style
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else len(lines)
            entries.append(" ".join(t for _, t in lines[s:e]))
    else:
        cur: List[str] = []
        prev_page = None
        for b in blocks:
            prev_x = None
            for x0, t in b.lines:
                t = t.strip()
                if not t:
                    continue
                new = prev_x is None or x0 < prev_x - 3          # block start or hanging outdent
                if prev_x is None and cur:
                    last = cur[-1]
                    ended = last.endswith((".", ")", "]")) or last[-1:].isdigit()
                    if t[0].islower() or last.endswith((",", "-")) or (not ended and (b.page != prev_page or len(t) < 40)):
                        new = False                                   # fragment continued from prior column/page
                if new and cur:
                    entries.append(" ".join(cur))
                    cur = []
                cur.append(t)
                prev_x = x0
            prev_page = b.page
        if cur:
            entries.append(" ".join(cur))
        merged: List[str] = []                                       # bare "[n]" labels glued to next entry
        carry = ""
        for e in entries:
            if re.fullmatch(r"\[?\d{1,3}\]?\.?", e.strip()):
                carry = e.strip() + " "
                continue
            merged.append(carry + e)
            carry = ""
        entries = merged
    out = []
    for e in entries:
        e = re.sub(r"-\s(?=[a-z])", "", e)                          # "optimi- zation"
        e = re.sub(r"\s+", " ", e).strip()
        if len(e) >= 25:
            out.append(e)
    return out[:400]


# ------------------------------------------------------------------- PDF equations
def _mathy(s: str) -> bool:
    long_words = re.findall(r"[A-Za-z]{4,}", s)
    return len(s) <= 300 and len(long_words) <= 4 and (bool(_MATHY.search(s)) or len(re.findall(r"[+\-/^_]", s)) >= 2)


_NUMTAIL = re.compile(r"\(((?:[A-Z]\.)?\d{1,3}(?:\.\d{1,2})?)\)\s*$")


def _eq_candidates(pb: List[Blk]) -> List[Tuple[Blk, str]]:
    """Blocks that end with a right-margin equation number such as '(3)'."""
    edges = [b.bbox[2] for b in pb if _paragraph_like(b)]
    out = []
    for b in pb:
        m = _NUMTAIL.search(b.text.strip())
        if not m or not edges:
            continue
        if min(abs(e - b.bbox[2]) for e in edges) > 8:        # must sit at a column's right edge
            continue
        rest = b.text.strip()[: m.start()]
        if len(re.findall(r"[A-Za-z]{4,}", rest)) > 3:        # prose such as "... see Eq. (3)"
            continue
        out.append((b, m.group(1)))
    return out


def _band_blocks(pb: List[Blk], nb: Blk, body: float) -> List[Blk]:
    """Equation fragments that share the number block's vertical band and column."""
    cols = [b for b in pb if _paragraph_like(b) and abs(b.bbox[2] - nb.bbox[2]) <= 8]
    cx0 = min((b.bbox[0] for b in cols), default=0.0)
    y0, y1 = nb.bbox[1] - 9, nb.bbox[3] + 9
    frags = [b for b in pb if b is not nb and y0 <= (b.bbox[1] + b.bbox[3]) / 2 <= y1
             and b.bbox[0] >= cx0 - 6 and b.bbox[2] <= nb.bbox[2] + 6
             and len(re.findall(r"[A-Za-z]{3,}", b.text)) < 3 and not _CAPTION.match(b.text.strip())
             and not (b.size >= body + 0.8) and not (b.bold and b.size >= body - 0.4)]
    frags.append(nb)
    frags.sort(key=lambda b: (round((b.bbox[1] + b.bbox[3]) / 2 / 8), b.bbox[0]))
    return frags


def _frag_text(frags: List[Blk], nb: Blk) -> str:
    parts = []
    for b in frags:
        t = b.text.replace("\n", " ")
        parts.append(_NUMTAIL.sub("", t).strip() if b is nb else t.strip())
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()


# ------------------------------------------------------------------------- driver
def parse_paper(
    pdf_path: str,
    paper_id: str,
    out_dir: str,
    title: str = "",
    category: str = "",
    arxiv_id: str = "",
    src_archive: Optional[str] = None,
    save_figures: bool = True,
    rel_to: Optional[str] = None,
) -> ParsedPaper:
    doc = fitz.open(pdf_path)
    out_dir_p = Path(out_dir)
    fig_dir = out_dir_p / paper_id
    rel_root = Path(rel_to) if rel_to else Path.cwd()

    # 1. blocks in reading order
    per_page: List[List[Blk]] = []
    pages = []
    for pno, page in enumerate(doc):
        pages.append(page)
        per_page.append(_order(_page_blocks(page, pno), page.rect.width))
    allb = [b for pb in per_page for b in pb]
    body = _body_size(allb)

    # 2. captions -> figures / tables (+ exclusion zones)
    figs: List[FigureInfo] = []
    tabs: List[TableInfo] = []
    zones: Dict[int, List["fitz.Rect"]] = collections.defaultdict(list)
    caption_blocks = set()
    for pno, pb in enumerate(per_page):
        page = pages[pno]
        gfx = None
        for b in pb:
            m = _CAPTION.match(b.text.replace("\n", " "))
            if not m or len(b.text) < 12:
                continue
            kind = "Figure" if m.group(1).lower().startswith("fig") else "Table"
            num = m.group(2)
            caption = _para(b.text)
            cap_rect = fitz.Rect(b.bbox)
            caption_blocks.add(id(b))
            if kind == "Figure":
                if any(f.id == f"Figure {num}" for f in figs):
                    continue
                gfx = gfx if gfx is not None else _graphics(page)
                rect, fallback = _figure_rect(page, cap_rect, gfx, pb)
                path = ""
                if save_figures and rect.width > 20 and rect.height > 20:
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    fp = fig_dir / f"fig_{num}.png"
                    page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect).save(str(fp))
                    try:
                        path = str(fp.resolve().relative_to(rel_root.resolve()))
                    except ValueError:
                        path = str(fp)
                zones[pno].append(rect)
                figs.append(FigureInfo(id=f"Figure {num}", path=path, page=pno + 1, caption=caption, fallback_crop=fallback))
            else:
                if any(t.id == f"Table {num}" for t in tabs):
                    continue
                text, trect = _table_text(page, cap_rect, pb, body)
                if trect is not None:
                    zones[pno].append(trect)
                tabs.append(TableInfo(id=f"Table {num}", page=pno + 1, caption=caption, text=text))

    # 3. title, headings, sections
    first = per_page[0] if per_page else []
    if not title and first:
        top = [b for b in first if b.bbox[1] < 0.4 * pages[0].rect.height and len(b.text.strip()) >= 8]
        if top:
            title = _para(max(top, key=lambda b: b.size).text)

    sections: Dict[str, List[str]] = collections.OrderedDict()
    ref_blocks: List[Blk] = []
    flat: List[Tuple[str, str, bool]] = []
    paragraphs: List[str] = []
    cur, after_refs, in_refs = "", False, False
    front: List[str] = []

    def add_section(name):
        base, k = name, 2
        while name in sections:
            name, k = f"{base} ({k})", k + 1
        sections[name] = []
        return name

    eq_frag_ids, cand_info = set(), {}          # PDF-fallback equations (numbered display lines)
    for pb in per_page:
        for nb, num in _eq_candidates(pb):
            frags = _band_blocks(pb, nb, body)
            eq_frag_ids.update(id(f) for f in frags)
            cand_info[id(nb)] = (num, _frag_text(frags, nb))
    pdf_eq_raw: List[Tuple[str, int, str, str]] = []   # (section, flat position, number, text)

    for pno, pb in enumerate(per_page):
        for b in pb:
            if id(b) in caption_blocks:
                continue
            if id(b) in eq_frag_ids:
                if id(b) in cand_info:
                    pdf_eq_raw.append((cur, len(flat), *cand_info[id(b)]))
                continue
            ctr = fitz.Rect(b.bbox)
            if any(z.contains(ctr) or (z & ctr).get_area() > 0.6 * ctr.get_area() for z in zones.get(pno, [])):
                continue
            t = b.text.strip()
            if pno == 0 and title and _para(t) == _para(title):
                continue
            head_text = t.replace("\n", " ")
            # run-in "Abstract. text" / "Abstract—text"
            mabs = re.match(r"^(abstract)\s*[\.\-–—:]\s*(\S.+)$", head_text, re.I | re.S)
            if mabs and "Abstract" not in sections:
                cur = add_section("Abstract")
                sections[cur].append(_para(t)[len(mabs.group(1)):].lstrip(" .:-–—"))
                paragraphs.append(sections[cur][-1])
                continue
            if _is_heading(b, body, after_refs):
                cur = add_section(head_text.strip(" :"))
                n = _norm_head(head_text)
                in_refs = n in ("references", "bibliography")
                if in_refs:
                    after_refs = True
                flat.append((cur, head_text, False))
                continue
            if in_refs:
                ref_blocks.append(b)
                continue
            if len(t) < 60 and not re.search(r"[A-Za-z]{3,}", t):      # stray math / axis fragments
                continue
            if not cur:
                if pno == 0:
                    front.append(_para(t))
                continue
            p = _para(t)
            sections[cur].append(p)
            paragraphs.append(p)
            for _, ln in b.lines:
                flat.append((cur, ln.strip(), True))

    if "Abstract" not in sections and front:                 # abstract without a heading
        longest = max(front, key=len)
        if len(longest) >= 300:
            sections = collections.OrderedDict([("Abstract", [longest])] + list(sections.items()))

    # 4. citing paragraphs for B4
    def citing(label: str) -> List[str]:
        kind, num = label.split()
        pat = (r"\b(?:Figure|Fig\.|Figs\.)\s*" if kind == "Figure" else r"\bTables?\s*") + re.escape(num) + r"\b"
        hits = [p[:700] for p in paragraphs if re.search(pat, p, re.I) and len(p) >= 40]
        return hits[:3]

    for f in figs:
        f.citing_paragraphs = citing(f.id)
    for t in tabs:
        t.citing_paragraphs = citing(t.id)

    # 5. equations: LaTeX source first, PDF heuristics as fallback
    eqs: List[EquationInfo] = []
    eq_source = "pdf"
    if src_archive and Path(src_archive).exists():
        try:
            from src.ingest.latex_eq import equations_from_archive
            eqs = equations_from_archive(src_archive)
        except Exception as exc:                               # corrupt or odd archive
            print(f"[{paper_id}] LaTeX source skipped: {exc}", file=sys.stderr)
    if eqs:
        eq_source = "tex"
    else:
        seen = set()
        for sec, pos, num, text in pdf_eq_raw:
            if num in seen or not _mathy(text):
                continue
            seen.add(num)
            before = " ".join(t for _, t, ok in flat[max(0, pos - 8): pos] if ok)[-450:]
            after = " ".join(t for _, t, ok in flat[pos: pos + 6] if ok)[:300]
            eqs.append(EquationInfo(index=len(eqs), number=num, latex=text, section=sec,
                                    before=before, after=after, source="pdf"))

    refs = split_references(ref_blocks)

    paper = ParsedPaper(
        id=paper_id,
        title=title,
        sections={k: "\n\n".join(v) for k, v in sections.items() if v},
        equations=[f"({e.number}) {e.latex}" if e.number else e.latex for e in eqs],
        figures=[f.path for f in figs if f.path],
        tables=[t.caption + (f"\n{t.text}" if t.text else "") for t in tabs],
        references=refs,
        arxiv_id=arxiv_id,
        category=category,
        n_pages=len(doc),
        equation_source=eq_source,
        equation_meta=eqs,
        figure_meta=figs,
        table_meta=tabs,
    )
    paper.profile_features = compute_features(paper)
    doc.close()
    return paper

def load_metadata(csv_path):
    """Optional labels from papers.csv: {id: row}. No file means no labels."""
    if not csv_path or not Path(csv_path).exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def parse_folder(pdf_dir, out_dir, src_dir=None, csv_path=None, ids=None):
    """Parse every PDF in pdf_dir. The file name without .pdf is the paper id.

    If src_dir/<id>.tar.gz exists it is used for equations (LaTeX source).
    Writes out_dir/<id>.json for each paper and returns the list of ParsedPaper.
    """
    meta = load_metadata(csv_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    papers = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        pid = pdf.stem
        if ids and pid not in ids:
            continue
        row = meta.get(pid, {})                       # category/title are optional
        src = Path(src_dir) / f"{pid}.tar.gz" if src_dir else None
        paper = parse_paper(
            str(pdf), pid, out_dir,
            title=row.get("title", ""), category=row.get("category", ""),
            arxiv_id=row.get("arxiv_id", ""),
            src_archive=str(src) if src and src.exists() else None,
        )
        (Path(out_dir) / f"{pid}.json").write_text(to_json(paper), encoding="utf-8")
        pf = paper.profile_features
        print(f"[{pid}] sec={int(pf['n_sections']):2d} eq={int(pf['n_equations']):3d}"
              f"({paper.equation_source}) fig={int(pf['n_figures']):2d} tab={int(pf['n_tables']):2d} "
              f"ref={int(pf['n_references']):3d} tok~{int(pf['n_tokens_est'])}")
        papers.append(paper)
    return papers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="data/papers/pdf")      # the input folder
    ap.add_argument("--src-dir", default="data/papers/src")      # optional LaTeX sources
    ap.add_argument("--out", default="data/papers/parsed")
    ap.add_argument("--csv", default="data/papers.csv")          # optional labels
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args(argv)

    papers = parse_folder(args.pdf_dir, args.out, args.src_dir, args.csv, args.ids)
    if not papers:
        print(f"no PDFs found in {args.pdf_dir}")
        return 1
    print(f"parsed {len(papers)} paper(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())