"""B1 tests: ingestion. No model is called here, so MockLLM only appears in the hand-off test.

Planted PDFs are generated with PyMuPDF at test time (nothing to download, no PDFs in git).
Ground truth is known by construction: 2 numbered equations, 1 figure, 1 table, 3 references.
The "clean control" PDF has prose only and must yield no equations, figures, tables or references.
"""
import asyncio
import csv
import gzip
import io
import tarfile
from pathlib import Path

import pytest

try:
    import pymupdf
except ImportError:                      # PyMuPDF < 1.24.3 only ships the old module name
    pymupdf = pytest.importorskip("fitz")

from src.common.mock_llm import MockLLM
from src.common.schemas import PaperDoc, TaskRequest
from src.ingest.features import compute_features
from src.ingest.latex_eq import extract_equations, equations_from_archive, read_tex_files
from src.ingest.models import ParsedPaper, from_json, to_json
from src.ingest.parse_pdf import main as parse_main
from src.ingest.parse_pdf import parse_paper

FILLER = ("the planted study measures regret and variance across many seeds while every estimator "
          "uses the same context features and the same confidence width for all arms ")


def _para(page, rect, text):
    left = page.insert_textbox(rect, text, fontsize=10, fontname="helv", align=3)
    assert left >= 0, "fixture text does not fit"


def _eq(page, y, body, number):
    """Display equation: fragments in the middle, number flush with the paragraph right edge (540)."""
    page.insert_text((230, y), body, fontsize=10, fontname="helv")
    w = pymupdf.get_text_length(f"({number})", fontname="helv", fontsize=10)
    page.insert_text((540 - w, y), f"({number})", fontsize=10, fontname="helv")


def make_planted_pdf(path, ref_style="numbered"):
    doc = pymupdf.open()
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((72, 80), "Planted Regret Study", fontsize=18, fontname="hebo")
    p1.insert_text((72, 130), "Abstract", fontsize=11, fontname="hebo")
    _para(p1, pymupdf.Rect(72, 138, 540, 215), "We plant two equations, one figure and one table. " + FILLER * 2)
    p1.insert_text((72, 240), "1 Introduction", fontsize=12, fontname="hebo")
    _para(p1, pymupdf.Rect(72, 248, 540, 340),
          "Figure 1 shows regret on synthetic data and Table 1 lists the final scores. "
          "As shown in Eq. (1) the estimate is stable. " + FILLER * 2)
    p1.insert_text((72, 365), "2 Method", fontsize=12, fontname="hebo")
    _para(p1, pymupdf.Rect(72, 373, 540, 440), "The estimator is defined next. " + FILLER)
    _eq(p1, 462, "E = m c^2", 1)
    _para(p1, pymupdf.Rect(72, 485, 540, 550), "Then the width is " + FILLER)
    _eq(p1, 572, "w = a + b / c", 2)
    p1.draw_rect(pymupdf.Rect(150, 600, 460, 700), color=(0, 0, 0), fill=(0.6, 0.7, 0.9))
    p1.draw_line((150, 650), (460, 650), color=(0.8, 0, 0))
    p1.insert_text((150, 718), "Figure 1: Planted regret curve on synthetic data.", fontsize=9, fontname="helv")

    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((72, 90), "Table 1: Planted final scores.", fontsize=9, fontname="helv")
    for i, row in enumerate([("Method", "Score", "Time"), ("LinUCB", "12.1", "3.4"), ("Greedy", "30.5", "1.2")]):
        for x, cell in zip((90, 220, 330), row):
            p2.insert_text((x, 110 + 14 * i), cell, fontsize=9, fontname="helv")
    p2.insert_text((72, 190), "3 Conclusion", fontsize=12, fontname="hebo")
    _para(p2, pymupdf.Rect(72, 198, 540, 270), "We conclude. " + FILLER)
    p2.insert_text((72, 300), "References", fontsize=12, fontname="hebo")
    if ref_style == "numbered":
        refs = [
            "[1] L. Li, W. Chu, J. Langford, and R. Schapire. A contextual-bandit approach to news",
            "article recommendation. In WWW, 2010.",
            "[2] D. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv:1412.6980,",
            "2014.",
            "[3] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition.",
            "In CVPR, 2016.",
        ]
        y = 320
        for ln in refs:
            p2.insert_text((72, y), ln, fontsize=9, fontname="helv")
            y += 11
    else:   # author-year, hanging indent, no numbers
        refs = [
            ("Li, L., Chu, W., Langford, J., and Schapire, R. A contextual-bandit approach to", "news article recommendation. In WWW, 2010."),
            ("Kingma, D. and Ba, J. Adam: A method for stochastic optimization. arXiv, 2014.", ""),
            ("He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image", "recognition. In CVPR, 2016."),
        ]
        y = 320
        for first, cont in refs:
            p2.insert_text((72, y), first, fontsize=9, fontname="helv")
            y += 11
            if cont:
                p2.insert_text((86, y), cont, fontsize=9, fontname="helv")
                y += 11
            y += 4
    doc.save(str(path))
    doc.close()


def make_control_pdf(path):
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((72, 80), "A Prose Only Note", fontsize=18, fontname="hebo")
    p.insert_text((72, 130), "1 Introduction", fontsize=12, fontname="hebo")
    _para(p, pymupdf.Rect(72, 138, 540, 230), FILLER * 3)
    p.insert_text((72, 260), "2 Discussion", fontsize=12, fontname="hebo")
    _para(p, pymupdf.Rect(72, 268, 540, 360), FILLER * 3)
    doc.save(str(path))
    doc.close()


@pytest.fixture(scope="module")
def planted_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("planted")


@pytest.fixture(scope="module")
def planted(planted_dir):
    make_planted_pdf(planted_dir / "p.pdf")
    return parse_paper(str(planted_dir / "p.pdf"), "p1", str(planted_dir / "out"), category="theory",
                       rel_to=str(planted_dir))


# --------------------------------------------------------------------------- PDF
def test_title_and_sections(planted):
    assert planted.title == "Planted Regret Study"
    assert list(planted.sections) == ["Abstract", "1 Introduction", "2 Method", "3 Conclusion"]
    assert "plant two equations" in planted.sections["Abstract"]
    assert planted.n_pages == 2


def test_planted_equations_found_without_latex_source(planted):
    assert planted.equation_source == "pdf"
    nums = [e.number for e in planted.equation_meta]
    assert nums == ["1", "2"]                    # the "Eq. (1)" mention inside prose is not an equation
    assert "m c^2" in planted.equation_meta[0].latex
    assert planted.equation_meta[0].section == "2 Method"
    assert planted.equations[0].startswith("(1) ")
    assert "estimator is defined" in planted.equation_meta[0].before


def test_figure_cropped_with_caption_and_citing_paragraph(planted, planted_dir):
    assert len(planted.figure_meta) == 1 and len(planted.figures) == 1
    fig = planted.figure_meta[0]
    assert fig.id == "Figure 1" and fig.caption.startswith("Figure 1: Planted regret")
    assert not fig.fallback_crop
    assert planted.figures == [fig.path] and not Path(fig.path).is_absolute()   # portable, repo-relative
    img = pymupdf.Pixmap(str(planted_dir / fig.path))
    assert img.width >= 400 and img.height >= 120                               # the 310x100 pt box at 2x
    assert fig.citing_paragraphs and "Figure 1 shows regret" in fig.citing_paragraphs[0]


def test_table_rows_and_caption(planted):
    t = planted.table_meta[0]
    assert t.id == "Table 1" and t.caption.startswith("Table 1: Planted final scores")
    assert "LinUCB | 12.1 | 3.4" in t.text and "Greedy | 30.5 | 1.2" in t.text
    assert any("Table 1" in c for c in t.citing_paragraphs)
    assert planted.tables[0].startswith("Table 1: Planted final scores.")


def test_references_numbered(planted):
    assert len(planted.references) == 3
    assert planted.references[0].startswith("[1] L. Li") and planted.references[0].endswith("WWW, 2010.")
    assert planted.references[2].endswith("CVPR, 2016.")


def test_references_hanging_indent(tmp_path):
    make_planted_pdf(tmp_path / "h.pdf", ref_style="hanging")
    doc = parse_paper(str(tmp_path / "h.pdf"), "h1", str(tmp_path / "out"), rel_to=str(tmp_path))
    assert len(doc.references) == 3
    assert doc.references[0].startswith("Li, L.") and doc.references[0].endswith("WWW, 2010.")
    assert doc.references[1].startswith("Kingma") and doc.references[2].startswith("He, K.")


def test_clean_control_has_nothing_planted(tmp_path):
    make_control_pdf(tmp_path / "c.pdf")
    doc = parse_paper(str(tmp_path / "c.pdf"), "c1", str(tmp_path / "out"), rel_to=str(tmp_path))
    assert doc.equations == [] and doc.figures == [] and doc.tables == [] and doc.references == []
    assert list(doc.sections) == ["1 Introduction", "2 Discussion"]
    f = doc.profile_features
    assert f["n_equations"] == 0 and f["eq_per_1k_tokens"] == 0 and f["n_tokens_est"] > 100


def test_cli_writes_json_and_features(tmp_path):
    (tmp_path / "pdf").mkdir()
    make_planted_pdf(tmp_path / "pdf" / "p1.pdf")
    with open(tmp_path / "papers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "arxiv_id", "title", "category"])
        w.writerow(["p1", "0000.0000", "Planted Regret Study", "theory"])
    rc = parse_main(["--csv", str(tmp_path / "papers.csv"), "--pdf-dir", str(tmp_path / "pdf"),
                     "--src-dir", str(tmp_path / "src"), "--out", str(tmp_path / "parsed")])
    assert rc == 0
    doc = from_json(ParsedPaper, (tmp_path / "parsed" / "p1.json").read_text())
    assert doc.id == "p1" and doc.category == "theory" and doc.profile_features["n_equations"] == 2


# -------------------------------------------------------------------------- LaTeX
TEX = r"""
\documentclass{article}
\numberwithin{equation}{section}
\begin{document}
\section{Intro}
Text with a 50\% sign. % a comment with \begin{equation} x=1 \end{equation}
\begin{equation}\label{eq:a} a = b + c \end{equation}
\begin{align}
 x &= y \label{eq:x}\\
 y &= z \nonumber\\
 z &= \begin{cases} 1 & t>0 \\ 0 & \text{else}\end{cases}
\end{align}
\section{Next}
\[ u = v \]
\begin{equation*} unnum = 1 \end{equation*}
\begin{equation} q = r \tag{Q} \end{equation}
$$ w = 2 $$
\begin{equation} s = t \end{equation}
\appendix
\section{Proofs}
\begin{gather} g_1 = 1 \\ g_2 = 2 \end{gather}
\end{document}
"""


def test_latex_numbering_labels_and_planted_cases():
    from src.ingest.latex_eq import _strip_comments
    eqs = extract_equations(_strip_comments(TEX))
    got = [(e.number, e.latex, e.label) for e in eqs]
    assert got[0] == ("1.1", "a = b + c", "eq:a")
    assert got[1] == ("1.2", "x = y", "eq:x")
    assert got[2] == (None, "y = z", None)                                 # \nonumber row
    assert got[3][0] == "1.3" and "\\begin{cases}" in got[3][1] and "t>0" in got[3][1]   # nested \\ not split
    assert got[4] == (None, "u = v", None)                                 # \[ \] unnumbered
    assert got[5] == (None, "unnum = 1", None)                             # starred
    assert got[6][0] == "Q"                                                # \tag
    assert got[7] == (None, "w = 2", None)
    assert got[8][0] == "2.1"                                              # counter restarts per section
    assert [e.number for e in eqs[9:]] == ["A.1", "A.2"]                   # appendix letters
    assert all("commented" not in e.latex for e in eqs) and len(eqs) == 11  # equation inside % comment skipped
    assert eqs[0].section == "Intro" and eqs[8].section == "Next"
    assert "50% sign" in eqs[0].before


def test_latex_without_numberwithin_uses_global_counter():
    tex = TEX.replace("\\numberwithin{equation}{section}", "")
    from src.ingest.latex_eq import _strip_comments
    nums = [e.number for e in extract_equations(_strip_comments(tex)) if e.number]
    assert nums == ["1", "2", "3", "Q", "4", "5", "6"]


def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in members.items():
            data = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_archive_input_expansion_and_untrusted_members(tmp_path):
    main = "\\documentclass{article}\\begin{document}\\input{sec/body}\n\\end{document}"
    body = "\\begin{equation} k = 1 \\end{equation}"
    arc = tmp_path / "s.tar.gz"
    arc.write_bytes(_tar({"main.tex": main, "sec/body.tex": body,
                          "../evil.tex": "\\begin{document}EVIL\\end{document}",
                          "/abs.tex": "\\begin{document}EVIL\\end{document}"}))
    files = read_tex_files(str(arc))
    assert set(files) == {"main.tex", "sec/body.tex"}                    # traversal and absolute paths skipped
    assert [e.latex for e in equations_from_archive(str(arc))] == ["k = 1"]


def test_archive_single_gz_and_pdf_only(tmp_path):
    single = tmp_path / "one.gz"
    single.write_bytes(gzip.compress(b"\\documentclass{article}\\begin{document}\\begin{equation}m=1\\end{equation}\\end{document}"))
    assert [e.latex for e in equations_from_archive(str(single))] == ["m=1"]
    pdf_only = tmp_path / "pdf.gz"
    pdf_only.write_bytes(b"%PDF-1.5 not latex")
    assert equations_from_archive(str(pdf_only)) == []


def test_tex_equations_take_priority_over_pdf(tmp_path):
    make_planted_pdf(tmp_path / "p.pdf")
    arc = tmp_path / "p.tar.gz"
    arc.write_bytes(_tar({"main.tex": "\\documentclass{article}\\begin{document}"
                                      "\\begin{equation} E = mc^2 \\end{equation}\\end{document}"}))
    doc = parse_paper(str(tmp_path / "p.pdf"), "p1", str(tmp_path / "out"), src_archive=str(arc), rel_to=str(tmp_path))
    assert doc.equation_source == "tex" and doc.equations == ["(1) E = mc^2"]


# ----------------------------------------------------- features, schema, hand-off
def test_features_exclude_references_and_count_numbered():
    doc = PaperDoc(
        id="x", sections={"1 Intro": "a" * 4000, "References": "b" * 40000},
        equations=["(1) a=b", "c=d"], figures=["f.png"], tables=["Table 1: t"], references=["r"] * 7,
    )
    f = compute_features(doc)
    assert f["n_tokens_est"] == 1000.0                     # references are not body text
    assert f["n_equations"] == 2 and f["n_numbered_equations"] == 1
    assert f["eq_per_1k_tokens"] == 2.0 and f["n_references"] == 7 and f["n_sections"] == 2


def test_parsedpaper_is_a_paperdoc_and_roundtrips(planted):
    assert isinstance(planted, PaperDoc)
    text = to_json(planted)
    again = from_json(ParsedPaper, text)
    assert again.equations == planted.equations and again.figure_meta[0].id == "Figure 1"
    plain = planted.to_paperdoc()
    assert type(plain) is PaperDoc and plain.references == planted.references
    if hasattr(PaperDoc, "model_validate_json"):          # old consumers ignore the extras
        assert PaperDoc.model_validate_json(text).id == "p1"


def test_handoff_to_agent_payload_with_mockllm(planted):
    """A parsed paper can feed a TaskRequest, and MockLLM answers it (no real model involved)."""
    task = TaskRequest(agent_type="summarizer", paper_id=planted.id,
                       payload={"abstract": planted.sections["Abstract"]},
                       context_features=planted.profile_features)
    resp = asyncio.run(MockLLM().call_model("qwen3:14b", [{"role": "user", "content": task.payload["abstract"]}]))
    assert resp.model_name == "qwen3:14b" and resp.tokens_in > 5


def test_folder_without_csv_uses_file_names_as_ids(tmp_path):
    """The input folder is enough: no papers.csv, no LaTeX source."""
    (tmp_path / "in").mkdir()
    make_planted_pdf(tmp_path / "in" / "my_paper.pdf")
    make_control_pdf(tmp_path / "in" / "note.pdf")
    rc = parse_main(["--pdf-dir", str(tmp_path / "in"), "--src-dir", str(tmp_path / "none"),
                     "--csv", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "out")])
    assert rc == 0
    assert sorted(p.name for p in (tmp_path / "out").glob("*.json")) == ["my_paper.json", "note.json"]
    doc = from_json(ParsedPaper, (tmp_path / "out" / "my_paper.json").read_text())
    assert doc.title == "Planted Regret Study" and doc.category == ""     # title read from the PDF itself


def test_empty_input_folder_returns_error(tmp_path):
    (tmp_path / "in").mkdir()
    assert parse_main(["--pdf-dir", str(tmp_path / "in"), "--out", str(tmp_path / "out"),
                       "--csv", str(tmp_path / "x.csv")]) == 1