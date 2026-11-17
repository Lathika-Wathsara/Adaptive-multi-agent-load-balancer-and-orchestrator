# Created: 2026-11-17
# Created by: Lathika
# Purpose: Live check on real parsed papers: runs automatic pass/fail gates and
#          writes a spot-check sheet for comparing equations and references by eye.
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.models import ParsedPaper, from_json  # noqa: E402

DEFAULT_IDS = ["t01", "t05", "e03", "e05", "m01"]       # 2 theory, 2 empirical, 1 mixed (all train split)


def gates(doc: ParsedPaper):
    pf, refs = doc.profile_features, doc.references
    abstract = doc.sections.get("Abstract", "")
    out = [
        ("title", bool(doc.title)),
        ("abstract >= 200 chars", len(abstract) >= 200),
        (">= 4 sections", len(doc.sections) >= 4),
        ("references >= 8", len(refs) >= 8),
        ("no merged reference (<= 700 chars each)", all(len(r) <= 700 for r in refs)),
        ("no fragment reference (>= 30 chars each)", all(len(r) >= 30 for r in refs)),
    ]
    if doc.category == "theory":
        out.append(("theory: >= 5 equations", pf.get("n_equations", 0) >= 5))
    else:
        out.append(("figures + tables >= 1", pf.get("n_figures", 0) + pf.get("n_tables", 0) >= 1))
    out.append(("figure PNGs exist", all(Path(p).exists() for p in doc.figures)))
    return out


def sample(items, k=3):
    if len(items) <= k:
        return list(items)
    idx = sorted({0, len(items) // 2, len(items) - 1})
    return [items[i] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="data/papers/parsed")
    ap.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--need", type=int, default=5)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    md, passed = ["# B1 spot-check sheet", ""], 0
    for pid in args.ids:
        f = Path(args.parsed) / f"{pid}.json"
        if not f.exists():
            print(f"[{pid}] not parsed yet ({f})")
            md += [f"## {pid}: NOT PARSED", ""]
            continue
        doc = from_json(ParsedPaper, f.read_text(encoding="utf-8"))
        res = gates(doc)
        ok = all(v for _, v in res)
        passed += ok
        pf = doc.profile_features
        print(f"[{pid}] {'PASS' if ok else 'FAIL'}  {doc.category:9s} pages={doc.n_pages} sec={len(doc.sections)} "
              f"eq={int(pf['n_equations'])}({doc.equation_source}) fig={int(pf['n_figures'])} "
              f"tab={int(pf['n_tables'])} ref={int(pf['n_references'])} tok~{int(pf['n_tokens_est'])}")
        for name, v in res:
            if not v:
                print(f"      FAILED gate: {name}")
        blind = sum(1 for x in doc.figure_meta if x.fallback_crop)
        md += [f"## {pid}  {doc.title}", f"{doc.category}, {doc.n_pages} pages, equations from {doc.equation_source}, "
               f"{blind} blind figure crop(s)", "", "Sections: " + " | ".join(doc.sections), "", "**Equations**"]
        md += [f"- ({e.number}) [{e.section}] `{e.latex[:160]}`" for e in sample(doc.equation_meta, 4)]
        md += ["", "**References** (total %d)" % len(doc.references)]
        md += [f"- {r[:200]}" for r in sample(doc.references, 4)]
        md += ["", "**Figures**"] + [f"- {x.id} -> `{x.path}`: {x.caption[:100]}" for x in doc.figure_meta[:3]]
        md += ["", f"- [ ] equations match the PDF ({len(sample(doc.equation_meta, 4))} checked)",
               "- [ ] references match the PDF (first, middle, last)", "- [ ] figure crops contain the whole figure", ""]
    print(f"\n{passed}/{len(args.ids)} papers pass the automatic gates (need {args.need}).")
    if args.report:
        Path(args.report).write_text("\n".join(md), encoding="utf-8")
        print(f"spot-check sheet: {args.report}  (tick the boxes after comparing with the PDFs)")
    return 0 if passed >= args.need else 1


if __name__ == "__main__":
    raise SystemExit(main())
