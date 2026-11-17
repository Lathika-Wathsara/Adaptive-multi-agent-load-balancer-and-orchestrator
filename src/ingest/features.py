"""Profile features of a parsed paper (B1) -> PaperDoc.profile_features.

Raw counts and densities only; Person A normalises them for the LinUCB context.
Token length is a character-count estimate (chars / 4), good enough for a context
feature and free of extra dependencies.
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict

from src.common.schemas import PaperDoc

_SKIP_SECTIONS = {"references", "bibliography"}
_NUMBERED_EQ = re.compile(r"^\(\S+\)\s")


def _norm(heading: str) -> str:
    h = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[A-Z]\.|[IVX]+\.)\s+", "", heading.strip())
    return h.strip(" :.").lower()


def body_text(doc: PaperDoc) -> str:
    return "\n\n".join(t for h, t in doc.sections.items() if _norm(h) not in _SKIP_SECTIONS)


def compute_features(doc: PaperDoc) -> Dict[str, float]:
    text = body_text(doc)
    n_tokens = len(text) / 4.0
    kt = max(n_tokens / 1000.0, 1e-9)
    n_eq = len(doc.equations)
    return {
        "n_tokens_est": round(n_tokens, 1),
        "n_sections": float(len(doc.sections)),
        "n_equations": float(n_eq),
        "n_numbered_equations": float(sum(1 for e in doc.equations if _NUMBERED_EQ.match(e))),
        "n_figures": float(len(doc.figures)),
        "n_tables": float(len(doc.tables)),
        "n_references": float(len(doc.references)),
        "eq_per_1k_tokens": round(n_eq / kt, 3),
        "fig_per_1k_tokens": round(len(doc.figures) / kt, 3),
        "tab_per_1k_tokens": round(len(doc.tables) / kt, 3),
    }


def add_features(doc: PaperDoc) -> PaperDoc:
    doc.profile_features = compute_features(doc)
    return doc


def main(argv=None) -> int:
    """python -m src.ingest.features data/papers/parsed --csv data/features.csv"""
    from src.ingest.models import ParsedPaper, from_json

    ap = argparse.ArgumentParser()
    ap.add_argument("parsed_dir")
    ap.add_argument("--csv", default="")
    args = ap.parse_args(argv)

    rows = []
    for p in sorted(Path(args.parsed_dir).glob("*.json")):
        doc = from_json(ParsedPaper, p.read_text(encoding="utf-8"))
        rows.append({"id": doc.id, "category": doc.category, **compute_features(doc)})
    if not rows:
        print("no parsed papers found", file=sys.stderr)
        return 1
    cols = list(rows[0])
    print("  ".join(f"{c[:14]:>14}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c])[:14]:>14}" for c in cols))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
