# File: scripts/vision_check.py
# Created: 2026-11-19
# Created by: Lathika
# Purpose: Live check of the vision checker on four hand-made papers (a labelled bar chart, a line
#          chart, an unlabelled bar chart, a table), each with a clean version and a version with
#          one changed number. Passes if 3 of 4 changed numbers are flagged and at most 1 of 8
#          clean claims gets a mismatch. Can also run parsed papers; results go to /data/vision/.
#
# Usage: python3 scripts/vision_check.py [paper_id ...] [--parsed /data/papers/parsed]
#            [--model qwen2.5vl:7b] [--cases 1 2 3 4] [--items "Figure 2" "Table 1"] [--skip-planted]

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.vision_checker import VISION_MODEL, check_vision  # noqa: E402
from src.ingest.models import FigureInfo, ParsedPaper, TableInfo, from_json  # noqa: E402
from src.serving.client import LLMClient  # noqa: E402
from src.tools.embedder import OLLAMA_URL  # noqa: E402

OUT_DIR = Path("/data/vision")
FIG_DIR = OUT_DIR / "figures"


# ------------------------------------------------------------------------------ the charts
def bars_labeled(path):
    names, values = ["Baseline", "Method A", "Method B", "Proposed"], [71.4, 78.2, 85.6, 91.3]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bars = ax.bar(names, values, color="#4c78a8")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Classification accuracy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def lines_snr(path):
    snr = [0, 5, 10, 15, 20]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(snr, [1.0, 1.9, 3.4, 5.1, 6.7], "o-", label="Proposed")
    ax.plot(snr, [0.8, 1.4, 2.3, 3.4, 4.5], "s--", label="Baseline")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Spectral efficiency (bit/s/Hz)")
    ax.set_xticks(snr)
    ax.set_yticks(range(0, 9))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def bars_unlabeled(path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(["Digital", "Hybrid", "Proposed"], [3.0, 5.0, 8.0], color="#59a14f")
    ax.set_ylabel("Energy efficiency (Gbit/J)")
    ax.set_ylim(0, 10)
    ax.set_yticks(range(0, 11))
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


TABLE_TEXT = ("Method | Accuracy (%) | F1 (%) | Latency (ms)\n"
              "Baseline | 71.4 | 69.8 | 12.5\n"
              "Method A | 78.2 | 76.9 | 15.1\n"
              "Proposed | 91.3 | 90.2 | 18.7")

# Each case: (name, kind, caption, drawing function, text template with {a} and {b}, correct a,
#             changed a, b). {a} is the number that is changed in the flawed version; b stays correct.
CASES = [
    ("labelled bars", "figure", "Figure 1: Classification accuracy of four methods on the test set.", bars_labeled,
     "Figure 1 compares four methods on the test set. The proposed scheme reaches {a}% accuracy, "
     "while the baseline reaches only {b}%. All results are averaged over 5 runs.",
     "91.3", "96.1", "71.4"),
    ("line chart", "figure", "Figure 1: Spectral efficiency of the proposed scheme and the baseline versus SNR.",
     lines_snr,
     "Figure 1 shows the spectral efficiency versus SNR. At 20 dB the proposed scheme achieves {a} bit/s/Hz, "
     "while at 10 dB the baseline achieves {b} bit/s/Hz. Each point is an average over 1000 channel draws.",
     "6.7", "8.7", "2.3"),
    ("unlabelled bars", "figure", "Figure 1: Energy efficiency of three beamforming designs.", bars_unlabeled,
     "Figure 1 compares the energy efficiency of three designs. The proposed design reaches {a} Gbit/J, "
     "while the fully digital design reaches only {b} Gbit/J. All designs use 64 antennas.",
     "8.0", "5.5", "3.0"),
    ("table", "table", "Table 1: Accuracy, F1 score and latency of three methods.", None,
     "Table 1 lists the results. The proposed model attains an F1 score of {a}% with a latency of {b} ms. "
     "The test set was split into 3 folds.",
     "90.2", "93.7", "18.7"),
]


def make_paper(case_number, flawed):
    """A ParsedPaper with one figure or table; the text states two numbers, a (changed if flawed) and b."""
    name, kind, caption, draw, template, good, bad, other = CASES[case_number - 1]
    text = template.format(a=bad if flawed else good, b=other)
    label = "flawed" if flawed else "clean"
    paper_id = f"planted_{case_number}_{label}"
    if kind == "figure":
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = FIG_DIR / f"case_{case_number}.png"
        draw(path)
        figure = FigureInfo(id="Figure 1", path=str(path), page=1, caption=caption, citing_paragraphs=[text])
        return ParsedPaper(id=paper_id, title=name, figure_meta=[figure]), bad if flawed else None
    table = TableInfo(id="Table 1", page=1, caption=caption, text=TABLE_TEXT, citing_paragraphs=[text])
    return ParsedPaper(id=paper_id, title=name, table_meta=[table]), bad if flawed else None


# -------------------------------------------------------------------------------- reporting
def mismatches(findings):
    return [f for f in findings if f.type.endswith("_mismatch")]


def show(paper, result, seconds):
    records = json.loads(result.raw_output)
    print(f"\n=== {paper.id} ({paper.title}): {len(records)} items, {len(result.findings)} findings, "
          f"{seconds:.0f} s")
    for r in records:
        print(f"{r['id']}: {r['status']} {r['reason']}")
        for d in r["dropped"]:
            print(f"      dropped claim ({d['dropped']}): {d['quote'][:80]}")
        for c in r["claims"]:
            decision = c["decision"][0] if c["decision"] else "-"
            read = "not found" if c["read"] is None else f"{c['read']:g} ({c['how']})"
            print(f"      text {c['claimed']:g}{c['unit']}  figure/table: {read}  -> {decision}")
            print(f"         {c['question'][:90]}  | {c['note'][:70]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{paper.id}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")


async def run_planted(client, model, case_numbers):
    caught, clean_flags, clean_claims = 0, 0, 0
    for n in case_numbers:
        for flawed in (True, False):
            paper, changed = make_paper(n, flawed)
            start = time.time()
            result = await check_vision(paper, client, model=model)
            show(paper, result, time.time() - start)
            wrong = mismatches(result.findings)
            if flawed:
                hit = any(f"the text says {changed}" in f.evidence for f in wrong)
                caught += hit
                others = len(wrong) - hit
                print(f"changed number {changed} reported: {'yes' if hit else 'NO'}   "
                      f"mismatches on the other number: {others}   all findings: {[f.type for f in result.findings]}")
            else:
                clean_claims += 2
                clean_flags += len(wrong)
                print(f"clean version: mismatch findings: {len(wrong)}   all findings: {[f.type for f in result.findings]}")
    print(f"\nchanged numbers reported: {caught}/{len(case_numbers)}   "
          f"clean claims with a mismatch finding: {clean_flags}/{clean_claims}")
    return caught >= 3 * len(case_numbers) / 4 and clean_flags <= clean_claims / 8


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="parsed paper ids to run as controls")
    ap.add_argument("--parsed", default="/data/papers/parsed")
    ap.add_argument("--model", default=VISION_MODEL)
    ap.add_argument("--cases", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--items", nargs="*", default=None, help='only these, e.g. "Figure 2" "Table 1"')
    ap.add_argument("--skip-planted", action="store_true", help="only run the control papers")
    args = ap.parse_args()

    client = LLMClient(use_mock=False, ollama_url=OLLAMA_URL)
    print(f"vision model: {args.model}")
    passed = True if args.skip_planted else await run_planted(client, args.model, args.cases)

    for paper_id in args.ids:
        paper = from_json(ParsedPaper, (Path(args.parsed) / f"{paper_id}.json").read_text(encoding="utf-8"))
        start = time.time()
        show(paper, await check_vision(paper, client, model=args.model, ids=args.items), time.time() - start)
        print("control paper: look at every figure and every claim above by hand")

    print("\nPASS" if passed else "\nCHECK BY HAND")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
