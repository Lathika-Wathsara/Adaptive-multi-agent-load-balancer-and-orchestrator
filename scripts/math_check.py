# File: scripts/math_check.py
# Created: 2026-11-18
# Created by: Lathika
# Purpose: Run a live smoke check of the math checker on four planted derivations (clean and
#             sign-flipped eq 2), passing if at least 3/4 flipped eq 2s are flagged and at most 1/12 clean
#             equations gets math_error; also prints CLI-specified parsed papers/equations and saves to /data/math/.
#
# Usage: python3 scripts/math_check.py [paper_id ...] [--parsed /data/papers/parsed]
#            [--model deepseek-r1:7b] [--cases 1 2 3 4] [--numbers 26 27]

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.math_checker import MATH_MODEL, check_math  # noqa: E402
from src.ingest.models import EquationInfo, ParsedPaper, from_json  # noqa: E402
from src.serving.client import LLMClient  # noqa: E402
from src.tools.embedder import OLLAMA_URL  # noqa: E402

OUT_DIR = Path("/data/math")

# Each case: (name, text before eq 1, eq 1, text before eq 2, correct eq 2, flipped eq 2,
#             text before eq 3, eq 3). Equation 2 is the derived step that gets the flipped sign.
CASES = [
    ("squared distance",
     "Let a be a received sample and b the transmitted symbol, both complex.",
     r"d = |a - b|^{2}",
     "Expanding the square gives",
     r"d = |a|^{2} + |b|^{2} - 2\,\mathrm{Re}\left(a b^{*}\right)",
     r"d = |a|^{2} + |b|^{2} + 2\,\mathrm{Re}\left(a b^{*}\right)",
     "By the reverse triangle inequality, the distance is bounded below by",
     r"d \ge \left(|a| - |b|\right)^{2}"),
    ("array factor",
     "The array factor of an N-element uniform linear array with phase step psi is",
     r"A(\psi) = \sum_{n=0}^{N-1} e^{jn\psi}",
     r"Summing the geometric series, for $e^{j\psi} \neq 1$ we obtain",
     r"A(\psi) = \frac{1 - e^{jN\psi}}{1 - e^{j\psi}}",
     r"A(\psi) = \frac{1 + e^{jN\psi}}{1 - e^{j\psi}}",
     "Taking the magnitude gives the familiar sinc-like pattern",
     r"|A(\psi)| = \left|\frac{\sin(N\psi/2)}{\sin(\psi/2)}\right|"),
    ("rate identity",
     "With transmit power P and noise power N_0, the achievable rate is",
     r"R = \log_{2}\left(1 + \frac{P}{N_0}\right)",
     "Writing the argument over a common denominator, the rate becomes",
     r"R = \log_{2}\left(P + N_0\right) - \log_{2}\left(N_0\right)",
     r"R = \log_{2}\left(P + N_0\right) + \log_{2}\left(N_0\right)",
     "Dropping the one inside the logarithm gives the high-SNR bound",
     r"R \ge \log_{2}\left(\frac{P}{N_0}\right)"),
    ("rank-one update",
     r"A rank-one interference term is added to the covariance matrix: with $\mathbf{u}$ the interferer's channel,",
     r"\mathbf{R}_u = \mathbf{R} + \mathbf{u}\mathbf{u}^{H}",
     "Applying the Sherman-Morrison formula, the inverse is",
     r"\mathbf{R}_u^{-1} = \mathbf{R}^{-1} - \frac{\mathbf{R}^{-1}\mathbf{u}\mathbf{u}^{H}\mathbf{R}^{-1}}{1 + \mathbf{u}^{H}\mathbf{R}^{-1}\mathbf{u}}",
     r"\mathbf{R}_u^{-1} = \mathbf{R}^{-1} + \frac{\mathbf{R}^{-1}\mathbf{u}\mathbf{u}^{H}\mathbf{R}^{-1}}{1 + \mathbf{u}^{H}\mathbf{R}^{-1}\mathbf{u}}",
     "The interferer's power seen through the new inverse is then",
     r"\mathbf{u}^{H}\mathbf{R}_u^{-1}\mathbf{u} = \frac{\mathbf{u}^{H}\mathbf{R}^{-1}\mathbf{u}}{1 + \mathbf{u}^{H}\mathbf{R}^{-1}\mathbf{u}}"),
]


def make_paper(case_number, flipped):
    """A small ParsedPaper with three numbered equations; equation 2 is flipped or correct."""
    name, intro, eq1, text2, good2, bad2, text3, eq3 = CASES[case_number - 1]
    parts = [(intro, eq1), (text2, bad2 if flipped else good2), (text3, eq3)]
    eqs = [EquationInfo(index=i, number=str(i + 1), latex=latex, section="2 Analysis",
                        before=before, after="", source="tex")
           for i, (before, latex) in enumerate(parts)]
    kind = "flipped" if flipped else "clean"
    return ParsedPaper(id=f"planted_{case_number}_{kind}", title=name, equation_meta=eqs)


def flagged(findings, number):
    """Finding types reported on one equation number."""
    return [f.type for f in findings if f.evidence.startswith(f"Equation ({number}):")]


def show(paper, result, seconds):
    records = json.loads(result.raw_output)
    print(f"\n=== {paper.id} ({paper.title}): {len(records)} equations, "
          f"{len(result.findings)} findings, {seconds:.0f} s")
    for r in records:
        decision = r["decision"][0] if r["decision"] else "-"
        print(f"({r['number']}) model: {r['verdict']} ({r['error_type']}, {r['confidence']:.2f})"
              f"  test: {r['test']['status']}  -> {decision}")
        print(f"      {r['latex'][:90]}")
        if r["reason"]:
            print(f"      {r['reason'][:160]}")
        if r["test"]["detail"]:
            print(f"      test detail: {r['test']['detail'][:120]}")
        if r["reply_end"] and r["test"]["status"] == "none":
            print(f"      end of the model's reply: {r['reply_end'][-300:]!r}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{paper.id}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")


async def run_planted(client, model, case_numbers):
    caught, clean_flags, clean_total = 0, 0, 0
    for n in case_numbers:
        for flipped in (True, False):
            paper = make_paper(n, flipped)
            start = time.time()
            result = await check_math(paper, client, model=model)
            show(paper, result, time.time() - start)
            if flipped:
                types = flagged(result.findings, "2")
                hit = bool(types)
                caught += hit
                print(f"planted sign flip in equation 2 reported: {'yes' if hit else 'NO'} {types}")
            else:
                clean_total += 3
                errors = [f for f in result.findings if f.type == "math_error"]
                clean_flags += len(errors)
                print(f"clean version: math_error findings: {len(errors)}, "
                      f"all findings: {[f.type for f in result.findings]}")
    print(f"\nflipped signs reported: {caught}/{len(case_numbers)}   "
          f"clean equations with a math_error finding: {clean_flags}/{clean_total}")
    return caught >= 3 * len(case_numbers) / 4 and clean_flags <= 1 * clean_total / 12


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="parsed paper ids to run as controls")
    ap.add_argument("--parsed", default="/data/papers/parsed")
    ap.add_argument("--model", default=MATH_MODEL)
    ap.add_argument("--cases", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--numbers", nargs="*", default=None, help="equation numbers to check")
    ap.add_argument("--skip-planted", action="store_true", help="only run the control papers")
    args = ap.parse_args()

    client = LLMClient(use_mock=False, ollama_url=OLLAMA_URL)
    print(f"math model: {args.model}")
    passed = True if args.skip_planted else await run_planted(client, args.model, args.cases)

    for paper_id in args.ids:
        paper = from_json(ParsedPaper, (Path(args.parsed) / f"{paper_id}.json").read_text(encoding="utf-8"))
        start = time.time()
        show(paper, await check_math(paper, client, model=args.model, numbers=args.numbers),
             time.time() - start)
        print("control paper: read every verdict and test above by hand")

    print("\nPASS" if passed else "\nCHECK BY HAND")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))