# Created: 2026-11-17
# Created by: Lathika
# Purpose: Live check of the reference verifier on parsed papers. Adds 3 invented
#          references to each paper's real ones, then prints what the verifier says.
#          Pass: all 3 invented references flagged and most real ones accepted.
#
# Usage: python3 scripts/ref_check.py [paper_id ...] [--parsed /data/papers/parsed]

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.ref_verifier import verify_references  # noqa: E402

# Invented by hand. They look plausible but are not real papers.
FAKE_REFS = [
    '[F1] J. K. Moreno and L. Tanaka, "Adaptive Terahertz Beam Alignment via Federated '
    'Spectral Hashing," IEEE Trans. Wireless Commun., vol. 22, no. 4, pp. 2210-2224, 2023.',
    '[F2] S. Varga, P. Iyer and M. Okafor, "Sensing-Aware Hybrid Precoding with Learned '
    'Phase Quantizers," IEEE J. Sel. Areas Commun., vol. 41, no. 9, pp. 2901-2915, 2023.',
    '[F3] H. Lindqvist and R. Abara, "Energy-Efficient Sub-Terahertz Radar-Communication '
    'Co-design under Hardware Impairments," IEEE Trans. Signal Process., vol. 71, '
    'pp. 1330-1345, 2023.',
]


def check_paper(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    real = data.get("references", [])
    checks = verify_references(real + FAKE_REFS)
    real_checks, fake_checks = checks[:len(real)], checks[len(real):]

    print(f"\n=== {json_path.stem}: {len(real)} real references + 3 invented")
    for c in checks:
        tag = "FAKE" if c in fake_checks else "real"
        print(f"{tag}  {c['status']:<10} score={c['title_score']:>3}  {c['ref'][:70]}")
        if c["status"] in ("mismatch", "not_found") and tag == "real":
            print(f"        why: {c['reason']}")

    fakes_flagged = sum(c["status"] in ("not_found", "mismatch") for c in fake_checks)
    real_ok = sum(c["status"] == "exists" for c in real_checks)
    unchecked = sum(c["status"] == "unchecked" for c in checks)
    print(f"invented flagged: {fakes_flagged}/3   real accepted: {real_ok}/{len(real)}"
          f"   unchecked: {unchecked}")
    if unchecked:
        print("Some lookups failed: is the network reachable from the container?")
    passed = fakes_flagged == 3 and real_ok >= 0.8 * len(real)
    print("PASS" if passed else "CHECK BY HAND")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="paper ids (default: every parsed paper)")
    ap.add_argument("--parsed", default="/data/papers/parsed")
    args = ap.parse_args()

    folder = Path(args.parsed)
    paths = [folder / f"{i}.json" for i in args.ids] if args.ids else sorted(folder.glob("*.json"))
    if not paths:
        print(f"no parsed papers in {folder}")
        return 1
    results = [check_paper(p) for p in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())