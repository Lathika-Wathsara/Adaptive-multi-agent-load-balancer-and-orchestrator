# Created: 2026-11-17
# Updated: 2026-11-18 (prints decisions, extra rounds, quotes and the human list)
# Created by: Lathika
# Purpose: Live check of the novelty checker with the real models and real paper APIs.
#          Runs two planted papers whose claims are famous, already published results
#          (the Transformer and dropout), then any parsed papers you name as a control.
#          Pass: for each planted paper, some claim is flagged (already_known, partly_known
#          or sent to a human) and its evidence or best matches contain the real original.
#          Parsed papers are only printed for a hand check. Results go to /data/novelty/.
#
# Usage: python3 scripts/novelty_check.py [paper_id ...] [--parsed /data/papers/parsed]

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.novelty import check_novelty  # noqa: E402
from src.common.schemas import PaperDoc  # noqa: E402
from src.serving.client import LLMClient  # noqa: E402
from src.tools.embedder import OLLAMA_URL  # noqa: E402

OUT_DIR = Path("/data/novelty")

# Titles differ from the real papers, so the paper itself is not filtered out.
# Each entry is (planted paper, year of the original paper, words that must appear in
# the original paper's title). Papers newer than the original are not counted as prior work.
PLANTED = [
    (PaperDoc(
        id="planted_transformer",
        title="Sequence modelling with self-attention only",
        sections={"Abstract": (
            "We propose a new network architecture for sequence transduction that is based "
            "solely on attention mechanisms, dispensing with recurrence and convolutions "
            "entirely. Experiments on machine translation show better quality and far "
            "less training time than recurrent models.")}), 2017, "attention is all you need"),
    (PaperDoc(
        id="planted_dropout",
        title="Random unit omission against overfitting",
        sections={"Abstract": (
            "Deep networks with many parameters overfit small datasets. We propose to "
            "randomly drop units, together with their connections, during training. This "
            "prevents units from co-adapting too much and gives large improvements over "
            "other regularization methods.")}), 2014, "dropout: a simple way"),
]


def show(paper, result):
    claims = json.loads(result.raw_output)
    print(f"\n=== {paper.id}: {len(claims)} claims, {len(result.findings)} findings")
    for c in claims:
        print(f"[{c['verdict']} -> {c['decision']}, {c['rounds']} extra rounds, "
              f"confidence {c['confidence']:.2f}] {c['claim'][:100]}")
        print(f"     {c['reason'][:150]}")
        if c["doubts"]:
            print(f"     doubts: {c['doubts'][:150]}")
        if c["rounds"]:
            print(f"     searched: {c['queries'][2:]}")
        for e in c["evidence"]:
            print(f"     - {e['title'][:70]} ({e['year']})  {e['url']}")
            if e.get("quote"):
                print(f"       quote: {e['quote'][:150]}")
        for e in c.get("best_matches", []):
            low = " LOW SIMILARITY" if e.get("low_similarity") else ""
            print(f"     * for the human: {e['title'][:60]} ({e['year']}, score {e['score']}){low}  {e['url']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{paper.id}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return claims


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="parsed paper ids to run as controls")
    ap.add_argument("--parsed", default="/data/papers/parsed")
    args = ap.parse_args()

    client = LLMClient(use_mock=False, ollama_url=OLLAMA_URL)
    passed = True

    for paper, year, original in PLANTED:
        claims = show(paper, await check_novelty(paper, client, paper_year=year))
        hit = any((c["verdict"] in ("already_known", "partly_known") or c["decision"] == "human")
                  and any(original in e["title"].lower()
                          for e in c["evidence"] + c.get("best_matches", []))
                  for c in claims)
        print(f"planted claim flagged with the original paper ({original}):",
              "yes" if hit else "NO")
        passed = passed and hit

    for paper_id in args.ids:
        data = json.loads((Path(args.parsed) / f"{paper_id}.json").read_text(encoding="utf-8"))
        paper = PaperDoc(**{k: v for k, v in data.items() if k in PaperDoc.model_fields})
        show(paper, await check_novelty(paper, client))
        print("control paper: read the verdicts and URLs above by hand")

    print("\nPASS" if passed else "\nCHECK BY HAND")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))