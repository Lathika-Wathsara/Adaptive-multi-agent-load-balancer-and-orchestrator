# Created: 2026-11-18
# Updated: 2026-11-18 (prints where the original paper ranks and the top papers per claim)
# Created by: Lathika
# Purpose: Temporary experiment. Does running the strict quote check on "novel" verdicts
#          fix the false "novel" answers, and what does it cost in time? For each paper it
#          builds the shared pool, judges every claim once (no retries), and for every
#          claim the judge calls novel it asks the strict checker about the top papers.
#          For planted papers it also shows whether the original paper is in the pool and
#          where it ranks, so a miss can be told apart from a retrieval problem.
#          Planted papers are all known work, so every flip there is correct. For a parsed
#          control paper, read the flips by hand.
#
# Usage: python3 scripts/novelty_novel_test.py [paper_id ...] [--parsed /data/papers/parsed]

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from src.tools import keys  # noqa: E402
keys.ask_for_keys()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novelty_check import PLANTED  # noqa: E402
from src.agents import novelty  # noqa: E402
from src.common.schemas import PaperDoc  # noqa: E402
from src.serving.client import LLMClient  # noqa: E402
from src.tools import embedder  # noqa: E402
from src.tools.embedder import OLLAMA_URL  # noqa: E402


async def test_paper(paper, year, original, client, embed):
    print(f"\n=== {paper.id}")
    claims = await novelty.extract_claims(paper, client)
    pool = {}
    novelty.add_to_pool(pool, novelty.retrieve(novelty.paper_queries(paper)))
    for item in claims:
        novelty.add_to_pool(pool, novelty.retrieve([item["query"], item["claim"]]))
    raw = list(pool.values())
    keep = novelty.usable(raw, paper.title, year)
    if original:
        print(f"pool {len(raw)}, usable {len(keep)}, original in pool: "
              f"{any(original in c['title'].lower() for c in raw)}")

    judged = novel = flipped = 0
    judge_time = check_time = 0.0
    for item in claims:
        ranked_all = novelty.rank(item["claim"], keep, embed, len(keep), novelty.MIN_SIMILARITY)
        ranked = ranked_all[:novelty.TOP_K]
        if not ranked:
            print(f"[no papers above the floor] {item['claim'][:90]}")
            continue
        start = time.time()
        out = await novelty.compare(item["claim"], ranked, client)
        judge_time += time.time() - start
        judged += 1
        print(f"[{out['verdict']} {out['confidence']:.2f}] {item['claim'][:90]}")
        if original:
            pos = next((i for i, c in enumerate(ranked_all, 1)
                        if original in c["title"].lower()), None)
            print(f"     original rank: {pos} of {len(ranked_all)}")
        for c in ranked:
            print(f"     saw {c['score']}  {c['title'][:70]} ({c['year']})")
        if out["verdict"] != "novel":
            continue
        novel += 1
        start = time.time()
        backed = await novelty.find_support(item["claim"], [], ranked, client)
        check_time += time.time() - start
        if backed:
            flipped += 1
            cand, quote = backed[0]
            print(f"     FLIPPED by: {cand['title'][:70]} ({cand['year']})")
            print(f"     quote: {quote[:150]}")
        else:
            print("     stays novel (no paper states it)")
    print(f"-- {judged} judged, {novel} novel, {flipped} flipped | "
          f"judge time {judge_time:.0f}s, extra check time {check_time:.0f}s")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="parsed paper ids to run as controls")
    ap.add_argument("--parsed", default="/data/papers/parsed")
    args = ap.parse_args()

    client = LLMClient(use_mock=False, ollama_url=OLLAMA_URL)
    embed = novelty.memoize_embed(embedder.embed)

    for paper, year, original in PLANTED:
        await test_paper(paper, year, original, client, embed)
    for paper_id in args.ids:
        data = json.loads((Path(args.parsed) / f"{paper_id}.json").read_text(encoding="utf-8"))
        paper = PaperDoc(**{k: v for k, v in data.items() if k in PaperDoc.model_fields})
        await test_paper(paper, None, None, client, embed)


if __name__ == "__main__":
    asyncio.run(main())