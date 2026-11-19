# Created: 2026-11-18
# Created by: Lathika
# Purpose: Find out why a planted original paper is not found. For each planted paper it
#          (1) asks every source for the original's exact title and shows whether the
#          record has an abstract, (2) builds the shared pool like the checker does and
#          shows whether the original is in it, and (3) for each claim shows the rank and
#          cosine score of the original next to the top 4 papers the judge would see.
#
# Usage: python3 scripts/novelty_debug.py

import asyncio
import sys
from pathlib import Path
from src.tools import keys  # noqa: E402
keys.ask_for_keys()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novelty_check import PLANTED  # noqa: E402
from src.agents import novelty  # noqa: E402
from src.serving.client import LLMClient  # noqa: E402
from src.tools import embedder, scholar  # noqa: E402
from src.tools.embedder import OLLAMA_URL  # noqa: E402

TITLES = {"attention is all you need": "Attention is all you need",
          "dropout: a simple way": "Dropout: a simple way to prevent neural networks from overfitting"}


async def main():
    client = LLMClient(use_mock=False, ollama_url=OLLAMA_URL)
    embed = novelty.memoize_embed(embedder.embed)

    for paper, year, original in PLANTED:
        print(f"\n=== {paper.id} (looking for '{original}', year {year})")

        print("-- exact title search in each source")
        for name, search in scholar.SOURCES:
            try:
                found = search(TITLES[original], limit=5)
            except scholar.ScholarError as err:
                print(f"   {name}: failed ({err})")
                continue
            hits = [c for c in found if original in c["title"].lower()]
            for c in hits:
                print(f"   {name}: found, year {c['year']}, abstract {len(c['abstract'])} chars")
            if not hits:
                print(f"   {name}: not in the top 5")

        claims = await novelty.extract_claims(paper, client)
        pool = {}
        novelty.add_to_pool(pool, novelty.retrieve(novelty.paper_queries(paper)))
        for item in claims:
            novelty.add_to_pool(pool, novelty.retrieve([item["query"], item["claim"]]))
        raw = list(pool.values())
        keep = novelty.usable(raw, paper.title, year)
        print(f"-- pool: {len(raw)} papers, {len(keep)} usable")
        print("   original in pool:", any(original in c["title"].lower() for c in raw),
              "| usable:", any(original in c["title"].lower() for c in keep))

        for item in claims:
            ranked = novelty.rank(item["claim"], keep, embed, len(keep), 0.0)
            pos = next((i for i, c in enumerate(ranked, 1) if original in c["title"].lower()), None)
            print(f"-- claim: {item['claim'][:80]}")
            print(f"   original rank: {pos} of {len(ranked)}"
                  + (f", score {ranked[pos - 1]['score']}" if pos else ""))
            for c in ranked[:4]:
                print(f"     {c['score']}  {c['title'][:70]} ({c['year']})")


if __name__ == "__main__":
    asyncio.run(main())