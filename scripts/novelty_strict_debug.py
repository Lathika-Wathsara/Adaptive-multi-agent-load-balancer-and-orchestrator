# Created: 2026-11-18
# Created by: Lathika
# Purpose: Temporary debug. Shows what the strict checker says when it is given the real
#          "Attention Is All You Need" abstract and claims that its abstract does state.
#          Prints the abstract, the raw model reply, the parsed answer, and how well the
#          quote matches the abstract, so we can tell a model "no" from a quote-matching
#          or JSON problem.
#
# Usage: python3 scripts/novelty_strict_debug.py

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidfuzz import fuzz  # noqa: E402

from src.agents import novelty  # noqa: E402
from src.agents.ref_verifier import normalize  # noqa: E402
from src.serving.client import LLMClient  # noqa: E402
from src.tools import scholar  # noqa: E402
from src.tools.embedder import OLLAMA_URL  # noqa: E402

CLAIMS = [
    "A neural network architecture for sequence transduction is proposed that uses only "
    "self-attention mechanisms, eliminating recurrence and convolution.",
    "Training time for the self-attention-only model is significantly reduced compared to "
    "recurrent models.",
    "The model demonstrates superior performance in sequence-to-sequence tasks without "
    "relying on any form of recurrence or convolution.",
]


async def main():
    client = LLMClient(use_mock=False, ollama_url=OLLAMA_URL)
    found = novelty.retrieve(["Attention is all you need"], dict(scholar.SOURCES) and
                             [s for s in scholar.SOURCES if s[0] == "openalex"])
    original = [c for c in found if c["title"].lower() == "attention is all you need"
                and c["year"] == 2017]
    if not original:
        print("original not found")
        return
    cand = original[0]
    print("ABSTRACT:", cand["abstract"], "\n")

    for claim in CLAIMS:
        system = "You are a strict fact checker."
        user = (
            f"Claim: {claim}\n\nPaper: {cand['title']} ({cand['year']})\n"
            f"Abstract: {cand['abstract'][:1500]}\n\n"
            "Does this abstract state the claim, or an important part of it, or clearly imply "
            "it? Being on the same topic is not enough. Reply with JSON only: "
            '{"supports": true or false, "quote": "the sentence from the abstract that shows it, or empty"}'
        )
        reply = await client.call_model(
            model=novelty.COMPARE_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        data = novelty.parse_json(reply.text)
        print("CLAIM:", claim[:80])
        print("RAW REPLY:", reply.text[:600].replace("\n", " "))
        print("PARSED:", data)
        if isinstance(data, dict) and data.get("quote"):
            score = fuzz.partial_ratio(normalize(str(data["quote"])), normalize(cand["abstract"]))
            print("QUOTE MATCH SCORE:", score, "(needs", novelty.QUOTE_MATCH, ")")
        print()


if __name__ == "__main__":
    asyncio.run(main())