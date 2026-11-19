# Created: 2026-11-18
# Updated: 2026-11-18 (pauses between requests, retries on 429, uses the OpenAlex key)
# Created by: Lathika
# Purpose: Temporary experiment, no language models. For several queries a claim extractor
#          could plausibly produce, shows where a known original paper lands in OpenAlex
#          results: sorted by relevance (top 15 and top 50) and sorted by citation count
#          (top 15), all restricted to papers published up to the original's year.
#          Tells us whether sorting by citations would find the originals reliably.
#
# Usage: python3 scripts/retrieval_probe.py

import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools import keys  # noqa: E402

URL = "https://api.openalex.org/works"
EMAIL = os.environ.get("SCHOLAR_EMAIL", "")
PAUSE = 1.5        # seconds between requests
RETRIES = 5        # tries per request when OpenAlex answers 429

TESTS = [
    ("attention is all you need", 2017, [
        "self-attention sequence transduction",
        "attention mechanisms dispensing with recurrence and convolutions",
        "attention-only sequence modelling machine translation",
        "neural network architecture sequence transduction only self-attention",
        "translation quality training time recurrent models attention",
    ]),
    ("dropout: a simple way", 2014, [
        "randomly dropping units training prevent co-adaptation",
        "random unit omission overfitting",
        "unit omission regularization neural networks",
        "dropout regularization small datasets",
        "stochastic unit removal during training generalization",
    ]),
]


def fetch(params, headers):
    """One OpenAlex request; waits and retries when the server says 429."""
    for attempt in range(RETRIES):
        time.sleep(PAUSE)
        resp = httpx.get(URL, params=params, headers=headers, timeout=60)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()["results"]
        time.sleep(5 * (attempt + 1))
    return None


def rank_of(original, query, year, sort, limit, headers):
    """Position (1-based) of the original in the OpenAlex results, "-" if it is not
    there, or "429" if the server kept refusing."""
    params = {"search": query, "per-page": limit, "select": "title",
              "filter": f"to_publication_date:{year}-12-31"}
    if sort:
        params["sort"] = sort
    if EMAIL:
        params["mailto"] = EMAIL
    results = fetch(params, headers)
    if results is None:
        return "429"
    for i, work in enumerate(results, 1):
        if original in (work["title"] or "").lower():
            return str(i)
    return "-"


def main():
    keys.ask_for_keys()
    key = keys.get_key("OPENALEX_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    print("using an OpenAlex key" if key else "no OpenAlex key (shared daily budget)")

    for original, year, queries in TESTS:
        print(f"\n=== {original} (papers up to {year})")
        print(f"{'relevance 15':>13} {'relevance 50':>13} {'citations 15':>13}  query")
        for query in queries:
            rel15 = rank_of(original, query, year, None, 15, headers)
            rel50 = rank_of(original, query, year, None, 50, headers)
            cite15 = rank_of(original, query, year, "cited_by_count:desc", 15, headers)
            print(f"{rel15:>13} {rel50:>13} {cite15:>13}  {query}", flush=True)


if __name__ == "__main__":
    main()