# Created: 2026-11-17
# Created by: Lathika
# Purpose: Turn texts into vectors with nomic-embed-text through the local Ollama
#          server. Used by the novelty checker to rank related work against a claim.

import os

import httpx

OLLAMA_URL = "http://" + os.environ.get("OLLAMA_HOST", "127.0.0.1:11500")
MODEL = "nomic-embed-text"


def embed(texts, kind="document"):
    """Return one vector per text. kind is "query" (the claim) or "document" (a paper).

    nomic-embed-text expects a short prefix that says which kind of text it is.
    """
    prefix = "search_query: " if kind == "query" else "search_document: "
    resp = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": MODEL, "input": [prefix + t for t in texts]},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]