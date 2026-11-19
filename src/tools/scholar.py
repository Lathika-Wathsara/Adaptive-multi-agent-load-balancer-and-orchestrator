# Created: 2026-11-17
# Created by: Lathika
# Purpose: Look up papers in Crossref, OpenAlex and Semantic Scholar. Every response
#          is cached on disk (so reruns are free) and requests are spaced out to
#          respect rate limits. Used by the reference verifier and novelty checker.

import hashlib
import json
import os
import re
import time
from pathlib import Path

import httpx
from src.tools import keys

# Cache lives in the shared data folder, never in the repo.
CACHE_DIR = Path(os.environ.get("SCHOLAR_CACHE_DIR", "/data/cache/scholar"))
EMAIL = os.environ.get("SCHOLAR_EMAIL", "")      # sent to Crossref/OpenAlex (polite pool)
#S2_KEY = os.environ.get("S2_API_KEY", "")        # optional Semantic Scholar key

# Minimum seconds between two requests to the same host.
MIN_GAP = {
    "api.crossref.org": 0.2,
    "api.openalex.org": 0.2,
    "api.semanticscholar.org": 1.1,
}
_last_call = {}


class ScholarError(Exception):
    """A lookup failed (network problem, rate limit, bad response)."""


def _http_get(url, params, headers):
    """One GET request with spacing and a few retries on rate limiting."""
    host = url.split("/")[2]
    wait = MIN_GAP.get(host, 0.5) - (time.time() - _last_call.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    for attempt in range(1, 4):
        resp = httpx.get(url, params=params, headers=headers, timeout=30)
        _last_call[host] = time.time()
        if resp.status_code == 429:          # too many requests: wait and retry
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 404:          # nothing found
            return {}
        resp.raise_for_status()
        return resp.json()
    raise ScholarError(f"rate limited by {host}")


def get_json(url, params, headers=None):
    """Return the JSON for a request, from the disk cache when we have it."""
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        data = _http_get(url, params, headers or {})
    except (httpx.HTTPError, ValueError) as err:
        raise ScholarError(f"{url}: {err}") from err
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _candidate(title, authors, year, doi, url, abstract, source):
    """Every source returns papers in this one shape."""
    return {
        "title": title or "",
        "authors": authors,          # family names only
        "year": year,
        "doi": doi or "",
        "url": url or "",
        "abstract": abstract or "",
        "source": source,
    }


def _family(full_name):
    """'Ada Lovelace' -> 'Lovelace'."""
    parts = full_name.split()
    return parts[-1] if parts else ""


def _openalex_abstract(inverted_index):
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inverted_index:
        return ""
    words = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def search_crossref(query, limit=5):
    params = {"query.bibliographic": query, "rows": limit}
    if EMAIL:
        params["mailto"] = EMAIL
    data = get_json("https://api.crossref.org/works", params)
    found = []
    for item in data.get("message", {}).get("items", []):
        title = (item.get("title") or [""])[0]
        authors = [a["family"] for a in item.get("author", []) if a.get("family")]
        date_parts = (item.get("issued", {}).get("date-parts") or [[None]])[0]
        year = date_parts[0] if date_parts else None
        abstract = re.sub(r"<[^>]+>", " ", item.get("abstract", "")).strip()
        found.append(_candidate(title, authors, year, item.get("DOI"),
                                item.get("URL"), abstract, "crossref"))
    return found

def _openalex_search(params):
    """Run one OpenAlex works search and return the papers in the common shape."""
    if EMAIL:
        params["mailto"] = EMAIL
    key = keys.get_key("OPENALEX_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = get_json("https://api.openalex.org/works", params, headers)
    found = []
    for item in data.get("results", []):
        authors = [
            _family(a["author"]["display_name"])
            for a in item.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]
        doi = (item.get("doi") or "").replace("https://doi.org/", "")
        found.append(_candidate(
            item.get("title") or item.get("display_name"), authors,
            item.get("publication_year"), doi, item.get("doi") or item.get("id"),
            _openalex_abstract(item.get("abstract_inverted_index")), "openalex"))
    return found


def search_openalex(query, limit=5):
    return _openalex_search({"search": query, "per-page": limit})


def search_openalex_cited(query, limit=15, before_year=None):
    """Same search, but the most cited papers first. Famous earlier papers show up even
    when their wording differs from the query. before_year keeps only papers published
    up to that year, so newer papers do not fill the list."""
    params = {"search": query, "per-page": limit, "sort": "cited_by_count:desc"}
    if before_year:
        params["filter"] = f"to_publication_date:{before_year}-12-31"
    return _openalex_search(params)

# def search_openalex(query, limit=5):
#     params = {"search": query, "per-page": limit}
#     if EMAIL:
#         params["mailto"] = EMAIL
#     key = keys.get_key("OPENALEX_API_KEY")
#     headers = {"Authorization": f"Bearer {key}"} if key else {}
#     data = get_json("https://api.openalex.org/works", params, headers)
#     #data = get_json("https://api.openalex.org/works", params)
#     found = []
#     for item in data.get("results", []):
#         authors = [
#             _family(a["author"]["display_name"])
#             for a in item.get("authorships", [])
#             if a.get("author", {}).get("display_name")
#         ]
#         doi = (item.get("doi") or "").replace("https://doi.org/", "")
#         found.append(_candidate(
#             item.get("title") or item.get("display_name"), authors,
#             item.get("publication_year"), doi, item.get("doi") or item.get("id"),
#             _openalex_abstract(item.get("abstract_inverted_index")), "openalex"))
#     return found


def search_semantic_scholar(query, limit=5):
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,year,authors,abstract,externalIds,url",
    }
    key = keys.get_key("S2_API_KEY")
    headers = {"x-api-key": key} if key else {}
    data = get_json("https://api.semanticscholar.org/graph/v1/paper/search",
                    params, headers)
    found = []
    for item in data.get("data") or []:
        authors = [_family(a.get("name", "")) for a in item.get("authors", [])]
        doi = (item.get("externalIds") or {}).get("DOI")
        found.append(_candidate(item.get("title"), [a for a in authors if a],
                                item.get("year"), doi, item.get("url"),
                                item.get("abstract"), "semantic_scholar"))
    return found


# Order in which the reference verifier tries them.
SOURCES = [
    ("crossref", search_crossref),
    ("openalex", search_openalex),
    ("semantic_scholar", search_semantic_scholar),
]