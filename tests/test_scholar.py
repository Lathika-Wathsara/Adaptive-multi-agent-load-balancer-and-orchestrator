# Created: 2026-11-17
# Created by: Lathika
# Purpose: Tests for src/tools/scholar.py using fake HTTP responses (we dont need network acess for this).

import httpx
import pytest

from src.tools import scholar

CROSSREF = {"message": {"items": [{
    "title": ["Deep learning for MIMO detection"],
    "author": [{"given": "Ada", "family": "Lovelace"}, {"family": "Babbage"}],
    "issued": {"date-parts": [[2021, 5]]},
    "DOI": "10.1000/xyz",
    "URL": "https://doi.org/10.1000/xyz",
    "abstract": "<jats:p>We study detection.</jats:p>",
}]}}

OPENALEX = {"results": [{
    "title": "Hybrid beamforming survey",
    "publication_year": 2022,
    "doi": "https://doi.org/10.1000/abc",
    "authorships": [{"author": {"display_name": "Grace Hopper"}}],
    "abstract_inverted_index": {"A": [0], "survey": [1], "of": [2], "beams": [3]},
}]}

S2 = {"data": [{
    "title": "Attention mechanisms",
    "year": 2017,
    "authors": [{"name": "Ashish Vaswani"}],
    "abstract": "About attention.",
    "externalIds": {"DOI": "10.1000/att"},
    "url": "https://example.org/att",
}]}


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(scholar, "CACHE_DIR", tmp_path)


def test_crossref_parsing(monkeypatch):
    monkeypatch.setattr(scholar, "_http_get", lambda url, params, headers: CROSSREF)
    paper = scholar.search_crossref("deep learning mimo")[0]
    assert paper["title"] == "Deep learning for MIMO detection"
    assert paper["authors"] == ["Lovelace", "Babbage"]
    assert paper["year"] == 2021
    assert paper["abstract"] == "We study detection."
    assert paper["source"] == "crossref"


def test_openalex_parsing_and_abstract(monkeypatch):
    monkeypatch.setattr(scholar, "_http_get", lambda url, params, headers: OPENALEX)
    paper = scholar.search_openalex("hybrid beamforming")[0]
    assert paper["authors"] == ["Hopper"]
    assert paper["doi"] == "10.1000/abc"
    assert paper["abstract"] == "A survey of beams"


def test_semantic_scholar_parsing(monkeypatch):
    monkeypatch.setattr(scholar, "_http_get", lambda url, params, headers: S2)
    paper = scholar.search_semantic_scholar("attention")[0]
    assert paper["authors"] == ["Vaswani"]
    assert paper["doi"] == "10.1000/att"


def test_second_call_comes_from_cache(monkeypatch):
    calls = []

    def fake(url, params, headers):
        calls.append(url)
        return CROSSREF

    monkeypatch.setattr(scholar, "_http_get", fake)
    scholar.search_crossref("same query")
    scholar.search_crossref("same query")
    assert len(calls) == 1
    scholar.search_crossref("another query")
    assert len(calls) == 2


def test_network_failure_becomes_scholar_error(monkeypatch):
    def broken(url, params, headers):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(scholar, "_http_get", broken)
    with pytest.raises(scholar.ScholarError):
        scholar.search_crossref("anything")


def test_failed_lookup_is_not_cached(monkeypatch, tmp_path):
    def broken(url, params, headers):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(scholar, "_http_get", broken)
    with pytest.raises(scholar.ScholarError):
        scholar.search_crossref("anything")
    assert list(tmp_path.glob("*.json")) == []


def test_empty_response_gives_empty_list(monkeypatch):
    monkeypatch.setattr(scholar, "_http_get", lambda url, params, headers: {})
    assert scholar.search_crossref("x") == []
    assert scholar.search_openalex("x") == []
    assert scholar.search_semantic_scholar("x") == []


def test_cited_search_sorts_by_citations_and_limits_the_year(monkeypatch):
    seen = {}

    def fake(url, params, headers):
        seen.update(params)
        return OPENALEX

    monkeypatch.setattr(scholar, "_http_get", fake)
    found = scholar.search_openalex_cited("hybrid beamforming", limit=15, before_year=2017)
    assert seen["sort"] == "cited_by_count:desc"
    assert seen["filter"] == "to_publication_date:2017-12-31"
    assert found[0]["title"] == "Hybrid beamforming survey"


def test_cited_search_without_a_year_has_no_date_filter(monkeypatch):
    seen = {}

    def fake(url, params, headers):
        seen.update(params)
        return OPENALEX

    monkeypatch.setattr(scholar, "_http_get", fake)
    scholar.search_openalex_cited("hybrid beamforming")
    assert "filter" not in seen