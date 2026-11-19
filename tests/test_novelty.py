# Created: 2026-11-17
# Updated: 2026-11-18 (retry loop, human decision, shared pool, quote verification)
# Created by: Lathika
# Purpose: Tests for src/agents/novelty.py with a scripted fake model, a fake paper
#          database and a fake embedder (no network, no Ollama).

import asyncio
import json
import zlib

from src.agents import novelty
from src.common.schemas import AgentResult, ModelResponse, PaperDoc

PAPER = PaperDoc(
    id="p1",
    title="Sensing-centric hybrid beamforming for terahertz systems",
    sections={
        "Abstract": "We propose a hybrid beamformer that raises energy efficiency in terahertz sensing.",
        "Introduction": "Terahertz links need narrow beams.",
        "Numerical Results": "Simulations are run in MATLAB.",
    },
)

BEAM = {"title": "Hybrid beamforming for terahertz energy efficiency", "authors": ["Lee"],
        "year": 2022, "doi": "10.1/beam", "url": "https://doi.org/10.1/beam",
        "abstract": "We design a hybrid beamformer that improves energy efficiency of terahertz "
                    "sensing and communication links.", "source": "fake"}
CAKE = {"title": "A study of sourdough fermentation", "authors": ["Baker"], "year": 2020,
        "doi": "10.1/cake", "url": "https://doi.org/10.1/cake",
        "abstract": "Sourdough starters ferment flour and water into bread with a sour taste "
                    "over many hours.", "source": "fake"}


def db(*papers):
    return [("fake", lambda query, limit=5: list(papers))]


def fake_embed(texts, kind="document"):
    """Bag of words in 64 buckets: texts that share words get similar vectors."""
    out = []
    for text in texts:
        vec = [0.0] * 64
        for word in text.lower().replace(".", " ").split():
            vec[zlib.crc32(word.encode()) % 64] += 1
        out.append(vec)
    return out


QUOTE = "improves energy efficiency of terahertz sensing and communication links"


def support(yes=True, quote=QUOTE):
    return json.dumps({"supports": yes, "quote": quote if yes else ""})


def default_verify(prompt):
    return support(True)


class FakeClient:
    """Answers the claim model from a fixed text, and the large model by kind of question:
    judging (scripted replies, used in order, last repeats), suggesting queries
    ("suggest") and checking one paper ("verify": a function of the prompt)."""

    def __init__(self, claims, replies, suggest="{}", verify=default_verify):
        self.claims = claims
        self.replies = list(replies)
        self.suggest = suggest
        self.verify = verify
        self.models = []
        self.calls = {"compare": 0, "suggest": 0, "verify": 0}

    @property
    def large_calls(self):
        return self.calls["compare"]

    async def call_model(self, model, messages, images=None):
        self.models.append(model)
        prompt = messages[-1]["content"]
        if model == novelty.CLAIM_MODEL:
            text = self.claims
        elif "strict fact checker" in messages[0]["content"]:
            self.calls["verify"] += 1
            text = self.verify(prompt)
        elif "No related papers were found" in prompt:
            self.calls["suggest"] += 1
            text = self.suggest
        else:
            self.calls["compare"] += 1
            text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return ModelResponse(text=text, model_name=model, latency=0.0)


def reply(verdict, confidence=0.9, closest=(1,), reason="because", doubts="", search_next=()):
    return json.dumps({"verdict": verdict, "closest": list(closest), "reason": reason,
                       "confidence": confidence, "doubts": doubts,
                       "search_next": list(search_next)})


CLAIMS = json.dumps({"claims": [
    {"claim": "A hybrid beamformer improves energy efficiency in terahertz sensing.",
     "query": "hybrid beamforming terahertz energy efficiency"}]})


def run(client, sources, **kw):
    return asyncio.run(novelty.check_novelty(PAPER, client, sources=sources,
                                             embed=fake_embed, **kw))


def first(result):
    return json.loads(result.raw_output)[0]


# helpers -------------------------------------------------------------------

def run(client, sources, **kw):
    kw.setdefault("verify_novel", 0)          # most tests do not use the check on "novel"
    return asyncio.run(novelty.check_novelty(PAPER, client, sources=sources,
                                             embed=fake_embed, **kw))

def test_parse_json_ignores_think_blocks_and_chatter():
    text = '<think>hmm {not json}</think> Sure! {"a": 1} Done.'
    assert novelty.parse_json(text) == {"a": 1}
    assert novelty.parse_json("no json here") is None


def test_clean_queries():
    assert novelty.clean_queries(["a b", "  ", "c d", "e f"]) == ["a b", "c d"]
    assert novelty.clean_queries("not a list") == []
    assert novelty.clean_queries(None) == []


def test_extract_claims_retries_once_on_bad_json():
    client = FakeClient("sorry, here you go", [])
    replies = ["not json", CLAIMS]

    async def call_model(model, messages, images=None):
        return ModelResponse(text=replies.pop(0), model_name=model, latency=0.0)

    client.call_model = call_model
    claims = asyncio.run(novelty.extract_claims(PAPER, client))
    assert claims[0]["query"] == "hybrid beamforming terahertz energy efficiency"


def test_extract_claims_limits_count_and_skips_empty():
    many = json.dumps({"claims": [{"claim": f"claim {i}", "query": "q"} for i in range(9)]
                       + [{"claim": ""}]})
    claims = asyncio.run(novelty.extract_claims(PAPER, FakeClient(many, []), max_claims=3))
    assert [c["claim"] for c in claims] == ["claim 0", "claim 1", "claim 2"]


def test_rank_puts_the_related_paper_first():
    ranked = novelty.rank("hybrid beamformer improves energy efficiency terahertz",
                          [CAKE, BEAM], fake_embed, min_score=0)
    assert ranked[0]["title"] == BEAM["title"]
    assert ranked[0]["score"] > ranked[1]["score"]


def test_rank_drops_papers_below_the_similarity_floor():
    ranked = novelty.rank("hybrid beamformer improves energy efficiency terahertz",
                          [CAKE, BEAM], fake_embed, min_score=0.5)
    assert [c["title"] for c in ranked] == [BEAM["title"]]


def test_usable_drops_self_future_and_no_abstract():
    itself = dict(BEAM, title=PAPER.title)
    future = dict(BEAM, title="Newer work", year=2030)
    empty = dict(BEAM, title="No abstract here", abstract="")
    keep = novelty.usable([itself, future, empty, BEAM], PAPER.title, paper_year=2025)
    assert keep == [BEAM]


def test_retrieve_merges_duplicates_and_keeps_the_abstract():
    no_abstract = dict(BEAM, abstract="", source="crossref")
    sources = [("a", lambda q, limit=5: [no_abstract]), ("b", lambda q, limit=5: [BEAM])]
    found = novelty.retrieve(["q"], sources)
    assert len(found) == 1 and found[0]["abstract"]


def test_retrieve_skips_a_failing_source():
    def broken(query, limit=5):
        raise novelty.scholar.ScholarError("rate limited")

    found = novelty.retrieve(["q"], [("a", broken), ("b", lambda q, limit=5: [BEAM])])
    assert found == [BEAM]


def test_retrieve_searches_every_query_and_passes_the_limit():
    seen = []

    def source(query, limit):
        seen.append((query, limit))
        return []

    novelty.retrieve(["keywords", "the claim sentence"], [("a", source)], limit=7)
    assert seen == [("keywords", 7), ("the claim sentence", 7)]


def test_best_matches_lists_cited_first_then_high_scores_without_repeats():
    cited = [{"title": "Cited", "year": 2020, "url": "u1", "score": 0.5}]
    pool = [dict(BEAM, title=f"Paper {i}", score=0.9 - i * 0.01) for i in range(8)]
    pool.append(dict(BEAM, title="Low", score=0.6))
    pool.append(dict(BEAM, title="Cited", score=0.95))
    out = novelty.best_matches(cited, pool)
    assert out[0]["title"] == "Cited"
    assert len(out) == novelty.HUMAN_LIST_MAX
    assert [e["title"] for e in out].count("Cited") == 1
    assert "Low" not in [e["title"] for e in out]


# the decision rules --------------------------------------------------------

def outcome(verdict, conf, search_next=()):
    return {"verdict": verdict, "confidence": conf, "search_next": list(search_next)}


def test_next_step_rules():
    step = novelty.next_step
    assert step(outcome("already_known", 0.9), 2) == "final"
    assert step(outcome("already_known", 0.8), 0) == "human"           # not sure enough
    assert step(outcome("already_known", 0.8, ["x y"]), 1) == "retry"
    assert step(outcome("partly_known", 0.85), 2) == "final"
    assert step(outcome("novel", 0.9, ["x y"]), 1) == "retry"           # try to disprove it
    assert step(outcome("novel", 0.9), 1) == "final"                    # nothing more to try
    assert step(outcome("novel", 0.9, ["x y"]), 0) == "final"           # retries used up
    assert step(outcome("novel", 0.5), 0) == "human"
    assert step(outcome("unchecked", 0.0), 2) == "human"                # no queries to try
    assert step(outcome("unchecked", 0.0, ["x y"]), 1) == "retry"
    assert step(outcome("unchecked", 0.0, ["x y"]), 0) == "human"


# the whole check -----------------------------------------------------------

def test_confident_already_known_is_final_with_url_and_no_human():
    result = run(FakeClient(CLAIMS, [reply("already_known", 0.9)]), db(BEAM, CAKE))
    assert isinstance(result, AgentResult)
    assert [f.type for f in result.findings] == ["novelty_overlap"]
    assert "https://doi.org/10.1/beam" in result.findings[0].evidence
    assert result.needs_human is False
    assert first(result)["decision"] == "final" and first(result)["rounds"] == 0


def test_unsure_already_known_goes_to_a_human_with_best_matches():
    result = run(FakeClient(CLAIMS, [reply("already_known", 0.6, doubts="abstract is vague")]),
                 db(BEAM, CAKE))
    assert [(f.type, f.severity) for f in result.findings] == [("novelty_review", "medium")]
    assert "abstract is vague" in result.findings[0].evidence
    assert "https://doi.org/10.1/beam" in result.findings[0].evidence
    assert result.needs_human is True
    rec = first(result)
    assert rec["decision"] == "human"
    assert rec["best_matches"][0]["url"] == "https://doi.org/10.1/beam"


def test_partly_known_is_medium_and_needs_no_human():
    result = run(FakeClient(CLAIMS, [reply("partly_known", 0.85)]), db(BEAM))
    assert [(f.type, f.severity) for f in result.findings] == [("novelty_partial", "medium")]
    assert result.needs_human is False


def test_novel_has_no_finding_but_still_lists_urls():
    result = run(FakeClient(CLAIMS, [reply("novel", 0.9, closest=())]), db(BEAM, CAKE))
    assert result.findings == []
    rec = first(result)
    assert rec["verdict"] == "novel" and rec["decision"] == "final"
    assert rec["evidence"] and all(e["url"] for e in rec["evidence"])


def test_confident_novel_is_searched_again_before_it_stands():
    replies = [reply("novel", 0.9, search_next=["beamformer sensing links"]),
               reply("novel", 0.9)]
    client = FakeClient(CLAIMS, replies)
    result = run(client, db(BEAM))
    rec = first(result)
    assert rec["rounds"] == 1 and rec["decision"] == "final"
    assert "beamformer sensing links" in rec["queries"]
    assert client.large_calls == 2


def test_retry_with_new_queries_can_find_the_missing_paper():
    def source(query, limit=5):
        return [BEAM] if query == "beamformer sensing links" else [CAKE]

    client = FakeClient(CLAIMS, [reply("already_known", 0.9)],
                        suggest='{"queries": ["beamformer sensing links"]}')
    result = run(client, [("fake", source)])
    rec = first(result)
    assert rec["verdict"] == "already_known" and rec["decision"] == "final"
    assert rec["rounds"] == 1
    assert rec["evidence"][0]["url"] == "https://doi.org/10.1/beam"


def test_retries_are_capped_and_then_the_claim_goes_to_a_human():
    replies = [reply("no_related", search_next=["aa bb cc dd"]),
               reply("no_related", search_next=["ee ff gg hh"]),
               reply("no_related", search_next=["ii jj kk ll"])]
    client = FakeClient(CLAIMS, replies)
    result = run(client, db(BEAM), max_retries=2)
    rec = first(result)
    assert rec["rounds"] == 2 and rec["decision"] == "human"
    assert client.large_calls == 3
    assert result.needs_human is True


def test_repeated_queries_are_not_searched_twice():
    replies = [reply("novel", 0.9, search_next=["hybrid beamforming terahertz energy efficiency"])]
    client = FakeClient(CLAIMS, replies)
    rec = first(run(client, db(BEAM)))
    assert rec["rounds"] == 0 and client.large_calls == 1


def test_bad_closest_numbers_fall_back_to_top_candidates():
    result = run(FakeClient(CLAIMS, [reply("already_known", 2, closest=[0, 7, "x"])]), db(BEAM))
    assert "https://doi.org/10.1/beam" in result.findings[0].evidence
    assert first(result)["confidence"] == 1.0            # clamped into 0-1


def test_no_related_papers_and_no_ideas_goes_to_a_human():
    result = run(FakeClient(CLAIMS, ["{}"]), db())
    rec = first(result)
    assert rec["verdict"] == "unchecked" and rec["decision"] == "human"
    assert rec["evidence"] == [] and rec["best_matches"] == []
    assert result.needs_human is True


def test_unusable_compare_answer_is_unchecked_and_goes_to_a_human():
    result = run(FakeClient(CLAIMS, ["I cannot say"]), db(BEAM))
    rec = first(result)
    assert rec["verdict"] == "unchecked" and rec["decision"] == "human"


def test_uses_small_model_for_claims_and_large_for_comparing():
    client = FakeClient(CLAIMS, [reply("novel", 0.9)])
    run(client, db(BEAM))
    assert client.models[0] == novelty.CLAIM_MODEL
    assert client.models[-1] == novelty.COMPARE_MODEL


def test_no_claims_means_empty_result():
    result = run(FakeClient("{}", []), db(BEAM))
    assert result.findings == [] and json.loads(result.raw_output) == []
    assert result.needs_human is False


def test_paper_year_hides_later_papers():
    later = dict(BEAM, year=2031)
    result = run(FakeClient(CLAIMS, [reply("already_known", 0.9)]), db(later), paper_year=2025)
    assert first(result)["verdict"] == "unchecked"


# verification, shared pool, human list ------------------------------------

def test_known_verdict_gets_a_quote_from_the_abstract():
    client = FakeClient(CLAIMS, [reply("already_known", 0.9)])
    result = run(client, db(BEAM))
    rec = first(result)
    assert rec["decision"] == "final" and client.calls["verify"] == 1
    assert rec["evidence"][0]["quote"] == QUOTE
    assert "Quote:" in result.findings[0].evidence


def test_partly_known_is_verified_too():
    client = FakeClient(CLAIMS, [reply("partly_known", 0.85)])
    run(client, db(BEAM))
    assert client.calls["verify"] == 1


def test_novel_is_not_verified_when_the_check_is_off():
    client = FakeClient(CLAIMS, [reply("novel", 0.9)])
    run(client, db(BEAM), verify_novel=0)
    assert client.calls["verify"] == 0


def test_novel_is_flipped_when_an_abstract_states_the_claim():
    client = FakeClient(CLAIMS, [reply("novel", 0.9)])
    result = run(client, db(BEAM), verify_novel=3)
    rec = first(result)
    assert rec["verdict"] == "partly_known" and rec["decision"] == "final"
    assert rec["evidence"][0]["quote"] == QUOTE and "judge said novel" in rec["reason"]
    assert [f.type for f in result.findings] == ["novelty_partial"]


def test_novel_stays_novel_when_no_abstract_states_it():
    client = FakeClient(CLAIMS, [reply("novel", 0.9)], verify=lambda p: support(False))
    rec = first(run(client, db(BEAM, BEAM_B, CAKE), verify_novel=3))
    assert rec["verdict"] == "novel" and rec["decision"] == "final"


def test_novel_check_tries_at_most_the_allowed_number_of_papers():
    client = FakeClient(CLAIMS, [reply("novel", 0.9, closest=())], verify=lambda p: support(False))
    run(client, db(BEAM, BEAM_B), verify_novel=1)
    assert client.calls["verify"] == 1


def test_novel_check_is_on_by_default():
    assert novelty.NOVEL_CHECK_MAX == 3


BEAM_B = dict(BEAM, title="Beamformers that save energy", doi="10.1/b", url="https://doi.org/10.1/b",
              abstract="A hybrid beamformer improves energy efficiency in terahertz sensing "
                       "links. The first design of a hybrid beamformer for energy efficiency, "
                       "with unique marker sentence zebra.")


def test_wrong_cited_paper_is_replaced_by_one_that_states_the_claim():
    def verify(prompt):
        return support(True, "first design with unique marker sentence zebra") \
            if "zebra" in prompt else support(False)

    client = FakeClient(CLAIMS, [reply("already_known", 0.95, closest=[1])], verify=verify)
    result = run(client, db(BEAM, BEAM_B))
    rec = first(result)
    assert rec["decision"] == "final"
    assert [e["url"] for e in rec["evidence"]] == ["https://doi.org/10.1/b"]


def test_no_paper_states_the_claim_sends_it_to_a_human():
    client = FakeClient(CLAIMS, [reply("already_known", 0.95)], verify=lambda p: support(False))
    result = run(client, db(BEAM))
    rec = first(result)
    assert rec["decision"] == "human" and "verification failed" in rec["doubts"]
    assert [f.type for f in result.findings] == ["novelty_review"]
    assert result.needs_human is True


def test_a_quote_that_is_not_in_the_abstract_is_rejected():
    fake_quote = support(True, "this sentence was invented by the model and is not there")
    client = FakeClient(CLAIMS, [reply("already_known", 0.95)], verify=lambda p: fake_quote)
    assert first(run(client, db(BEAM)))["decision"] == "human"


TWO_CLAIMS = json.dumps({"claims": [
    {"claim": "A hybrid beamformer improves energy efficiency in terahertz sensing.",
     "query": "alpha beamformer"},
    {"claim": "A hybrid beamformer improves energy efficiency in terahertz communication.",
     "query": "zzz unrelated words"}]})


def test_claims_share_one_pool_of_papers():
    def source(query, limit=5):
        return [BEAM] if "alpha" in query else []     # only claim 1's query finds the paper

    result = run(FakeClient(TWO_CLAIMS, [reply("novel", 0.9)]), [("fake", source)])
    records = json.loads(result.raw_output)
    assert [r["verdict"] for r in records] == ["novel", "novel"]      # claim 2 also saw it
    assert all(r["evidence"] for r in records)


def test_whole_paper_queries_are_searched():
    seen = []

    def source(query, limit=5):
        seen.append(query)
        return []

    run(FakeClient(CLAIMS, ["{}"]), [("fake", source)])
    assert PAPER.title in seen


def test_default_is_one_retry():
    assert novelty.MAX_RETRIES == 1


def test_memoize_embed_embeds_each_text_once():
    calls = []

    def counting(texts, kind="document"):
        calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    cached = novelty.memoize_embed(counting)
    assert cached(["aa", "bbb"], "document") == [[2.0], [3.0]]
    assert cached(["aa", "bbb", "c"], "document") == [[2.0], [3.0], [1.0]]
    assert cached(["aa"], "query") == [[2.0]]            # a different kind is separate
    assert calls == [["aa", "bbb"], ["c"], ["aa"]]


def test_human_item_shows_closest_papers_even_below_the_floor():
    result = run(FakeClient(CLAIMS, ["{}"]), db(CAKE))
    rec = first(result)
    assert rec["decision"] == "human" and rec["verdict"] == "unchecked"
    assert rec["best_matches"][0]["url"] == "https://doi.org/10.1/cake"
    assert rec["best_matches"][0]["low_similarity"] is True


def test_default_sources_add_a_citation_sorted_search_up_to_the_paper_year(monkeypatch):
    seen = {}

    def fake(query, limit=15, before_year=None):
        seen["year"] = before_year
        return [BEAM]

    monkeypatch.setattr(novelty.scholar, "search_openalex_cited", fake)
    sources = novelty.default_sources(2017)
    assert [name for name, _ in sources][-1] == "openalex_cited"
    assert sources[-1][1]("q", 15) == [BEAM] and seen["year"] == 2017


def test_another_version_of_the_paper_is_dropped_even_with_a_different_title():
    abstract = PAPER.sections["Abstract"]
    other_version = dict(BEAM, title="A totally different title", doi="10.1/v2",
                         abstract=abstract + " Extra text that only the longer version has.")
    keep = novelty.usable([other_version, BEAM], PAPER.title,
                          paper_abstract=novelty.own_abstract(PAPER))
    assert keep == [BEAM]


def test_own_abstract_uses_the_abstract_section_or_the_first_one():
    assert novelty.own_abstract(PAPER) == PAPER.sections["Abstract"]
    no_name = PaperDoc(id="x", title="T", sections={"Intro": "first text", "Other": "second"})
    assert novelty.own_abstract(no_name) == "first text"

