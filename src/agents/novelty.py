# Created: 2026-11-17
# Updated: 2026-11-18 (judge can ask for more searching; uncertain claims go to a human;
#          claims share one pool of papers; a known claim must be backed by a quote)
# Created by: Lathika
# Purpose: Check whether the claims a paper makes are already known. A small model
#          extracts the claims, src/tools/scholar.py finds related papers, the
#          embedder ranks them, and the large model compares each claim with the
#          closest abstracts. If the judge is unsure it suggests new search queries
#          and we look again (at most MAX_RETRIES times). All claims of a paper share
#          one pool of papers. An already_known or partly_known verdict is only
#          accepted when a retrieved abstract really states the claim (quoted);
#          otherwise, and when the judge stays unsure, the claim goes to a human
#          with its best matching papers.
#          Every claim comes back with the URLs its verdict is based on.
#          Verdicts: novel, partly_known, already_known, unchecked.
#          Decisions: final (the verdict stands) or human (needs a person to look).

import asyncio
import json
import re

import numpy as np
from rapidfuzz import fuzz

from src.agents.ref_verifier import normalize
from src.common.schemas import AgentResult, Finding
from src.tools import embedder, scholar
from functools import partial

CLAIM_MODEL = "qwen3:4b-instruct"   # small model: pulls the claims out of the paper
COMPARE_MODEL = "qwen3:14b"         # large model: compares a claim with related work
MAX_CLAIMS = 5
TOP_K = 4                           # abstracts shown to the large model per claim
MIN_ABSTRACT = 50                   # shorter abstracts cannot be compared
RETRIEVE_LIMIT = 15                 # papers asked from each source per query
MIN_SIMILARITY = 0.70               # below this cosine a paper is not worth comparing
SAME_PAPER = 90                     # title score at or above this: it is the paper itself
SAME_ABSTRACT = 90                  # abstract match that also means "this is the paper itself"
VERDICTS = ("novel", "partly_known", "already_known")

MAX_RETRIES = 1                     # extra search rounds the judge may ask for
KNOWN_CONF = 0.85                   # already_known at this confidence stands without a human
OTHER_CONF = 0.80                   # same for partly_known, and for novel after retries
HUMAN_LIST_SCORE = 0.75             # pool papers at this cosine are listed for the human
HUMAN_LIST_MAX = 5
LOW_LIST_MAX = 3                    # closest papers shown when nothing reaches the floor
QUOTE_MATCH = 85                    # how well a quote must appear in the abstract it cites
NOVEL_CHECK_MAX = 3                 # papers the strict check tries on a "novel" verdict (0 = off)


def paper_text(paper):
    """Title plus the abstract, introduction and conclusion (or all sections if none)."""
    wanted = ("abstract", "introduction", "conclusion")
    parts = [text for name, text in paper.sections.items()
             if any(w in name.lower() for w in wanted)]
    if not parts:
        parts = list(paper.sections.values())
    return paper.title + "\n\n" + "\n\n".join(p[:2000] for p in parts)


def parse_json(text):
    """The first JSON object inside a model reply (ignores <think> blocks), or None."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


async def ask_json(client, model, system, user):
    """Ask a model for a JSON object; one retry if the reply is not valid JSON."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(2):
        reply = await client.call_model(model=model, messages=messages)
        data = parse_json(reply.text)
        if isinstance(data, dict):
            return data
        messages = messages + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": "That was not valid JSON. Reply with the JSON object only."},
        ]
    return None


def clean_queries(raw):
    """Up to two non-empty query strings from whatever the model returned."""
    if not isinstance(raw, list):
        return []
    return [str(q).strip() for q in raw if str(q).strip()][:2]


async def extract_claims(paper, client, model=CLAIM_MODEL, max_claims=MAX_CLAIMS):
    """Return [{"claim": ..., "query": ...}]: what the paper says is new, and a
    short keyword query to search for related work."""
    system = ("You read research papers and list what each paper claims as its own "
              "contribution (new method, new result, new bound, improved performance).")
    user = (
        f"Paper:\n{paper_text(paper)}\n\n"
        f"List at most {max_claims} distinct novelty claims. Only claims about the idea itself "
        "count: the method, architecture, formulation, algorithm, metric or theoretical result. "
        "Skip claims that only report numbers from the paper's own simulations or experiments "
        "(for example 'achieves 40% improvement'). For each give the claim as "
        "one self-contained sentence and a search query of 4 to 8 keywords that would "
        'find prior work on it. Reply with JSON only: {"claims": [{"claim": "...", "query": "..."}]}'
        "one self-contained sentence and a search query of 4 to 8 keywords that would "
        'find prior work on it. Reply with JSON only: {"claims": [{"claim": "...", "query": "..."}]}'
    )
    data = await ask_json(client, model, system, user) or {}
    claims = []
    for item in data.get("claims", []):
        if isinstance(item, dict) and item.get("claim"):
            claim = str(item["claim"]).strip()
            claims.append({"claim": claim, "query": str(item.get("query") or claim).strip()})
    return claims[:max_claims]

def default_sources(paper_year=None):
    """The normal sources plus a citation-sorted OpenAlex search (papers up to paper_year),
    so well-known earlier papers are found even when their wording differs from the query."""
    cited = partial(scholar.search_openalex_cited, before_year=paper_year)
    return scholar.SOURCES + [("openalex_cited", cited)]

def retrieve(queries, sources=None, limit=RETRIEVE_LIMIT):
    """Search every source with every query and merge duplicates (keep the record
    with the longest abstract)."""
    sources = scholar.SOURCES if sources is None else sources
    best = {}
    for query in queries:
        for _name, search in sources:
            try:
                found = search(query, limit=limit)
            except scholar.ScholarError:
                continue
            for cand in found:
                key = normalize(cand["title"])
                if key and (key not in best or len(cand["abstract"]) > len(best[key]["abstract"])):
                    best[key] = cand
    return list(best.values())


def own_abstract(paper):
    """The paper's own abstract (the first section if none is called abstract)."""
    for name, text in paper.sections.items():
        if "abstract" in name.lower():
            return text
    return next(iter(paper.sections.values()), "")


def is_same_paper(cand, paper_title, paper_abstract=""):
    """True if a retrieved record is the paper under review (another version of it):
    a near-identical title, or the start of its abstract appears in the paper's abstract."""
    if paper_title and fuzz.ratio(normalize(cand["title"]), normalize(paper_title)) >= SAME_PAPER:
        return True
    if len(paper_abstract) >= MIN_ABSTRACT:
        start = normalize(cand["abstract"])[:300]
        return fuzz.partial_ratio(start, normalize(paper_abstract)[:1500]) >= SAME_ABSTRACT
    return False


def usable(cands, paper_title, paper_year=None, paper_abstract=""):
    """Drop papers that cannot count as prior work: no abstract, the paper itself
    (any version), or published after the paper."""
    keep = []
    for c in cands:
        if len(c["abstract"]) < MIN_ABSTRACT:
            continue
        if is_same_paper(c, paper_title, paper_abstract):
            continue
        if paper_year and c["year"] and c["year"] > paper_year:
            continue
        keep.append(c)
    return keep


def rank(claim, cands, embed, top_k=TOP_K, min_score=MIN_SIMILARITY):
    """The top_k candidates most similar to the claim (cosine "score" at least
    min_score), best first."""
    if not cands:
        return []
    query = np.array(embed([claim], "query")[0], dtype=float)
    docs = np.array(embed([f"{c['title']}. {c['abstract'][:1500]}" for c in cands],
                          "document"), dtype=float)
    sims = docs @ query / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query) + 1e-9)
    order = [i for i in np.argsort(-sims) if sims[i] >= min_score][:top_k]
    return [dict(cands[i], score=round(float(sims[i]), 3)) for i in order]


def paper_url(cand):
    return cand["url"] or (f"https://doi.org/{cand['doi']}" if cand["doi"] else "")


def evidence_list(cands):
    return [{"title": c["title"], "year": c["year"], "url": paper_url(c),
             "score": c.get("score")} for c in cands]


async def suggest_queries(claim, client, model=COMPARE_MODEL):
    """When nothing related was found: ask the model which standard terms the field
    uses for this idea. The queries are only used for searching, never as evidence."""
    system = "You help find prior work for a research claim."
    user = (
        f"Claim: {claim}\n\nNo related papers were found with the claim's own wording. "
        "Give up to 2 search queries of 4 to 8 keywords that use the standard terms the "
        'research field uses for this idea. Reply with JSON only: {"queries": ["...", "..."]}'
    )
    data = await ask_json(client, model, system, user) or {}
    return clean_queries(data.get("queries"))


async def compare(claim, ranked, client, model=COMPARE_MODEL):
    """Ask the large model how the claim relates to the ranked abstracts."""
    listing = "\n\n".join(
        f"[{i}] {c['title']} ({c['year']})\n{c['abstract'][:900]}"
        for i, c in enumerate(ranked, 1))
    system = "You are a careful peer reviewer judging whether a claim is new."
    user = (
        f"Claim from a paper under review:\n{claim}\n\nRelated papers:\n\n{listing}\n\n"
        "Decide: no_related (none of these papers is about the same topic, so nothing can "
        "be said), novel (the papers are on the same topic but none does what the claim "
        "does), partly_known (some of the claim is already in these papers), or "
        "already_known (one of these papers clearly reports the same result or method). "
        "If unsure, do not choose already_known. "
        'Reply with JSON only: {"verdict": "...", "closest": [numbers of the papers you '
        'relied on], "reason": "one or two sentences", "confidence": 0.0 to 1.0, '
        '"doubts": "what you are unsure about, or empty", '
        '"search_next": ["up to 2 new keyword queries that could find related work these '
        'papers hint exists but is missing"]}'
    )
    data = await ask_json(client, model, system, user) or {}
    verdict = str(data.get("verdict", "")).strip().lower()
    extra = {"doubts": str(data.get("doubts") or "").strip(),
             "search_next": clean_queries(data.get("search_next"))}
    if verdict == "no_related":
        return {"verdict": "unchecked", "confidence": 0.0, "evidence": evidence_list(ranked[:2]),
                "cited": [], "reason": "the model found none of the retrieved papers on the same topic",
                **extra}
    if verdict not in VERDICTS:
        return {"verdict": "unchecked", "reason": "the model gave no usable answer",
                "confidence": 0.0, "evidence": evidence_list(ranked[:2]), "cited": [], **extra}

    picked = []
    for n in data.get("closest") or []:
        try:
            number = int(n)
        except (ValueError, TypeError):
            continue
        if 1 <= number <= len(ranked):            # ignore numbers that are not in the list
            picked.append(ranked[number - 1])
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
    except (ValueError, TypeError):
        confidence = 0.5
    cited = picked or ranked[:2]
    return {"verdict": verdict, "reason": str(data.get("reason", "")).strip(),
            "confidence": confidence, "evidence": evidence_list(cited), "cited": cited,
            **extra}


async def supports(claim, cand, client, model=COMPARE_MODEL):
    """Strict check of one paper. Returns the sentence of its abstract that states the
    claim, or "" if it does not (a quote that is not in the abstract does not count)."""
    system = "You are a strict fact checker."
    user = (
        f"Claim: {claim}\n\nPaper: {cand['title']} ({cand['year']})\n"
        f"Abstract: {cand['abstract'][:1500]}\n\n"
        "Does this abstract state the claim, or an important part of it, or clearly imply "
        "it? Being on the same topic is not enough. Reply with JSON only: "
        '{"supports": true or false, "quote": "the sentence from the abstract that shows it, or empty"}'
    )
    data = await ask_json(client, model, system, user) or {}
    quote = str(data.get("quote") or "").strip()
    said_yes = data.get("supports") in (True, "true", "True")
    found = bool(quote) and fuzz.partial_ratio(normalize(quote), normalize(cand["abstract"])) >= QUOTE_MATCH
    return quote if said_yes and found else ""


async def find_support(claim, cited, others, client, model=COMPARE_MODEL):
    """Check the papers the judge cited first, then the other top papers. Returns
    [(paper, quote)] for the first paper whose abstract states the claim, or []."""
    for cand in cited + others:
        quote = await supports(claim, cand, client, model)
        if quote:
            return [(cand, quote)]
    return []


def next_step(outcome, retries_left):
    """Decide what happens after one judging round: "final", "retry" or "human".

    Plain rules, not the model's opinion: a confident already_known or partly_known
    stands; anything else is searched again while retries remain and the judge has
    new queries; a confident novel only stands once searching has been tried.
    """
    verdict, conf = outcome["verdict"], outcome["confidence"]
    if verdict == "already_known" and conf >= KNOWN_CONF:
        return "final"
    if verdict == "partly_known" and conf >= OTHER_CONF:
        return "final"
    if retries_left > 0 and outcome["search_next"]:
        return "retry"
    if verdict == "novel" and conf >= OTHER_CONF:
        return "final"
    return "human"


def best_matches(cited, ranked_all):
    """The papers a human should look at: the ones the judge cited, then any pool
    paper scoring at least HUMAN_LIST_SCORE, best first, at most HUMAN_LIST_MAX."""
    out = list(cited)
    seen = {e["title"] for e in out}
    for cand in ranked_all:
        if cand["score"] >= HUMAN_LIST_SCORE and cand["title"] not in seen:
            out.extend(evidence_list([cand]))
            seen.add(cand["title"])
    return out[:HUMAN_LIST_MAX]


def add_to_pool(pool, cands):
    """Put papers into the shared pool (one entry per title, keep the longest abstract)."""
    for cand in cands:
        key = normalize(cand["title"])
        if key and (key not in pool or len(cand["abstract"]) > len(pool[key]["abstract"])):
            pool[key] = cand


def memoize_embed(embed):
    """Wrap an embed function so the same text is only embedded once."""
    memo = {}

    def cached(texts, kind="document"):
        missing = [t for t in dict.fromkeys(texts) if (kind, t) not in memo]
        if missing:
            for text, vec in zip(missing, embed(missing, kind)):
                memo[(kind, text)] = vec
        return [memo[(kind, t)] for t in texts]

    return cached

async def check_claim(item, pool, paper, client, sources, embed, paper_year,
                      compare_model, top_k, max_retries, verify_novel=0):
    """Judge one claim against the shared pool, searching again when the judge asks,
    and return its record. New papers found on a retry are added to the pool."""
    tried = [item["query"], item["claim"]]            # already searched for all claims
    rounds = 0
    while True:
        candidates = usable(list(pool.values()), paper.title, paper_year, own_abstract(paper))
        ranked_all = await asyncio.to_thread(rank, item["claim"], candidates, embed,
                                             len(candidates), MIN_SIMILARITY)
        if ranked_all:
            outcome = await compare(item["claim"], ranked_all[:top_k], client, compare_model)
        else:
            outcome = {"verdict": "unchecked", "confidence": 0.0, "evidence": [], "cited": [],
                       "doubts": "", "reason": "no related papers with abstracts were found close enough",
                       "search_next": await suggest_queries(item["claim"], client, compare_model)}
        outcome["search_next"] = [q for q in outcome["search_next"] if q not in tried]
        step = next_step(outcome, max_retries - rounds)
        if step != "retry":
            break
        tried.extend(outcome["search_next"])
        rounds += 1
        add_to_pool(pool, await asyncio.to_thread(retrieve, outcome["search_next"], sources))

    cited_titles = {c["title"] for c in outcome["cited"]}
    others = [c for c in ranked_all[:top_k] if c["title"] not in cited_titles]
    if step == "final" and outcome["verdict"] in ("already_known", "partly_known"):
        backed = await find_support(item["claim"], outcome["cited"], others, client, compare_model)
        if backed:
            outcome["evidence"] = [dict(evidence_list([c])[0], quote=q) for c, q in backed]
        else:
            step = "human"
            outcome["doubts"] = (outcome["doubts"] + " " if outcome["doubts"] else "") + \
                "No retrieved abstract states the claim (verification failed)."
    elif step == "final" and outcome["verdict"] == "novel" and verify_novel > 0:
        # A judge that says "novel" can be wrong: the cited papers (then the others) get the
        # strict check too, at most verify_novel papers. A quote-backed hit makes it partly_known.
        shown = (outcome["cited"] + others)[:verify_novel]
        backed = await find_support(item["claim"], shown, [], client, compare_model)
        if backed:
            outcome["verdict"] = "partly_known"
            outcome["reason"] = "The judge said novel, but a retrieved abstract states it. " + outcome["reason"]
            outcome["evidence"] = [dict(evidence_list([c])[0], quote=q) for c, q in backed]

    record = {"claim": item["claim"], "query": item["query"], "verdict": outcome["verdict"],
              "reason": outcome["reason"], "confidence": outcome["confidence"],
              "doubts": outcome["doubts"], "evidence": outcome["evidence"],
              "decision": step, "rounds": rounds, "queries": tried}
    if step == "human":
        matches = best_matches(outcome["evidence"], ranked_all)
        if not matches:                               # nothing reached the floor: show the closest
            closest = await asyncio.to_thread(rank, item["claim"], candidates, embed, LOW_LIST_MAX, 0.0)
            matches = [dict(e, low_similarity=True) for e in evidence_list(closest)]
        record["best_matches"] = matches
    return record


def paper_queries(paper):
    """Queries about the whole paper: its title, and the start of its abstract."""
    queries = [paper.title, paper_text(paper)[:300]]
    return [q for i, q in enumerate(queries) if q.strip() and q not in queries[:i]]


async def check_novelty(paper, client, sources=None, embed=None, paper_year=None,
                        claim_model=CLAIM_MODEL, compare_model=COMPARE_MODEL,
                        max_claims=MAX_CLAIMS, top_k=TOP_K, max_retries=MAX_RETRIES,
                        verify_novel=NOVEL_CHECK_MAX):
    """Run the whole novelty check on a PaperDoc and return an AgentResult.

    client: anything with async call_model(model=..., messages=...) such as LLMClient.
    sources / embed: replaceable for tests; default is the real APIs and Ollama.
    paper_year: papers published after this year are not counted as prior work.
    max_retries: how many extra search rounds the judge may ask for per claim.
    verify_novel: how many papers the strict quote check tries on a "novel" verdict
                  (costs one large-model call each; 0 switches it off).
    """
    embed = memoize_embed(embedder.embed if embed is None else embed)
    sources = default_sources(paper_year) if sources is None else sources
    claims = await extract_claims(paper, client, claim_model, max_claims)

    # One pool for the whole paper: a paper found for one claim can be used by the others.
    pool = {}
    add_to_pool(pool, await asyncio.to_thread(retrieve, paper_queries(paper), sources))
    for item in claims:
        add_to_pool(pool, await asyncio.to_thread(retrieve, [item["query"], item["claim"]], sources))

    results = []
    for item in claims:
        results.append(await check_claim(item, pool, paper, client, sources, embed, paper_year,
                                         compare_model, top_k, max_retries, verify_novel))
    return to_agent_result(results)


def urls_of(evidence):
    return " ; ".join(e["url"] for e in evidence if e["url"])


def to_agent_result(results):
    """Turn the per-claim records into the shared AgentResult."""
    findings = []
    for r in results:
        if r["decision"] == "human":
            findings.append(Finding(
                type="novelty_review", severity="medium", confidence=r["confidence"],
                evidence=(f"{r['claim'][:150]} - needs a human. Judge said {r['verdict']}: "
                          f"{r['reason']} Doubts: {r['doubts'] or 'none given'}. "
                          f"Best matches: {urls_of(r['best_matches'])}")))
        elif r["verdict"] in ("already_known", "partly_known"):
            known = r["verdict"] == "already_known"
            quote = r["evidence"][0].get("quote", "") if r["evidence"] else ""
            findings.append(Finding(
                type="novelty_overlap" if known else "novelty_partial",
                severity="high" if known else "medium", confidence=r["confidence"],
                evidence=(f"{r['claim'][:150]} - {r['reason']} Related work: {urls_of(r['evidence'])}"
                          + (f" Quote: {quote[:200]}" if quote else ""))))
    return AgentResult(
        findings=findings,
        needs_human=any(r["decision"] == "human" for r in results),
        raw_output=json.dumps(results),
    )