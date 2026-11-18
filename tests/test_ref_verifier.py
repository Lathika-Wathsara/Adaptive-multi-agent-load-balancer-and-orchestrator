# Created: 2026-11-17
# Created by: Lathika
# Purpose: Tests for src/agents/ref_verifier.py with a fake paper database (no network).

from src.agents import ref_verifier as rv
from src.common.schemas import AgentResult
from src.tools.scholar import ScholarError

REAL_TITLE = "Deep learning for channel estimation in massive MIMO systems"
REAL = {"title": REAL_TITLE, "authors": ["Smith", "Lee"], "year": 2021,
        "doi": "10.1/x", "url": "https://doi.org/10.1/x", "abstract": "", "source": "fake"}

REAL_REF = ('[1] J. Smith and K. Lee, "Deep learning for channel estimation in massive '
            'MIMO systems," IEEE Trans. Wireless Commun., vol. 20, no. 3, 2021.')


def db(*papers):
    """A fake source that returns the same papers for any query."""
    return [("fake", lambda query: list(papers))]


def test_real_reference_exists():
    result = rv.verify_reference(REAL_REF, sources=db(REAL))
    assert result["status"] == "exists"
    assert result["title_score"] >= 90


def test_invented_reference_not_found():
    fake = ('[2] H. Lindqvist and R. Abara, "Quantum Spectral Hashing for Terahertz '
            'Radar Co-design," IEEE Trans. Signal Process., vol. 71, 2023.')
    result = rv.verify_reference(fake, sources=db(REAL))
    assert result["status"] == "not_found"


def test_nothing_returned_is_not_found():
    result = rv.verify_reference(REAL_REF, sources=db())
    assert result["status"] == "not_found"


def test_real_title_with_wrong_authors_and_year_is_mismatch():
    ref = ('[3] P. Novak and Q. Zhou, "Deep learning for channel estimation in massive '
           'MIMO systems," IEEE Trans. Wireless Commun., 2015.')
    result = rv.verify_reference(ref, sources=db(REAL))
    assert result["status"] == "mismatch"
    assert "authors differ" in result["reason"]
    assert "year differs" in result["reason"]


def test_year_slack_allows_preprint_year():
    ref = REAL_REF.replace("2021", "2019")
    assert rv.verify_reference(ref, sources=db(REAL))["status"] == "exists"


def test_et_al_reference_still_exists():
    ref = ('[4] J. Smith et al., "Deep learning for channel estimation in massive '
           'MIMO systems," 2021.')
    assert rv.verify_reference(ref, sources=db(REAL))["status"] == "exists"


AMBIGUOUS_REF = ('[5] J. Smith and K. Lee, "Deep learning for channel prediction in '
                 'massive MIMO," IEEE Trans. Wireless Commun., 2021.')


def test_ambiguous_title_uses_judge():
    first = rv.verify_reference(AMBIGUOUS_REF, sources=db(REAL))
    assert rv.TITLE_MAYBE <= first["title_score"] < rv.TITLE_SAME
    assert first["status"] == "mismatch"                      # no judge given
    yes = rv.verify_reference(AMBIGUOUS_REF, sources=db(REAL), judge=lambda r, c: True)
    no = rv.verify_reference(AMBIGUOUS_REF, sources=db(REAL), judge=lambda r, c: False)
    assert yes["status"] == "exists"
    assert no["status"] == "not_found"


def test_all_sources_failing_is_unchecked_not_fake():
    def broken(query):
        raise ScholarError("no network")

    result = rv.verify_reference(REAL_REF, sources=[("a", broken), ("b", broken)])
    assert result["status"] == "unchecked"


def test_one_failing_source_is_skipped():
    def broken(query):
        raise ScholarError("rate limited")

    sources = [("a", broken), ("b", lambda q: [REAL])]
    assert rv.verify_reference(REAL_REF, sources=sources)["status"] == "exists"


def test_stops_after_a_good_match():
    calls = []

    def first(query):
        calls.append("first")
        return [REAL]

    def second(query):
        calls.append("second")
        return []

    rv.verify_reference(REAL_REF, sources=[("a", first), ("b", second)])
    assert calls == ["first"]


def test_guess_title_styles():
    assert rv.guess_title(REAL_REF) == REAL_TITLE
    apa = "Smith, J., & Lee, K. (2021). A study of wireless sensing. Journal of Radio, 5(2), 1-10."
    assert rv.guess_title(apa) == "A study of wireless sensing"
    plain = "[7] Smith J. Lee K. Terahertz beam training with side information. Proc ICC 2022"
    assert "Terahertz beam training" in rv.guess_title(plain)


def test_three_fakes_flagged_and_real_passes():
    refs = [REAL_REF,
            '[F1] A. Moreno, "Adaptive Terahertz Beam Alignment via Federated Spectral Hashing," 2023.',
            '[F2] S. Varga, "Sensing-Aware Hybrid Precoding with Learned Phase Quantizers," 2023.',
            '[F3] H. Lindqvist, "Radar-Communication Co-design under Hardware Impairments," 2023.']
    checks = rv.verify_references(refs, sources=db(REAL))
    assert [c["status"] for c in checks] == ["exists", "not_found", "not_found", "not_found"]


def test_agent_result_findings():
    refs = [REAL_REF,
            '[F1] A. Moreno, "Adaptive Terahertz Beam Alignment via Federated Spectral Hashing," 2023.']
    result = rv.to_agent_result(rv.verify_references(refs, sources=db(REAL)))
    assert isinstance(result, AgentResult)
    assert [f.type for f in result.findings] == ["fake_reference"]
    assert result.needs_human is True
    assert 0 <= result.findings[0].confidence <= 1


def test_clean_paper_has_no_findings():
    result = rv.to_agent_result(rv.verify_references([REAL_REF], sources=db(REAL)))
    assert result.findings == []
    assert result.needs_human is False

    
def test_prefers_the_record_that_matches_year_and_authors():
    reprint = dict(REAL, year=1999, authors=["Someone"])
    assert rv.verify_reference(REAL_REF, sources=db(reprint, REAL))["status"] == "exists"