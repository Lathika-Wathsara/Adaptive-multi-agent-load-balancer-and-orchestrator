# File: tests/test_vision_checker.py
# Created: 2026-11-19
# Created by: Lathika
# Purpose: Tests for src/agents/vision_checker.py with a scripted fake vision model (no Ollama).

import asyncio
import base64
import json
import re

import pytest

from src.agents import vision_checker as vc
from src.common.schemas import AgentResult, ModelResponse, PaperDoc
from src.ingest.models import FigureInfo, ParsedPaper, TableInfo

IMAGE_BYTES = b"\x89PNG fake image bytes"
TEXT = ("Figure 1 compares four methods. The proposed scheme reaches 91.3% accuracy at 20 dB. "
        "The baseline reaches only 71.4% accuracy. All results are averaged over 5 runs.")
TABLE = "Method | Accuracy | F1\nBaseline | 71.4 | 69.8\nProposed | 91.3 | 90.2"


def claims_reply(*claims):
    """A model reply to the claims request: claims are (quote, question, claimed, unit)."""
    items = [{"quote": q, "question": question, "claimed": n, "unit": u}
             for q, question, n, u in claims]
    return "<think>hmm</think>```json\n" + json.dumps({"claims": items}) + "\n```"


def readings_reply(*readings):
    """Figure readings are (value, printed); table readings are (cell, row)."""
    items = []
    for i, (a, b) in enumerate(readings, 1):
        if isinstance(a, str):
            items.append({"n": i, "cell": a, "row": b})
        else:
            items.append({"n": i, "value": a, "printed": b, "note": "looked at the bar"})
    return json.dumps({"readings": items})


CLAIM_PROPOSED = ("The proposed scheme reaches 91.3% accuracy at 20 dB", "Accuracy of the proposed scheme", 91.3, "%")
CLAIM_BASELINE = ("The baseline reaches only 71.4% accuracy", "Accuracy of the baseline", 71.4, "%")


class FakeClient:
    """Answers from scripts: claims[item id] and readings[item id] (a reply text or a list of them).
    The item id is the first "Figure N" or "Table N" in the prompt."""

    def __init__(self, claims=None, readings=None):
        self.scripts = {"claims": self._queues(claims), "readings": self._queues(readings)}
        self.calls = []                                   # (kind, prompt, images, model)

    @staticmethod
    def _queues(replies):
        return {k: (list(v) if isinstance(v, list) else [v]) for k, v in (replies or {}).items()}

    async def call_model(self, model, messages, images=None):
        prompt = messages[1]["content"]                   # the first user message, even on a retry
        kind = "claims" if "List the numeric statements" in prompt else "readings"
        item_id = re.search(r"(?:Figure|Table) \d+", prompt).group()
        self.calls.append((kind, prompt, images, model))
        queue = self.scripts[kind].get(item_id)
        text = (queue.pop(0) if len(queue) > 1 else queue[0]) if queue else '{"claims": [], "readings": []}'
        return ModelResponse(text=text, model_name=model, latency=0.0)

    def kinds(self):
        return [c[0] for c in self.calls]


def figure_paper(tmp_path, paragraphs=(TEXT,), fallback=False, write_image=True):
    image = tmp_path / "fig_1.png"
    if write_image:
        image.write_bytes(IMAGE_BYTES)
    fig = FigureInfo(id="Figure 1", path=str(image), page=1, caption="Figure 1: Accuracy.",
                     citing_paragraphs=list(paragraphs), fallback_crop=fallback)
    return ParsedPaper(id="p1", figure_meta=[fig])


def table_paper(paragraphs=(TEXT.replace("Figure 1", "Table 1"),), text=TABLE):
    tab = TableInfo(id="Table 1", page=1, caption="Table 1: Results.", text=text,
                    citing_paragraphs=list(paragraphs))
    return ParsedPaper(id="p2", table_meta=[tab])


def run(client, paper, **kw):
    return asyncio.run(vc.check_vision(paper, client, **kw))


# ---------------------------------------------------------------------------- helpers
def test_numbers_in_text():
    assert vc.numbers_in("91.3% at 20 dB, -3 dB and 1,000 samples (5-10 runs)") == [91.3, 20, -3, 1000, 5, 10]
    assert vc.numbers_in("Fig2 has x2 and 3.") == [3.0]
    assert vc.numbers_in("a −3.5 b") == [-3.5]


def test_to_number_accepts_numbers_and_strings_only():
    assert vc.to_number(5) == 5.0 and vc.to_number("91.3%") == 91.3
    assert vc.to_number(None) is None and vc.to_number(True) is None and vc.to_number("n/a") is None
    assert vc.to_number(float("nan")) is None


def test_rescale_picks_the_scale_of_the_claim():
    assert vc.rescale(0.913, 91.3) == pytest.approx(91.3)
    assert vc.rescale(91.3, 0.913) == pytest.approx(0.913)
    assert vc.rescale(87.0, 91.3) == 87.0


def test_read_json_ignores_think_block_and_fences_and_backslashes():
    assert vc.read_json("<think>{\"a\": 1}</think>```json\n{\"b\": 2}\n```") == {"b": 2}
    assert vc.read_json(r'{"note": "uses \alpha"}')["note"] == r"uses \alpha"
    assert vc.read_json("no json") is None and vc.read_json("[1, 2]") is None


def test_load_image_is_base64_and_resolved_against_the_root(tmp_path):
    (tmp_path / "figs").mkdir()
    (tmp_path / "figs" / "a.png").write_bytes(IMAGE_BYTES)
    got = vc.load_image("figs/a.png", tmp_path)
    assert base64.b64decode(got) == IMAGE_BYTES
    assert vc.load_image(str(tmp_path / "figs" / "a.png"), "/nowhere") == got
    assert vc.load_image("figs/missing.png", tmp_path) is None and vc.load_image("", tmp_path) is None


# ---------------------------------------------------------------------- what is checked
def test_items_are_figures_then_tables_and_a_plain_paperdoc_has_none(tmp_path):
    paper = figure_paper(tmp_path)
    paper.table_meta = table_paper().table_meta
    assert [i["id"] for i in vc.build_items(paper)] == ["Figure 1", "Table 1"]
    assert [i["id"] for i in vc.build_items(paper, ids=["Table 1"])] == ["Table 1"]
    assert len(vc.build_items(paper, max_items=1)) == 1
    assert vc.build_items(PaperDoc(id="x", figures=["a.png"])) == []


def test_item_without_citing_paragraph_or_image_or_text_is_skipped(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for paper in (figure_paper(tmp_path / "a", paragraphs=()), figure_paper(tmp_path / "b", write_image=False),
                  table_paper(paragraphs=()), table_paper(text="")):
        client = FakeClient()
        result = run(client, paper)
        records = json.loads(result.raw_output)
        assert records[0]["status"] == "skipped" and records[0]["reason"]
        assert result.findings == [] and client.calls == []


# ----------------------------------------------------------------------- the claims
def test_claim_with_invented_quote_or_missing_number_is_dropped():
    raw = [{"quote": "The proposed scheme reaches 91.3% accuracy at 20 dB", "question": "q", "claimed": 91.3, "unit": "%"},
           {"quote": "The method wins by a huge margin of 12.5 points", "question": "q", "claimed": 12.5, "unit": ""},
           {"quote": "The baseline reaches only 71.4% accuracy", "question": "q", "claimed": 80, "unit": "%"},
           {"quote": "", "question": "q", "claimed": 5, "unit": ""}, "junk"]
    good, dropped = vc.valid_claims(raw, [TEXT])
    assert [c["claimed"] for c in good] == [91.3]
    assert [d["dropped"] for d in dropped] == ["quote not found in the text",
                                               "claimed number not in the quote", "incomplete"]


def test_at_most_max_claims_are_kept():
    text = " ".join(f"Method {i} reaches {i}0.5 percent." for i in range(1, 8))
    raw = [{"quote": f"Method {i} reaches {i}0.5 percent", "question": "q", "claimed": i * 10 + 0.5}
           for i in range(1, 8)]
    assert len(vc.valid_claims(raw, [text])[0]) == vc.MAX_CLAIMS


# ---------------------------------------------------------------------- the decision
def test_decision_rules():
    assert vc.decide("table", 90.2, 90.2, "table") is None
    assert vc.decide("table", 93.7, 90.2, "table") == ("table_mismatch", "high", vc.TABLE_CONF, False)
    assert vc.decide("figure", 91.3, 91.0, "printed") is None                     # 0.3 %
    assert vc.decide("figure", 96.1, 91.3, "printed") == ("figure_mismatch", "high", vc.PRINTED_CONF, False)
    assert vc.decide("figure", 6.7, 6.9, "estimated") is None                     # 3 %
    assert vc.decide("figure", 7.8, 6.9, "estimated") == ("figure_unsure", "low", vc.UNSURE_CONF, True)    # 13 %
    assert vc.decide("figure", 8.7, 6.7, "estimated") == ("figure_mismatch", "medium", vc.ESTIMATED_CONF_MAX, True)


def test_blind_crop_lowers_a_finding_and_asks_a_human():
    assert vc.for_blind_crop(("figure_mismatch", "high", 0.85, False)) == ("figure_mismatch", "medium", vc.CROP_CONF_MAX, True)
    assert vc.for_blind_crop(("figure_unsure", "low", 0.3, True)) == ("figure_unsure", "low", 0.3, True)


# ------------------------------------------------------------ figures, from reply to finding
def test_matching_printed_numbers_give_no_finding(tmp_path):
    client = FakeClient({"Figure 1": claims_reply(CLAIM_PROPOSED, CLAIM_BASELINE)},
                        {"Figure 1": readings_reply((91.3, True), (71.4, True))})
    result = run(client, figure_paper(tmp_path))
    assert isinstance(result, AgentResult) and result.findings == [] and result.needs_human is False
    assert client.kinds() == ["claims", "readings"]


def test_printed_number_that_differs_is_a_high_confidence_mismatch(tmp_path):
    claim = ("The proposed scheme reaches 96.1% accuracy at 20 dB", "Accuracy of the proposed scheme", 96.1, "%")
    paper = figure_paper(tmp_path, paragraphs=(TEXT.replace("91.3", "96.1"),))
    client = FakeClient({"Figure 1": claims_reply(claim, CLAIM_BASELINE)},
                        {"Figure 1": readings_reply((91.3, True), (71.4, True))})
    result = run(client, paper)
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert (finding.type, finding.severity, finding.confidence) == ("figure_mismatch", "high", vc.PRINTED_CONF)
    assert finding.evidence.startswith("Figure 1: the text says 96.1%") and "91.3" in finding.evidence
    assert result.needs_human is False


def test_estimated_values_use_the_loose_rules(tmp_path):
    claims = claims_reply(CLAIM_PROPOSED, CLAIM_BASELINE)
    client = FakeClient({"Figure 1": claims}, {"Figure 1": readings_reply((80.0, False), (50.0, False))})
    result = run(client, figure_paper(tmp_path))
    assert [(f.type, f.severity) for f in result.findings] == [("figure_unsure", "low"),
                                                               ("figure_mismatch", "medium")]
    assert result.findings[1].confidence == vc.ESTIMATED_CONF_MAX and result.needs_human is True


def test_claim_that_cannot_be_found_in_the_figure_is_not_a_finding(tmp_path):
    client = FakeClient({"Figure 1": claims_reply(CLAIM_PROPOSED)}, {"Figure 1": readings_reply((None, False))})
    result = run(client, figure_paper(tmp_path))
    assert result.findings == []
    claim = json.loads(result.raw_output)[0]["claims"][0]
    assert claim["read"] is None and claim["note"] == "not found"


def test_scale_difference_is_not_a_mismatch(tmp_path):
    client = FakeClient({"Figure 1": claims_reply(CLAIM_PROPOSED)}, {"Figure 1": readings_reply((0.913, True))})
    assert run(client, figure_paper(tmp_path)).findings == []


def test_the_image_is_sent_as_base64_and_the_claimed_numbers_are_not_shown(tmp_path):
    client = FakeClient({"Figure 1": claims_reply(CLAIM_PROPOSED)}, {"Figure 1": readings_reply((91.3, True))})
    run(client, figure_paper(tmp_path), model="gemma3:4b")
    kind, prompt, images, model = client.calls[1]
    assert base64.b64decode(images[0]) == IMAGE_BYTES and model == "gemma3:4b"
    assert "Accuracy of the proposed scheme" in prompt and "91.3" not in prompt
    assert client.calls[0][2] is None                       # the claims request has no image
    assert "91.3% accuracy" in client.calls[0][1]           # but it sees the text


def test_blind_crop_findings_are_lowered(tmp_path):
    client = FakeClient({"Figure 1": claims_reply(CLAIM_PROPOSED)}, {"Figure 1": readings_reply((70.0, True))})
    result = run(client, figure_paper(tmp_path, fallback=True))
    finding = result.findings[0]
    assert (finding.severity, finding.confidence) == ("medium", vc.CROP_CONF_MAX)
    assert result.needs_human is True and "blind guess" in finding.evidence


# ------------------------------------------------------------------------------ tables
def test_table_cell_that_differs_is_a_table_mismatch():
    paper = table_paper(paragraphs=("Table 1 shows the results. The proposed model attains an F1 of 93.7% overall.",))
    claim = ("The proposed model attains an F1 of 93.7% overall", "F1 of the proposed model", 93.7, "%")
    client = FakeClient({"Table 1": claims_reply(claim)}, {"Table 1": readings_reply(("90.2", "Proposed"))})
    result = run(client, paper)
    finding = result.findings[0]
    assert (finding.type, finding.severity, finding.confidence) == ("table_mismatch", "high", vc.TABLE_CONF)
    assert finding.evidence.startswith("Table 1: the text says 93.7%") and "90.2" in finding.evidence
    assert client.calls[1][2] is None and "Proposed | 91.3 | 90.2" in client.calls[1][1]


def test_table_cell_that_matches_gives_no_finding():
    claim = ("The proposed scheme reaches 91.3% accuracy at 20 dB", "Accuracy of the proposed", 91.3, "%")
    client = FakeClient({"Table 1": claims_reply(claim)}, {"Table 1": readings_reply(("91.3", "Proposed"))})
    assert run(client, table_paper()).findings == []


def test_cell_not_in_the_table_or_in_the_wrong_row_is_not_trusted():
    claim = ("The proposed scheme reaches 91.3% accuracy at 20 dB", "Accuracy of the proposed", 91.3, "%")
    for cell, row in (("88.8", "Proposed"), ("71.4", "Proposed"), ("", "Proposed")):
        client = FakeClient({"Table 1": claims_reply(claim)}, {"Table 1": readings_reply((cell, row))})
        result = run(client, table_paper())
        assert result.findings == [], (cell, row)
        assert json.loads(result.raw_output)[0]["claims"][0]["read"] is None


# --------------------------------------------------------------------------- robustness
def test_one_retry_on_an_unreadable_reply(tmp_path):
    client = FakeClient({"Figure 1": ["not json", claims_reply(CLAIM_PROPOSED)]},
                        {"Figure 1": readings_reply((91.3, True))})
    run(client, figure_paper(tmp_path))
    assert client.kinds() == ["claims", "claims", "readings"]


def test_unreadable_replies_leave_the_item_unchecked(tmp_path):
    client = FakeClient({"Figure 1": "I cannot say"})
    result = run(client, figure_paper(tmp_path))
    record = json.loads(result.raw_output)[0]
    assert record["status"] == "unchecked" and "claims" in record["reason"] and result.findings == []
    client = FakeClient({"Figure 1": claims_reply(CLAIM_PROPOSED)}, {"Figure 1": "no idea"})
    record = json.loads(run(client, figure_paper(tmp_path)).raw_output)[0]
    assert record["status"] == "unchecked" and "values" in record["reason"]


def test_no_claims_means_nothing_to_report(tmp_path):
    client = FakeClient({"Figure 1": '{"claims": []}'})
    result = run(client, figure_paper(tmp_path))
    assert result.findings == [] and client.kinds() == ["claims"]


def test_model_failure_on_one_item_keeps_the_others(tmp_path):
    class Flaky(FakeClient):
        async def call_model(self, model, messages, images=None):
            if "Figure 1" in messages[1]["content"]:
                raise TimeoutError("slow model")
            return await super().call_model(model, messages, images)

    paper = figure_paper(tmp_path)
    paper.table_meta = table_paper().table_meta
    claim = ("The proposed scheme reaches 91.3% accuracy at 20 dB", "Accuracy of the proposed", 91.3, "%")
    client = Flaky({"Table 1": claims_reply(claim)}, {"Table 1": readings_reply(("99.9", "Proposed"))})
    client.scripts["readings"]["Table 1"] = [readings_reply(("91.3", "Proposed"))]
    records = json.loads(run(client, paper).raw_output)
    assert [r["status"] for r in records] == ["unchecked", "checked"]
    assert "TimeoutError" in records[0]["reason"]


def test_raw_output_keeps_what_was_read_and_dropped(tmp_path):
    bad = ("The method wins by 12.5 points", "q", 12.5, "")
    client = FakeClient({"Figure 1": claims_reply(CLAIM_PROPOSED, bad)}, {"Figure 1": readings_reply((91.3, True))})
    record = json.loads(run(client, figure_paper(tmp_path)).raw_output)[0]
    assert record["id"] == "Figure 1" and record["kind"] == "figure"
    assert record["claims"][0]["read"] == 91.3 and record["claims"][0]["how"] == "printed"
    assert record["dropped"][0]["dropped"] == "quote not found in the text"
