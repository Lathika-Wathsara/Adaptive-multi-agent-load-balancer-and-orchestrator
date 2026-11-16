
from src.common.schemas import (
    PaperDoc,
    TaskRequest,
    ModelResponse,
    Finding,
    AgentResult,
)


def test_schemas():
    paper = PaperDoc(
        id="paper001",
        title="Example Research Paper",
        sections={"Abstract": "This paper studies multi-agent systems."},
    )

    task = TaskRequest(
        agent_type="summarizer",
        paper_id=paper.id,
        payload={"text": paper.sections["Abstract"]},
        context_features={"input_length": 0.2},
    )

    response = ModelResponse(
        text="This paper investigates multi-agent systems.",
        model_name="qwen3:4b",
        latency=1.25,
        tokens_in=100,
        tokens_out=30,
    )

    finding = Finding(
        type="summary",
        severity="info",
        confidence=0.95,
        evidence=response.text,
    )

    result = AgentResult(
        findings=[finding],
        needs_human=False,
        raw_output=response.text,
    )

    assert task.paper_id == paper.id
    assert response.latency == 1.25
    assert result.findings[0].confidence == 0.95


if __name__ == "__main__":
    test_schemas()
    print("All schema tests passed!")
