
import asyncio

from src.common.mock_llm import MockLLM


def test_mock_llm():

    mock = MockLLM()

    response = asyncio.run(
        mock.call_model(
            model="qwen3:4b",
            messages=[
                {
                    "role": "user",
                    "content": "Summarize this research paper."
                }
            ],
        )
    )

    assert response.model_name == "qwen3:4b"
    assert "Mock response" in response.text
    assert response.latency >= 0
    assert response.tokens_in > 0


def test_unknown_model():

    mock = MockLLM()

    try:
        asyncio.run(
            mock.call_model(
                model="unknown-model",
                messages=[{"role": "user", "content": "Test"}],
            )
        )
    except ValueError:
        return

    assert False, "Expected ValueError for unknown model"
