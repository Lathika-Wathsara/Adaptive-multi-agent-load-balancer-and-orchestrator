
import asyncio

from src.serving.client import LLMClient


def test_mock_client():

    client = LLMClient(use_mock=True)

    response = asyncio.run(
        client.call_model(
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
    assert response.latency >= 0
    assert response.tokens_in > 0
    assert "Mock response" in response.text


def test_large_model():

    client = LLMClient(use_mock=True)

    response = asyncio.run(
        client.call_model(
            model="qwen3:14b",
            messages=[
                {
                    "role": "user",
                    "content": "Evaluate the methodology."
                }
            ],
        )
    )

    assert response.model_name == "qwen3:14b"
    assert response.latency >= 0
