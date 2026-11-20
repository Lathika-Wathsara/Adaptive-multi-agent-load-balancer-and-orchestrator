# Filename: tests/test_client_limits.py
# Created: 2026-11-19
# Created by: Thulashan
# Purpose: Verify the Ollama output budget and rejection of
#          responses truncated at the generation limit.

import asyncio

import httpx
import pytest

from src.serving.client import LLMClient


@pytest.mark.parametrize("done_reason", ["stop", "length"])
def test_output_limit(monkeypatch, done_reason):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "model": "deepseek-r1:7b",
                "message": {"content": "Test response"},
                "done_reason": done_reason,
                "prompt_eval_count": 10,
                "eval_count": 20,
            }

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json):
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeHTTPClient)

    async def run():
        client = LLMClient(use_mock=False)

        return await client.call_model(
            model="deepseek-r1:7b",
            messages=[{"role": "user", "content": "Test"}],
        )

    if done_reason == "length":
        with pytest.raises(RuntimeError, match="incomplete"):
            asyncio.run(run())
    else:
        result = asyncio.run(run())
        assert result.text == "Test response"

    assert captured["payload"]["options"]["num_predict"] == 4096