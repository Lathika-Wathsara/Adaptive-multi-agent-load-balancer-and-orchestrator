
import asyncio
import time
from typing import Dict, List, Optional

from src.common.mock_llm import MockLLM
from src.common.schemas import ModelResponse


class LLMClient:
    def __init__(
        self,
        use_mock: bool = True,
        ollama_url: str = "http://127.0.0.1:11434",
    ):
        self.use_mock = use_mock
        self.ollama_url = ollama_url.rstrip("/")
        self.mock = MockLLM()

    async def call_model(
        self,
        model: str,
        messages: List[Dict[str, str]],
        images: Optional[List[str]] = None,
    ) -> ModelResponse:

        if self.use_mock:
            return await self.mock.call_model(
                model=model,
                messages=messages,
                images=images,
            )

        return await self._call_ollama(
            model=model,
            messages=messages,
            images=images,
        )

    async def _call_ollama(
        self,
        model: str,
        messages: List[Dict[str, str]],
        images: Optional[List[str]] = None,
    ) -> ModelResponse:

        import httpx

        payload_messages = [dict(message) for message in messages]

        if images:
            payload_messages[-1]["images"] = images

        payload = {
            "model": model,
            "messages": payload_messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 4096,
            },
        }

        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("done_reason") == "length":
                raise RuntimeError(
                    "Model output reached the 4096-token limit; "
                    "the response is incomplete."
                )

        elapsed = time.perf_counter() - start

        return ModelResponse(
            text=data["message"]["content"],
            model_name=data.get("model", model),
            latency=elapsed,
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
        )
