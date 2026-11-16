
import asyncio
import time
from typing import List, Dict

from src.common.schemas import ModelResponse


class MockLLM:
    def __init__(self):
        self.model_latencies = {
            "qwen3:4b": 0.1,
            "qwen3:14b": 0.3,
            "deepseek-r1:7b": 0.2,
            "qwen2.5vl:7b": 0.25,
            "gemma3:4b": 0.15,
        }

    async def call_model(
        self,
        model: str,
        messages: List[Dict[str, str]],
        images=None,
    ) -> ModelResponse:

        if model not in self.model_latencies:
            raise ValueError(f"Unknown model: {model}")

        start = time.perf_counter()

        # Simulate model processing time
        await asyncio.sleep(self.model_latencies[model])

        prompt = " ".join(
            message.get("content", "")
            for message in messages
        )

        return ModelResponse(
            text=f"Mock response from {model}: {prompt[:100]}",
            model_name=model,
            latency=time.perf_counter() - start,
            tokens_in=len(prompt.split()),
            tokens_out=15,
        )
