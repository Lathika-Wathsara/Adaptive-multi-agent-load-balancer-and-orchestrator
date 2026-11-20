# Filename: tests/test_agent_client.py
# Created: 2026-11-19
# Created by: Thulashan
# Purpose: Verify repeated and multi-model agent calls, accounting,
#          capacity limits, cancellation, and failure cleanup.

import asyncio

import pytest

from src.common.schemas import ModelResponse
from src.serving.agent_client import AgentClient
from src.serving.pool import ModelPool


MESSAGES = [{"role": "user", "content": "Test request"}]


class RecordingBackend:
    def __init__(self):
        self.active = 0
        self.peak = 0

    async def call_model(self, model, messages, images=None):
        self.active += 1
        self.peak = max(self.peak, self.active)

        try:
            await asyncio.sleep(0.01)

            return ModelResponse(
                text="Test response",
                model_name=model,
                latency=0.01,
                tokens_in=10,
                tokens_out=5,
            )
        finally:
            self.active -= 1


def assert_idle(pool):
    assert pool.active_total == 0
    assert all(value == 0 for value in pool.active_counts.values())
    assert all(value == 0 for value in pool.queue_counts.values())


def test_repeated_calls_are_allowed_and_counted():
    async def run():
        pool = ModelPool(
            {"small": 1},
            client=RecordingBackend(),
        )
        client = AgentClient(pool, allowed_models=["small"])

        await client.call_model("small", MESSAGES)
        await client.call_model("small", MESSAGES)

        metrics = client.metrics()

        assert metrics["calls_completed"] == 2
        assert metrics["tokens_in"] == 20
        assert metrics["tokens_out"] == 10
        assert metrics["model_latency_seconds"] == pytest.approx(0.02)

        assert_idle(pool)

        client.close()
        with pytest.raises(RuntimeError, match="closed"):
            await client.call_model("small", MESSAGES)

    asyncio.run(run())


def test_multiple_models_share_global_capacity():
    async def run():
        backend = RecordingBackend()
        pool = ModelPool(
            {"small": 1, "large": 1},
            max_concurrent_total=1,
            client=backend,
        )
        client = AgentClient(pool)

        await asyncio.gather(
            client.call_model("small", MESSAGES),
            client.call_model("large", MESSAGES),
        )

        assert backend.peak == 1
        assert client.metrics()["calls_completed"] == 2
        assert {call["model"] for call in client.calls} == {
            "small", "large"
        }

        assert_idle(pool)

    asyncio.run(run())


def test_failed_call_is_recorded_and_releases_capacity():
    class FailingBackend:
        async def call_model(self, model, messages, images=None):
            raise RuntimeError("Backend failure")

    async def run():
        pool = ModelPool({"small": 1}, client=FailingBackend())
        client = AgentClient(pool)

        with pytest.raises(RuntimeError, match="Backend failure"):
            await client.call_model("small", MESSAGES)

        metrics = client.metrics()

        assert metrics["calls_attempted"] == 1
        assert metrics["calls_completed"] == 0
        assert metrics["calls"][0]["status"] == "failed"
        assert metrics["calls"][0]["error_type"] == "RuntimeError"

        assert_idle(pool)

    asyncio.run(run())


def test_cancellation_is_recorded_and_releases_capacity():
    class BlockingBackend:
        def __init__(self):
            self.started = asyncio.Event()

        async def call_model(self, model, messages, images=None):
            self.started.set()
            await asyncio.Event().wait()

    async def run():
        backend = BlockingBackend()
        pool = ModelPool({"small": 1}, client=backend)
        client = AgentClient(pool)

        task = asyncio.create_task(
            client.call_model("small", MESSAGES)
        )

        await asyncio.wait_for(backend.started.wait(), timeout=1.0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert client.calls[0]["status"] == "cancelled"
        assert_idle(pool)

    asyncio.run(run())