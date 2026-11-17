# Filename: tests/test_pool.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify pool concurrency limits and capacity cleanup
#          after request failure or cancellation.

import asyncio

import pytest

from src.common.schemas import ModelResponse
from src.serving.pool import ModelPool


MESSAGES = [{"role": "user", "content": "Review this paper."}]


class CountingClient:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.by_model = {}
        self.model_peaks = {}

    async def call_model(self, model, messages, images=None):
        self.active += 1
        self.peak = max(self.peak, self.active)

        self.by_model[model] = self.by_model.get(model, 0) + 1
        self.model_peaks[model] = max(
            self.model_peaks.get(model, 0),
            self.by_model[model],
        )

        try:
            await asyncio.sleep(0.02)
            return ModelResponse(
                text="Completed",
                model_name=model,
                latency=0.02,
            )
        finally:
            self.active -= 1
            self.by_model[model] -= 1


def assert_idle(pool):
    assert pool.active_total == 0
    assert all(value == 0 for value in pool.active_counts.values())
    assert all(value == 0 for value in pool.queue_counts.values())


def test_per_model_limit_and_parallel_execution():
    async def run():
        client = CountingClient()
        pool = ModelPool(
            {"small": 1, "large": 1},
            max_concurrent_total=2,
            client=client,
        )

        await asyncio.gather(
            pool.call_model("small", MESSAGES),
            pool.call_model("small", MESSAGES),
            pool.call_model("large", MESSAGES),
            pool.call_model("large", MESSAGES),
        )

        assert client.peak == 2
        assert client.model_peaks == {"small": 1, "large": 1}
        assert_idle(pool)

    asyncio.run(run())


def test_total_limit_applies_across_models():
    async def run():
        client = CountingClient()
        pool = ModelPool(
            {"small": 2, "large": 2},
            max_concurrent_total=1,
            client=client,
        )

        await asyncio.gather(
            pool.call_model("small", MESSAGES),
            pool.call_model("large", MESSAGES),
        )

        assert client.peak == 1
        assert_idle(pool)

    asyncio.run(run())


def test_failure_releases_capacity():
    class FailingClient:
        async def call_model(self, model, messages, images=None):
            raise RuntimeError("Backend failed")

    async def run():
        pool = ModelPool({"small": 1}, client=FailingClient())

        with pytest.raises(RuntimeError, match="Backend failed"):
            await pool.call_model("small", MESSAGES)

        assert_idle(pool)
        assert pool.available_models == ("small",)

    asyncio.run(run())


def test_cancelling_running_and_waiting_requests():
    class BlockingClient:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def call_model(self, model, messages, images=None):
            self.started.set()
            await self.release.wait()

            return ModelResponse(
                text="Completed",
                model_name=model,
                latency=0.0,
            )

    async def run():
        client = BlockingClient()
        pool = ModelPool({"small": 1}, client=client)

        running = asyncio.create_task(
            pool.call_model("small", MESSAGES)
        )

        await asyncio.wait_for(client.started.wait(), timeout=1.0)

        waiting = asyncio.create_task(
            pool.call_model("small", MESSAGES)
        )
        await asyncio.sleep(0)

        assert pool.queue_counts["small"] == 2
        assert pool.active_total == 1

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        assert pool.queue_counts["small"] == 1

        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        assert_idle(pool)

        # Confirm the released capacity can be reused.
        client.release.set()
        response = await pool.call_model("small", MESSAGES)
        assert response.text == "Completed"
        assert_idle(pool)

    asyncio.run(run())