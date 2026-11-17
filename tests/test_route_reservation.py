# Filename: tests/test_route_reservation.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify atomic model reservation and cleanup after
#          waiting-task cancellation or adapter failure.

import asyncio

import pytest

from src.common.schemas import PaperDoc
from src.orchestrator.router import ModelRouter
from src.serving.pool import ModelPool


def test_busy_model_is_excluded_at_selection():
    async def run():
        router = ModelRouter()
        paper = PaperDoc(id="paper001")

        pool = ModelPool(
            {"qwen3:4b": 1, "qwen3:14b": 1},
            max_concurrent_total=2,
        )

        async with pool.reserve_route(
            router, paper, "general"
        ) as (first, _):
            assert first.model == "qwen3:4b"

            async with pool.reserve_route(
                router, paper, "general"
            ) as (second, _):
                assert second.model == "qwen3:14b"
                assert second.eligible_models == ("qwen3:14b",)
                assert pool.active_total == 2

        assert pool.active_total == 0
        assert pool.pending_routes == 0
        assert all(
            value == 0 for value in pool.queue_counts.values()
        )

    asyncio.run(run())


def test_cancelled_waiter_does_not_leak_capacity():
    async def run():
        router = ModelRouter()
        paper = PaperDoc(id="paper001")
        pool = ModelPool(
            {"qwen3:4b": 1},
            max_concurrent_total=1,
        )

        async def wait_for_slot():
            async with pool.reserve_route(
                router, paper, "general"
            ):
                raise AssertionError(
                    "Waiter should not be admitted."
                )

        async with pool.reserve_route(
            router, paper, "general"
        ):
            waiter = asyncio.create_task(wait_for_slot())
            await asyncio.sleep(0)

            assert pool.pending_routes == 1

            waiter.cancel()

            with pytest.raises(asyncio.CancelledError):
                await waiter

            assert pool.pending_routes == 0
            assert pool.active_total == 1

        assert pool.active_total == 0
        assert all(
            value == 0 for value in pool.queue_counts.values()
        )

    asyncio.run(run())


def test_adapter_failure_releases_reservation():
    async def run():
        router = ModelRouter()
        paper = PaperDoc(id="paper001")
        pool = ModelPool({"qwen3:4b": 1})

        with pytest.raises(RuntimeError, match="Adapter failed"):
            async with pool.reserve_route(
                router, paper, "general"
            ):
                raise RuntimeError("Adapter failed")

        assert pool.active_total == 0
        assert pool.pending_routes == 0
        assert pool.available_models == ("qwen3:4b",)

    asyncio.run(run())