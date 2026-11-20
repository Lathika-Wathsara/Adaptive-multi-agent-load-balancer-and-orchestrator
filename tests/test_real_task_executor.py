# Filename: tests/test_real_task_executor.py
# Created: 2026-11-19
# Created by: Thulashan
# Purpose: Verify multi-call execution and timeout cleanup without GPUs.

import asyncio
from types import SimpleNamespace

import pytest

from src.orchestrator.real_task_executor import RealTaskExecutor
from src.orchestrator.router import RoutingDecision
from src.serving.pool import ModelPool


MODEL = "qwen2.5vl:7b"


class FakeRouter:
    feedback = SimpleNamespace(learning_enabled=False)

    def route(self, **kwargs):
        assert MODEL in kwargs["available_models"]
        return RoutingDecision(
            decision_id="vision-test",
            paper_id=kwargs["paper"].id,
            task_type="vision",
            model=MODEL,
            context=(1.0,),
            context_version=1,
            eligible_models=(MODEL,),
            scores=((MODEL, 1.0),),
        )


class FakeBackend:
    def __init__(self, block=False):
        self.block = block
        self.started = asyncio.Event()
        self.cancelled = False

    async def call_model(self, model, messages, images=None):
        self.started.set()
        try:
            if self.block:
                await asyncio.Event().wait()
            return SimpleNamespace(
                text="test response",
                model_name=model,
                latency=0.01,
                tokens_in=10,
                tokens_out=5,
            )
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class TwoCallAdapter:
    async def execute(self, **kwargs):
        for _ in range(2):
            await kwargs["client"].call_model(
                model=kwargs["model"],
                messages=[{"role": "user", "content": "test"}],
            )

        return SimpleNamespace(
            model_dump=lambda: {
                "findings": [],
                "needs_human": False,
                "raw_output": "[]",
            }
        )


def make_executor(backend, timeout=1.0):
    pool = ModelPool(
        model_limits={MODEL: 1},
        max_concurrent_total=1,
        client=backend,
    )
    executor = RealTaskExecutor(
        router=FakeRouter(),
        pool=pool,
        adapter=TwoCallAdapter(),
        timeout=timeout,
    )
    return executor, pool


def test_two_calls_share_one_slot_without_deadlock():
    async def scenario():
        executor, pool = make_executor(FakeBackend())

        result = await executor.execute(
            paper=SimpleNamespace(id="paper-test"),
            agent="figure_checker",
            ids=["Figure 2"],
        )

        assert result["model_calls"]["calls_completed"] == 2
        assert result["status"] == "completed"
        assert result["orchestration_review"]["needs_human"] is True
        assert result["agent_result"]["needs_human"] is False
        assert pool.active_total == 0
        assert sum(pool.queue_counts.values()) == 0

    asyncio.run(scenario())


def test_timeout_releases_active_slot():
    async def scenario():
        backend = FakeBackend(block=True)
        executor, pool = make_executor(backend, timeout=0.1)

        with pytest.raises(asyncio.TimeoutError):
            await executor.execute(
                paper=SimpleNamespace(id="paper-test"),
                agent="figure_checker",
            )

        assert backend.started.is_set()
        assert backend.cancelled
        assert pool.active_total == 0
        assert sum(pool.queue_counts.values()) == 0

    asyncio.run(scenario())
def test_scheduler_runs_selected_specialist(monkeypatch):
    from src.orchestrator.real_agent_adapter import RealAgentAdapter
    from src.orchestrator.scheduler import TaskScheduler

    async def fake_execute(self, **kwargs):
        return await TwoCallAdapter().execute(**kwargs)

    monkeypatch.setattr(
        RealAgentAdapter,
        "execute",
        fake_execute,
    )

    async def scenario():
        scheduler = TaskScheduler(
            client=FakeBackend(),
            task_timeout_seconds=1.0,
        )

        result = await scheduler.execute_specialists(
            paper=SimpleNamespace(id="paper-test"),
            figure_ids=["Figure 2"],
        )
        return result, scheduler

    # Replace only routing/context construction for this scheduling test.
    def fake_route(self, **kwargs):
        return FakeRouter().route(**kwargs)

    monkeypatch.setattr(
        "src.orchestrator.router.ModelRouter.route",
        fake_route,
    )

    result, scheduler = asyncio.run(scenario())

    assert result["review_scope"] == "selected_specialists_only"
    assert result["tasks_executed"] == 1
    assert result["status"] == "completed"
    assert result["needs_human"] is True
    assert result["results"][0]["model_calls"]["calls_completed"] == 2
    assert scheduler.pool.active_total == 0
    assert sum(scheduler.pool.queue_counts.values()) == 0