
import asyncio

import pytest

from src.common.schemas import PaperDoc, ModelResponse
from src.orchestrator.planner import PaperPlanner
from src.orchestrator.scheduler import TaskScheduler


def create_test_paper():
    return PaperDoc(
        id="paper001",
        title="Massive MIMO Signal Processing",
        sections={
            "Abstract": "A study of massive MIMO.",
            "Methodology": "We derive channel models.",
        },
        equations=[
            "y = Hx + n",
            "R = log2(1 + SINR)",
            "SINR = S / (I + N)",
        ],
        figures=[
            "figure1.png",
            "figure2.png",
        ],
        references=["Reference 1"],
    )


def test_scheduler_execution():
    async def run():
        paper = create_test_paper()

        planner = PaperPlanner(use_mock=True)
        scheduler = TaskScheduler()

        plan = await planner.plan(paper)
        result = await scheduler.execute(paper, plan)

        assert result["paper_id"] == paper.id
        assert result["total_latency"] > 0
        assert result["tasks_executed"] == len(plan.tasks)

        agents = [
            r["agent"] for r in result["results"]
        ]

        assert agents[0] == "summarizer"
        assert agents[-1] == "final_reviewer"

        assert "math_checker" in agents
        assert "figure_checker" in agents

        for task_result in result["results"]:
            assert task_result["model"] is not None
            assert task_result["latency"] >= 0
            assert len(task_result["context"]) == 16

        assert all(
            count == 0
            for count in scheduler.model_queues.values()
        )

    asyncio.run(run())


def test_scheduler_rejects_wrong_paper():
    async def run():
        paper = create_test_paper()
        planner = PaperPlanner(use_mock=True)

        plan = await planner.plan(paper)
        plan.paper_id = "wrong-paper"

        scheduler = TaskScheduler()

        with pytest.raises(ValueError):
            await scheduler.execute(paper, plan)

    asyncio.run(run())


def test_specialists_can_run_concurrently():
    class CountingClient:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def call_model(
            self, model, messages, images=None
        ):
            self.active += 1
            self.max_active = max(
                self.max_active, self.active
            )

            try:
                await asyncio.sleep(0.05)

                return ModelResponse(
                    text=f"Response from {model}",
                    model_name=model,
                    latency=0.05,
                    tokens_in=20,
                    tokens_out=10,
                )
            finally:
                self.active -= 1

    async def run():
        paper = create_test_paper()
        planner = PaperPlanner(use_mock=True)
        plan = await planner.plan(paper)

        client = CountingClient()

        scheduler = TaskScheduler(
            client=client,
            max_concurrent_per_model=1,
        )

        result = await scheduler.execute(paper, plan)

        assert result["tasks_executed"] == len(plan.tasks)

        # Different eligible models can execute
        # independent specialist tasks concurrently.
        assert client.max_active >= 2

    asyncio.run(run())
