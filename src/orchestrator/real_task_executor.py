# Filename: src/orchestrator/real_task_executor.py
# Created: 2026-11-19
# Created by: Thulashan
# Purpose: Route a real specialist task and manage multiple model
#          calls through the shared serving pool.

import asyncio
import math
import time
from dataclasses import asdict

from src.orchestrator.real_agent_adapter import RealAgentAdapter
from src.serving.agent_client import AgentClient


class RealTaskExecutor:
    TASK_TYPES = {
        "math_checker": "math",
        "figure_checker": "vision",
    }

    def __init__(self, router, pool, adapter=None, timeout=180.0):
        self.router = router
        self.pool = pool
        self.adapter = (
            RealAgentAdapter(max_equations=1, max_items=1)
            if adapter is None
            else adapter
        )
        self.timeout = float(timeout)

        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Timeout must be finite and positive.")

        if router.feedback.learning_enabled:
            raise ValueError(
                "Keep learning disabled during real-agent integration."
            )

    async def execute(
        self,
        paper,
        agent,
        difficulty=0.5,
        numbers=None,
        ids=None,
        image_root=None,
    ):
        if agent not in self.TASK_TYPES:
            raise NotImplementedError(
                f"Real task execution does not support {agent!r}."
            )

        supported = RealAgentAdapter.SUPPORTED_MODELS[agent]
        candidates = [
            model
            for model in self.pool.model_limits
            if model in supported
        ]

        # Select once from enabled, supported pool models.
        # Each model call subsequently waits for its own pool slot.
        decision = self.router.route(
            paper=paper,
            task_type=self.TASK_TYPES[agent],
            difficulty=difficulty,
            queue_utilization=self.pool.queue_utilization,
            available_models=candidates,
        )

        client = AgentClient(
            self.pool,
            allowed_models=[decision.model],
        )
        started = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self.adapter.execute(
                    paper=paper,
                    agent=agent,
                    model=decision.model,
                    client=client,
                    numbers=numbers,
                    ids=ids,
                    image_root=image_root,
                ),
                timeout=self.timeout,
            )
        finally:
            # Timeout, failure, and cancellation also close the client.
            client.close()

        metrics = client.metrics()
        incomplete = (
            metrics["calls_completed"] == 0
            or metrics["calls_attempted"] != metrics["calls_completed"]
        )

        return {
            "paper_id": paper.id,
            "agent": agent,
            "execution_mode": "real",
            "status": "incomplete" if incomplete else "completed",
            "routing_decision": asdict(decision),
            "model_calls": metrics,
            "task_elapsed": time.perf_counter() - started,
            "agent_result": (
                result.model_dump()
                if hasattr(result, "model_dump")
                else result.dict()
            ),
            "orchestration_review": {
                "assessment": (
                    "incomplete" if incomplete else "requires_validation"
                ),
                "needs_human": True,
                "reasons": [
                    "Inspect agent records for unchecked or skipped items.",
                    "Execution metrics do not establish review correctness.",
                ],
            },
        }