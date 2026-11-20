# Filename: src/orchestrator/scheduler.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Validate and execute review plans using dependency waves,
#          LinUCB routing, an injected agent adapter, and a shared pool.

import asyncio
import time
import math
from pathlib import Path

import yaml

from src.orchestrator.real_task_executor import RealTaskExecutor
from src.orchestrator.agent_adapter import MockAgentAdapter
from src.orchestrator.router import (
    DEFAULT_CONFIG_PATH,
    ModelRouter,
)
from src.serving.pool import ModelPool


class TaskScheduler:
    def __init__(
        self,
        client=None,
        max_concurrent_per_model=None,
        config_path=None,
        adapter=None,
        checkpoint_path=None,
        task_timeout_seconds=None,
    ):
        path = (
            DEFAULT_CONFIG_PATH
            if config_path is None
            else Path(config_path)
        )

        self.router = ModelRouter(config_path=path, checkpoint_path=checkpoint_path)

        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        timeout = (
            config["serving"].get("task_timeout_seconds", 900.0)
            if task_timeout_seconds is None
            else task_timeout_seconds
        )

        self.task_timeout_seconds = float(timeout)

        if (
            not math.isfinite(self.task_timeout_seconds)
            or self.task_timeout_seconds <= 0
        ):
            raise ValueError(
                "Task timeout must be finite and positive."
            )

        limits = {
            model: (
                settings.get("max_concurrent", 1)
                if max_concurrent_per_model is None
                else max_concurrent_per_model
            )
            for model, settings in config["models"].items()
            if settings.get("enabled") is True
        }

        self.pool = ModelPool(
            model_limits=limits,
            max_concurrent_total=config["serving"].get(
                "max_concurrent_total", 2
            ),
            client=client,
        )

        self.adapter = (
            MockAgentAdapter()
            if adapter is None
            else adapter
        )

        # Preserve the interface expected by existing scheduler tests.
        self.model_queues = self.pool.queue_counts

    def _validate_plan(self, paper, plan):
        if plan.paper_id != paper.id:
            raise ValueError("Plan paper_id does not match the paper.")

        if not plan.tasks:
            raise ValueError("The plan has no tasks.")

        ids = [task.task_id for task in plan.tasks]

        if any(not task_id for task_id in ids):
            raise ValueError("Task IDs must not be empty.")

        if len(ids) != len(set(ids)):
            raise ValueError("Task IDs must be unique.")

        known_ids = set(ids)

        summaries = [
            task for task in plan.tasks
            if task.agent == "summarizer"
        ]
        finals = [
            task for task in plan.tasks
            if task.agent == "final_reviewer"
        ]

        if len(summaries) != 1 or len(finals) != 1:
            raise ValueError(
                "Exactly one summarizer and final reviewer are required."
            )

        summary = summaries[0]
        final = finals[0]

        if summary.depends_on:
            raise ValueError("Summarizer must have no dependencies.")

        if set(final.depends_on) != known_ids - {final.task_id}:
            raise ValueError(
                "Final reviewer must depend on every other task."
            )

        for task in plan.tasks:
            dependencies = set(task.depends_on)

            if task.task_id in dependencies:
                raise ValueError("A task cannot depend on itself.")

            if not dependencies <= known_ids:
                raise ValueError("Plan contains an unknown dependency.")

            if (
                task.task_id not in {summary.task_id, final.task_id}
                and summary.task_id not in dependencies
            ):
                raise ValueError(
                    "Each specialist must depend on the summarizer."
                )

            if task.task_type not in self.router.eligibility:
                raise ValueError(
                    f"Unknown task type: {task.task_type!r}"
                )

            if not self.router.eligibility[task.task_type]:
                raise ValueError(
                    f"No enabled models for {task.task_type!r}"
                )

            # Validate context inputs before executing any requests.
            self.router.context_builder.build(
                paper,
                task.task_type,
                difficulty=task.est_difficulty,
            )

        # Detect dependency cycles before making model calls.
        completed = set()
        remaining = list(plan.tasks)

        while remaining:
            ready = [
                task for task in remaining
                if set(task.depends_on) <= completed
            ]

            if not ready:
                raise ValueError("Plan contains a dependency cycle.")

            completed.update(task.task_id for task in ready)
            remaining = [
                task for task in remaining
                if task.task_id not in completed
            ]

    async def execute_specialists(
        self,
        paper,
        *,
        math_numbers=None,
        figure_ids=None,
        image_root=None,
    ):
        """Execute explicitly selected specialists as a partial review.

        The caller must supply the intended backend when constructing
        TaskScheduler. Production runs require LLMClient(use_mock=False).
        """
        selections = []

        for agent, values, argument in (
            ("math_checker", math_numbers, "numbers"),
            ("figure_checker", figure_ids, "ids"),
        ):
            if values is None:
                continue

            if (
                not isinstance(values, (list, tuple))
                or len(values) != 1
                or not isinstance(values[0], str)
                or not values[0].strip()
            ):
                raise ValueError(
                    "Select exactly one non-empty identifier per specialist."
                )

            selections.append((agent, argument, list(values)))

        if not selections:
            raise ValueError("Select at least one specialist task.")

        executor = RealTaskExecutor(
            router=self.router,
            pool=self.pool,
            timeout=self.task_timeout_seconds,
        )

        # Validate every selection before starting any model requests.
        for agent, _, _ in selections:
            task_type = executor.TASK_TYPES[agent]
            supported = executor.adapter.SUPPORTED_MODELS[agent]

            candidates = (
                set(self.router.eligibility[task_type])
                & set(self.pool.model_limits)
                & set(supported)
            )
            if not candidates:
                raise ValueError(
                    f"No enabled pool model supports {agent!r}."
                )

        started = time.perf_counter()
        jobs = [
            asyncio.create_task(
                executor.execute(
                    paper=paper,
                    agent=agent,
                    image_root=image_root,
                    **{argument: values},
                )
            )
            for agent, argument, values in selections
        ]

        try:
            results = await asyncio.gather(*jobs)
        except BaseException:
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            raise

        return {
            "paper_id": paper.id,
            "review_scope": "selected_specialists_only",
            "status": (
                "completed"
                if all(item["status"] == "completed" for item in results)
                else "incomplete"
            ),
            "tasks_executed": len(results),
            "results": results,
            "total_latency": time.perf_counter() - started,
            "needs_human": any(
                item["orchestration_review"]["needs_human"]
                for item in results
            ),
        }
    async def _execute_task_body(self, paper, task, completed):
        dependencies = {
            task_id: completed[task_id]
            for task_id in task.depends_on
        }

        started = time.perf_counter()

        async with self.pool.reserve_route(
            router=self.router,
            paper=paper,
            task_type=task.task_type,
            difficulty=task.est_difficulty,
        ) as (decision, reserved_client):
            queue_wait = time.perf_counter() - started

            agent_result, response = await self.adapter.execute(
                paper=paper,
                task=task,
                decision=decision,
                dependency_results=dependencies,
                pool=reserved_client,
            )

        return {
            "task_id": task.task_id,
            "agent": task.agent,
            "model": decision.model,
            "response_model": response.model_name,
            "latency": response.latency,
            "queue_wait": queue_wait,
            "task_elapsed": time.perf_counter() - started,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "context": list(decision.context),
            "context_version": decision.context_version,
            "decision_id": decision.decision_id,
            "scores": dict(decision.scores),
            "raw_output": agent_result.raw_output,
            "findings": [
                finding.model_dump()
                if hasattr(finding, "model_dump")
                else finding.dict()
                for finding in agent_result.findings
            ],
            "needs_human": agent_result.needs_human,
        }

    async def _execute_task(self, paper, task, completed):
        try:
            return await asyncio.wait_for(
                self._execute_task_body(paper, task, completed),
                timeout=self.task_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Task {task.task_id!r} ({task.agent}) timed out "
                f"after {self.task_timeout_seconds:g} seconds, "
                "including queue waiting."
            ) from exc

    async def execute(self, paper, plan):
        self._validate_plan(paper, plan)

        started = time.perf_counter()
        completed = {}
        remaining = list(plan.tasks)

        while remaining:
            ready = sorted(
                [
                    task for task in remaining
                    if set(task.depends_on) <= set(completed)
                ],
                key=lambda task: task.priority,
            )

            jobs = [
                asyncio.create_task(
                    self._execute_task(paper, task, completed)
                )
                for task in ready
            ]

            try:
                results = await asyncio.gather(*jobs)
            except BaseException:
                # Finish cancellation cleanup before propagating failure.
                for job in jobs:
                    job.cancel()

                await asyncio.gather(*jobs, return_exceptions=True)
                raise

            for result in results:
                completed[result["task_id"]] = result

            remaining = [
                task for task in remaining
                if task.task_id not in completed
            ]

        # Present results in plan order regardless of completion order.
        ordered = [
            completed[task.task_id]
            for task in plan.tasks
        ]

        final = next(
            result for result in ordered
            if result["agent"] == "final_reviewer"
        )

        return {
            "paper_id": paper.id,
            "tasks_executed": len(ordered),
            "results": ordered,
            "total_latency": time.perf_counter() - started,
            "final_review": final["raw_output"],
            "needs_human": any(
                result["needs_human"] for result in ordered
            ),
            "adapter": type(self.adapter).__name__,
        }