# Filename: src/orchestrator/real_agent_adapter.py
# Created: 2026-11-19
# Created by: Thulashan
# Purpose: Connect orchestration to teammate-owned math and vision
#          agents without changing their implementations.

from src.agents import math_checker, vision_checker
from src.common.schemas import AgentResult


class RealAgentAdapter:
    """Initial bounded integration for math and vision.

    Pass an AgentClient backed by the shared ModelPool.
    Do not call this adapter inside pool.reserve_route().

    Unlike MockAgentAdapter.execute(), this method returns
    AgentResult directly. Model-call metrics come from AgentClient.
    """

    SUPPORTED_MODELS = {
        "math_checker": frozenset({
            "deepseek-r1:7b",
            "qwen3:14b",
        }),
        "figure_checker": frozenset({
            "qwen2.5vl:7b",
            "gemma3:4b",
        }),
    }

    def __init__(self, max_equations=3, max_items=2):
        for name, value in (
            ("max_equations", max_equations),
            ("max_items", max_items),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")

        self.max_equations = max_equations
        self.max_items = max_items

    async def execute(
        self,
        paper,
        agent,
        model,
        client,
        numbers=None,
        ids=None,
        image_root=None,
    ) -> AgentResult:
        if agent not in self.SUPPORTED_MODELS:
            raise NotImplementedError(
                f"Real adapter does not yet support {agent!r}."
            )

        if model not in self.SUPPORTED_MODELS[agent]:
            raise ValueError(
                f"Model {model!r} is not supported for {agent!r}."
            )

        if agent == "math_checker":
            result = await math_checker.check_math(
                paper=paper,
                client=client,
                model=model,
                numbers=numbers,
                max_equations=self.max_equations,
            )
        else:
            options = {
                "paper": paper,
                "client": client,
                "model": model,
                "ids": ids,
                "max_items": self.max_items,
            }

            if image_root is not None:
                options["image_root"] = image_root

            result = await vision_checker.check_vision(**options)

        if not isinstance(result, AgentResult):
            raise TypeError(
                f"{agent!r} returned an unexpected result type."
            )

        return result