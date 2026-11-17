# Filename: src/serving/pool.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Manage model concurrency, atomically route and reserve
#          capacity, and clean up after failures or cancellation.

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from src.common.schemas import ModelResponse
from src.serving.client import LLMClient


class ModelPool:
    """Shared capacity manager for one event loop.

    queue_counts:
        Direct calls waiting or running, plus admitted routed calls.

    pending_routes:
        Routed tasks waiting for a model reservation.

    active_counts:
        Reserved or running slots for each model.
    """

    def __init__(
        self,
        model_limits: Dict[str, int],
        max_concurrent_total: int = 2,
        client=None,
    ):
        if not model_limits:
            raise ValueError("At least one model is required.")

        limits = dict(model_limits)

        for model, limit in limits.items():
            if not isinstance(model, str) or not model:
                raise ValueError("Model names must be nonempty strings.")

            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or limit < 1
            ):
                raise ValueError(
                    "Model limits must be positive integers."
                )

        if (
            isinstance(max_concurrent_total, bool)
            or not isinstance(max_concurrent_total, int)
            or max_concurrent_total < 1
        ):
            raise ValueError(
                "Total concurrency must be a positive integer."
            )

        self.model_limits = limits
        self.max_concurrent_total = max_concurrent_total

        self.client = (
            client
            if client is not None
            else LLMClient(use_mock=True)
        )

        self.queue_counts = {model: 0 for model in limits}
        self.active_counts = {model: 0 for model in limits}

        self.active_total = 0
        self.pending_routes = 0

        self._condition = asyncio.Condition()

    def _has_capacity(self, model: str) -> bool:
        return (
            self.active_total < self.max_concurrent_total
            and self.active_counts[model] < self.model_limits[model]
        )

    @property
    def available_models(self):
        """Return a snapshot of models with available capacity."""
        return tuple(
            model
            for model in self.model_limits
            if self._has_capacity(model)
        )

    @property
    def queue_utilization(self) -> float:
        """Normalized pending-plus-running load, capped at one."""
        return min(
            (
                sum(self.queue_counts.values())
                + self.pending_routes
            ) / self.max_concurrent_total,
            1.0,
        )

    async def call_model(
        self,
        model: str,
        messages: List[Dict[str, str]],
        images: Optional[List[str]] = None,
    ) -> ModelResponse:
        """Call a specified model, waiting for capacity if necessary."""
        if model not in self.model_limits:
            raise ValueError(f"Unknown model: {model!r}")

        admitted = False
        self.queue_counts[model] += 1

        try:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._has_capacity(model)
                )

                self.active_counts[model] += 1
                self.active_total += 1
                admitted = True

            return await self.client.call_model(
                model=model,
                messages=messages,
                images=images,
            )

        finally:
            self.queue_counts[model] -= 1

            if admitted:
                self.active_counts[model] -= 1
                self.active_total -= 1

                async with self._condition:
                    self._condition.notify_all()

    @asynccontextmanager
    async def reserve_route(
        self,
        router,
        paper,
        task_type,
        difficulty=0.5,
    ):
        """Wait for capacity, then select and reserve atomically."""
        if task_type not in router.eligibility:
            raise ValueError(f"Unknown task type: {task_type!r}")

        eligible = tuple(
            model
            for model in router.eligibility[task_type]
            if model in self.model_limits
        )

        if not eligible:
            raise ValueError(
                f"No configured pool models support {task_type!r}."
            )

        selected = None
        pending = True
        self.pending_routes += 1

        def candidates():
            return [
                model
                for model in eligible
                if self._has_capacity(model)
            ]

        try:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: bool(candidates())
                )

                # Select only among models with capacity.
                # No await occurs between selection and reservation.
                decision = router.route(
                    paper=paper,
                    task_type=task_type,
                    difficulty=difficulty,
                    queue_utilization=self.queue_utilization,
                    available_models=candidates(),
                )

                selected = decision.model

                self.pending_routes -= 1
                pending = False

                self.queue_counts[selected] += 1
                self.active_counts[selected] += 1
                self.active_total += 1

            reserved_client = _ReservedClient(
                self.client,
                selected,
            )

            try:
                yield decision, reserved_client
            finally:
                reserved_client.close()

        finally:
            if pending:
                self.pending_routes -= 1

            if selected is not None:
                self.queue_counts[selected] -= 1
                self.active_counts[selected] -= 1
                self.active_total -= 1

                async with self._condition:
                    self._condition.notify_all()


class _ReservedClient:
    """Permit one call through an already reserved model slot."""

    def __init__(self, client, model):
        self._client = client
        self._model = model
        self._used = False
        self._closed = False

    async def call_model(self, model, messages, images=None):
        if self._closed:
            raise RuntimeError("The model reservation has expired.")

        if model != self._model:
            raise ValueError("Model does not match the reservation.")

        if self._used:
            raise RuntimeError(
                "This reservation permits only one model call."
            )

        self._used = True

        return await self._client.call_model(
            model=model,
            messages=messages,
            images=images,
        )

    def close(self):
        self._closed = True