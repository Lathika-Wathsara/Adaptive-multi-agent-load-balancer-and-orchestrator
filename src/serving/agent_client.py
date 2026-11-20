# Filename: src/serving/agent_client.py
# Created: 2026-11-19
# Created by: Thulashan
# Purpose: Provide a reusable, capacity-limited client for agents
#          that make multiple model calls and collect per-call metrics.

import asyncio
import time
from typing import Optional, Sequence


class AgentClient:
    """One instance per agent task, backed by a shared ModelPool.

    Every call reserves its own capacity through pool.call_model().
    Do not use this client inside pool.reserve_route(): doing so
    would reserve capacity twice and could cause a deadlock.

    The agent must await all calls before returning.
    """

    def __init__(
        self,
        pool,
        allowed_models: Optional[Sequence[str]] = None,
    ):
        self.pool = pool

        models = (
            tuple(pool.model_limits)
            if allowed_models is None
            else tuple(allowed_models)
        )

        if not models:
            raise ValueError("At least one allowed model is required.")

        unknown = set(models) - set(pool.model_limits)

        if unknown:
            raise ValueError(
                f"Models are not configured in the pool: {sorted(unknown)}"
            )

        self.allowed_models = frozenset(models)
        self.calls = []
        self._closed = False

    async def call_model(self, model, messages, images=None):
        if self._closed:
            raise RuntimeError("This agent client has been closed.")

        if model not in self.allowed_models:
            raise ValueError(
                f"Model {model!r} is not allowed for this agent task."
            )

        record = {
            "call_index": len(self.calls),
            "model": model,
            "status": "running",
            "elapsed_seconds": 0.0,
        }
        self.calls.append(record)

        started = time.perf_counter()

        try:
            response = await self.pool.call_model(
                model=model,
                messages=messages,
                images=images,
            )

            record.update({
                "status": "completed",
                "response_model": response.model_name,
                "model_latency_seconds": response.latency,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
            })

            return response

        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise

        except Exception as error:
            record.update({
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error)[:1000],
            })
            raise
        finally:
            record["elapsed_seconds"] = (
                time.perf_counter() - started
            )

    def metrics(self):
        """Return totals for successful calls plus all call records.

        Summed model latency is service time, not agent wall time.
        elapsed_seconds includes pool waiting and client overhead.
        """
        completed = [
            call for call in self.calls
            if call["status"] == "completed"
        ]

        return {
            "calls_attempted": len(self.calls),
            "calls_completed": len(completed),
            "model_latency_seconds": sum(
                call["model_latency_seconds"]
                for call in completed
            ),
            "tokens_in": sum(
                call["tokens_in"] for call in completed
            ),
            "tokens_out": sum(
                call["tokens_out"] for call in completed
            ),
            "calls": [dict(call) for call in self.calls],
        }

    def close(self):
        """Prevent new calls after the agent task has finished."""
        self._closed = True