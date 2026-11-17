# Filename: tests/test_execution_timeout.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify timeout cleanup and structured failure logging.

import asyncio
import json

import pytest

from src.common.schemas import PaperDoc
from src.orchestrator.planner import PaperPlanner
from src.orchestrator.scheduler import TaskScheduler
from src.serving.latency_log import append_failure_metrics


def test_timeout_releases_reserved_capacity():
    class BlockingClient:
        def __init__(self):
            self.started = False
            self.cleaned_up = False

        async def call_model(self, model, messages, images=None):
            self.started = True

            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned_up = True

    async def run():
        paper = PaperDoc(
            id="paper001",
            sections={"Abstract": "Test paper."},
        )
        plan = await PaperPlanner().plan(paper)
        client = BlockingClient()

        scheduler = TaskScheduler(
            client=client,
            task_timeout_seconds=0.1,
        )

        with pytest.raises(TimeoutError, match="summary"):
            await scheduler.execute(paper, plan)

        assert client.started
        assert client.cleaned_up
        assert scheduler.pool.active_total == 0
        assert scheduler.pool.pending_routes == 0
        assert all(
            count == 0
            for count in scheduler.pool.queue_counts.values()
        )
        assert all(
            count == 0
            for count in scheduler.pool.active_counts.values()
        )

    asyncio.run(run())


def test_timeout_is_logged_without_exception_text(tmp_path):
    path = tmp_path / "metrics.jsonl"

    append_failure_metrics(
        paper_id="paper001",
        mode="mock",
        elapsed=0.1,
        error=TimeoutError("Private diagnostic details"),
        path=path,
    )

    record = json.loads(path.read_text())

    assert record["status"] == "timeout"
    assert record["error_type"] == "TimeoutError"
    assert record["execution_mode"] == "mock"
    assert "Private diagnostic details" not in path.read_text()