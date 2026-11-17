# Filename: src/serving/latency_log.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Append completed-review timing and usage measurements
#          to a JSON Lines file without logging paper text.

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def nonnegative_number(value, name):
    number = float(value)

    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")

    return number


def append_review_metrics(result, path):
    """Append one record per completed review.

    Use with a single writer process. Token counts and timings in
    mock mode are simulated and must not be treated as real benchmarks.
    """
    mode = result.get("execution_mode")

    if mode not in {"mock", "real"}:
        raise ValueError("Execution mode must be 'mock' or 'real'.")

    tasks = []

    for task in result["results"]:
        tasks.append({
            "task_id": task["task_id"],
            "decision_id": task["decision_id"],
            "agent": task["agent"],
            "model": task["model"],
            "response_model": task.get(
                "response_model", task["model"]
            ),
            "model_latency_seconds": nonnegative_number(
                task["latency"], "latency"
            ),
            "queue_wait_seconds": nonnegative_number(
                task["queue_wait"], "queue_wait"
            ),
            "task_elapsed_seconds": nonnegative_number(
                task["task_elapsed"], "task_elapsed"
            ),
            "tokens_in": nonnegative_number(
                task["tokens_in"], "tokens_in"
            ),
            "tokens_out": nonnegative_number(
                task["tokens_out"], "tokens_out"
            ),
        })

    record = {
        "schema_version": 1,
        "run_id": uuid4().hex,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_id": result["paper_id"],
        "execution_mode": mode,
        "status": "completed",
        "total_latency_seconds": nonnegative_number(
            result["total_latency"], "total_latency"
        ),
        "tasks_executed": len(tasks),
        "tasks": tasks,
    }

    # Serialize before opening the file, so invalid records
    # do not produce partial output during validation.
    line = json.dumps(record, allow_nan=False) + "\n"

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("a", encoding="utf-8") as file:
        file.write(line)

    return record["run_id"]

def append_failure_metrics(paper_id, mode, elapsed, error, path):
    """Record a failed run without saving exception text or paper text."""
    if mode not in {"mock", "real"}:
        raise ValueError("Execution mode must be 'mock' or 'real'.")

    record = {
        "schema_version": 1,
        "run_id": uuid4().hex,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_id": paper_id,
        "execution_mode": mode,
        "status": (
            "timeout"
            if isinstance(error, TimeoutError)
            else "failed"
        ),
        "error_type": type(error).__name__,
        "elapsed_seconds": nonnegative_number(elapsed, "elapsed"),
    }

    line = json.dumps(record, allow_nan=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("a", encoding="utf-8") as file:
        file.write(line)

    return record["run_id"]