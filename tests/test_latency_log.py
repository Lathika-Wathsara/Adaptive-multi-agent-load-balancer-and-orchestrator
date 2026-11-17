# Filename: tests/test_latency_log.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Verify append behavior, timing validation,
#          and exclusion of review text from metrics.

import json

import pytest

from src.serving.latency_log import append_review_metrics


def example_result():
    return {
        "paper_id": "paper001",
        "execution_mode": "mock",
        "total_latency": 0.3,
        "results": [{
            "task_id": "summary",
            "decision_id": "decision001",
            "agent": "summarizer",
            "model": "qwen3:4b",
            "latency": 0.1,
            "queue_wait": 0.2,
            "task_elapsed": 0.3,
            "tokens_in": 20,
            "tokens_out": 15,
            "raw_output": "Private review text",
        }],
    }


def test_appends_distinct_records_without_review_text(tmp_path):
    path = tmp_path / "metrics.jsonl"

    first = append_review_metrics(example_result(), path)
    second = append_review_metrics(example_result(), path)

    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines]

    assert len(records) == 2
    assert first != second
    assert records[0]["execution_mode"] == "mock"
    assert records[0]["tasks"][0]["queue_wait_seconds"] == 0.2
    assert "Private review text" not in path.read_text()


def test_invalid_timing_is_not_written(tmp_path):
    path = tmp_path / "metrics.jsonl"
    result = example_result()
    result["results"][0]["latency"] = float("nan")

    with pytest.raises(ValueError):
        append_review_metrics(result, path)

    assert not path.exists()