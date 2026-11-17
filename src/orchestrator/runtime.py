# Filename: src/orchestrator/runtime.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Run a mock paper review, restore or save router state,
#          and export structured execution results.

import argparse
import asyncio
import json
import sys
import time

from pathlib import Path

from src.ingest.models import ParsedPaper, from_json
from src.orchestrator.planner import PaperPlanner
from src.orchestrator.router import DEFAULT_CONFIG_PATH
from src.orchestrator.scheduler import TaskScheduler
from src.serving.client import LLMClient
from src.serving.latency_log import (
    append_review_metrics,
    append_failure_metrics,
)


async def run_review(
    paper_path: Path,
    config_path: Path,
    checkpoint_path=None,
):
    paper = from_json(
        ParsedPaper,
        paper_path.read_text(encoding="utf-8"),
    )

    restored = (
        checkpoint_path is not None
        and Path(checkpoint_path).is_file()
    )

    scheduler = TaskScheduler(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        client=LLMClient(use_mock=True),
    )

    # Mock review outputs are not evaluation rewards.
    if scheduler.router.feedback.learning_enabled:
        raise ValueError(
            "This mock runtime requires learning_enabled: false."
        )

    planner = PaperPlanner(use_mock=True)
    plan = await planner.plan(paper)
    result = await scheduler.execute(paper, plan)

    if checkpoint_path is not None:
        scheduler.router.save_checkpoint()

    result["execution_mode"] = "mock"
    result["source_file"] = str(paper_path.resolve())
    result["checkpoint_restored"] = restored
    result["checkpoint_path"] = (
        str(Path(checkpoint_path).resolve())
        if checkpoint_path is not None
        else None
    )
    result["policy_update_counts"] = dict(
        scheduler.router.policy.update_counts
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run a mock review with optional router persistence."
    )

    parser.add_argument(
        "--paper",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New results JSON file; existing files are not overwritten.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Combined router checkpoint to restore or create.",
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Append completed-review measurements to this JSONL file.",
    )

    args = parser.parse_args()

    if not args.paper.is_file():
        parser.error(f"Parsed paper not found: {args.paper}")

    if not args.config.is_file():
        parser.error(f"Configuration not found: {args.config}")

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    if (
        args.checkpoint is not None
        and args.checkpoint.exists()
        and not args.checkpoint.is_file()
    ):
        parser.error("Checkpoint path must refer to a file.")

    paths = [args.paper, args.config, args.output]
    if args.checkpoint is not None:
        paths.append(args.checkpoint)

    if args.metrics is not None:
        paths.append(args.metrics)

    resolved = [path.resolve() for path in paths]

    if len(resolved) != len(set(resolved)):
        parser.error(
            "Paper, config, output, and checkpoint paths must be distinct."
        )

    started = time.perf_counter()

    try:
        result = asyncio.run(
            run_review(
                paper_path=args.paper,
                config_path=args.config,
                checkpoint_path=args.checkpoint,
            )
        )
    except Exception as error:
        if args.metrics is not None:
            try:
                append_failure_metrics(
                    paper_id=args.paper.stem,
                    mode="mock",
                    elapsed=time.perf_counter() - started,
                    error=error,
                    path=args.metrics,
                )
            except Exception as logging_error:
                # Preserve the original execution failure.
                print(
                    "Failure metrics could not be written: "
                    f"{type(logging_error).__name__}",
                    file=sys.stderr,
                )

        raise

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")

    if args.metrics is not None:
        run_id = append_review_metrics(result, args.metrics)
        print(f"Metrics saved: {args.metrics}")
        print(f"Metrics run ID: {run_id}")

    print(f"Paper: {result['paper_id']}")
    print(f"Mode: {result['execution_mode']}")
    print(f"Tasks completed: {result['tasks_executed']}")
    print(f"Total latency: {result['total_latency']:.3f} seconds")
    print(f"Checkpoint restored: {result['checkpoint_restored']}")

    print("\nTask results:")
    for task in result["results"]:
        print(
            f"  {task['agent']:20s} "
            f"{task['model']:18s} "
            f"model={task['latency']:.3f}s "
            f"wait={task.get('queue_wait', 0.0):.3f}s"
        )

    print(f"\nPolicy updates: {result['policy_update_counts']}")
    print(f"Results saved: {args.output}")

    if args.checkpoint is not None:
        print(f"Checkpoint saved: {args.checkpoint}")


if __name__ == "__main__":
    main()