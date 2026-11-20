# Filename: src/orchestrator/run_vision.py
# Created: 2026-11-19
# Created by: Thulashan
# Purpose: Run one real vision-agent task through the serving pool,
#          preserving the agent result and recording execution metrics.

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.ingest.models import ParsedPaper, from_json
from src.orchestrator.real_agent_adapter import RealAgentAdapter
from src.serving.agent_client import AgentClient
from src.serving.client import LLMClient
from src.serving.pool import ModelPool


async def execute(paper, item_id, model, timeout):
    backend = LLMClient(
        use_mock=False,
        ollama_url="http://127.0.0.1:11500",
    )
    pool = ModelPool(
        model_limits={model: 1},
        max_concurrent_total=1,
        client=backend,
    )
    client = AgentClient(pool, allowed_models=[model])
    adapter = RealAgentAdapter(max_items=1)

    output = {
        "paper_id": paper.id,
        "execution_mode": "real",
        "agent": "figure_checker",
        "model": model,
        "selected_item": item_id,
        "learning_enabled": False,
        "orchestration_review": {
            "assessment": "unavailable",
            "needs_human": True,
            "reasons": ["No usable agent assessment has been established."],
        },
    }

    started = time.perf_counter()

    try:
        result = await asyncio.wait_for(
            adapter.execute(
                paper=paper,
                agent="figure_checker",
                model=model,
                client=client,
                ids=[item_id],
                image_root=Path("/"),
            ),
            timeout=timeout,
        )

        output["status"] = "completed"
        output["agent_result"] = (
            result.model_dump()
            if hasattr(result, "model_dump")
            else result.dict()
        )
        output["orchestration_review"] = {
            "assessment": "requires_validation",
            "needs_human": True,
            "reasons": [
                "Inspect the agent records for skipped or unchecked items.",
                "The visual assessment has not been independently validated.",
            ],
        }

    except asyncio.TimeoutError:
        output["status"] = "timeout"
        output["error"] = f"Task exceeded {timeout:g} seconds."

    except Exception as error:
        output["status"] = "failed"
        output["error_type"] = type(error).__name__
        output["error"] = str(error)

    finally:
        client.close()
        output["elapsed_seconds"] = time.perf_counter() - started
        output["model_calls"] = client.metrics()
        output["active_slots_after_run"] = pool.active_total
        output["queued_calls_after_run"] = sum(pool.queue_counts.values())

    metrics = output["model_calls"]

    if output["status"] == "completed" and (
        metrics["calls_completed"] == 0
        or metrics["calls_attempted"] != metrics["calls_completed"]
    ):
        output["status"] = "incomplete"
        output["orchestration_review"]["assessment"] = "incomplete"
        output["orchestration_review"]["reasons"].append(
            "No model call completed, or at least one model call failed."
        )

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run one bounded real vision-agent integration check."
    )
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", default="Figure 2")
    parser.add_argument(
        "--model",
        choices=["qwen2.5vl:7b", "gemma3:4b"],
        default="qwen2.5vl:7b",
    )
    args = parser.parse_args()

    if not args.paper.is_file():
        parser.error(f"Paper not found: {args.paper}")

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    text = args.paper.read_text(encoding="utf-8")
    metadata = json.loads(text)

    selected = [
        item
        for item in metadata.get("figure_meta", [])
        if item.get("id") == args.id
    ]
    if len(selected) != 1:
        parser.error(f"Expected exactly one figure matching {args.id!r}.")

    image_path = Path(selected[0].get("path") or "")
    if not image_path.is_absolute() or not image_path.is_file():
        parser.error(f"Expected an existing absolute image path: {image_path}")

    paper = from_json(ParsedPaper, text)

    print(f"Selected figure: {args.id}", flush=True)
    print(f"Image: {image_path}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print("Running real vision check...", flush=True)

    output = asyncio.run(
        execute(paper, args.id, args.model, timeout=180.0)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"\nStatus: {output['status']}")
    print(f"Elapsed: {output['elapsed_seconds']:.3f} seconds")
    print("Model calls:", json.dumps(output["model_calls"], indent=2))
    print("Active slots remaining:", output["active_slots_after_run"])
    print("Needs human review:", output["orchestration_review"]["needs_human"])

    if "agent_result" in output:
        print("Agent result:")
        print(json.dumps(output["agent_result"], indent=2, ensure_ascii=False))

    if "error" in output:
        print("Error:", output["error"])

    print(f"Saved: {args.output}")

    if output["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()