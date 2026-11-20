# Filename: src/orchestrator/run_math.py
# Created: 2026-11-19
# Created by: Thulashan
# Purpose: Run one real math-checking task through the agent adapter
#          and serving pool, saving findings and model-call metrics.

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.agents.math_checker import build_steps
from src.ingest.models import ParsedPaper, from_json
from src.orchestrator.real_agent_adapter import RealAgentAdapter
from src.serving.agent_client import AgentClient
from src.serving.client import LLMClient
from src.serving.pool import ModelPool


async def execute(paper, model, numbers, timeout):
    steps = build_steps(
        paper,
        numbers=numbers,
        max_equations=1,
    )

    if not steps:
        raise ValueError(
            "No eligible equations were found for this selection."
        )

    print(
        "Selected equation:",
        steps[0].get("number"),
        flush=True,
    )
    print(f"Model: {model}", flush=True)
    print("Running real math check...", flush=True)

    backend = LLMClient(
        use_mock=False,
        ollama_url="http://127.0.0.1:11500",
    )

    pool = ModelPool(
        model_limits={model: 1},
        max_concurrent_total=1,
        client=backend,
    )

    client = AgentClient(
        pool,
        allowed_models=[model],
    )

    adapter = RealAgentAdapter(max_equations=1)
    started = time.perf_counter()

    output = {
        "paper_id": paper.id,
        "execution_mode": "real",
        "agent": "math_checker",
        "model": model,
        "max_equations": 1,
        "learning_enabled": False,
    }

    try:
        result = await asyncio.wait_for(
            adapter.execute(
                paper=paper,
                agent="math_checker",
                model=model,
                client=client,
                numbers=numbers,
            ),
            timeout=timeout,
        )

        output["status"] = "completed"
        output["agent_result"] = (
            result.model_dump()
            if hasattr(result, "model_dump")
            else result.dict()
        )

        # Keep the original agent result unchanged.
        # Execution completion does not establish mathematical correctness.
        review_reasons = []

        try:
            records = json.loads(result.raw_output)

            if (
                not isinstance(records, list)
                or not records
                or not all(isinstance(record, dict) for record in records)
            ):
                raise ValueError("Expected a non-empty list of records.")

            unchecked = sum(
                record.get("verdict") == "unchecked"
                for record in records
            )
            output["unchecked_equations"] = unchecked

            if unchecked:
                output["status"] = "incomplete"
                review_reasons.append(
                    "One or more equations were left unchecked."
                )

            for record in records:
                number = record.get("number", "?")
                latex = str(record.get("latex", ""))

                if record.get("source") == "pdf":
                    review_reasons.append(
                        f"Equation {number}: confirm the extracted equation "
                        "against the original PDF."
                    )

                if any(
                    ord(character) < 32
                    and character not in "\n\r\t"
                    for character in latex
                ):
                    review_reasons.append(
                        f"Equation {number}: extracted text contains "
                        "unexpected control characters."
                    )

                test = record.get("test")
                if (
                    not isinstance(test, dict)
                    or test.get("status") in (None, "", "none")
                ):
                    review_reasons.append(
                        f"Equation {number}: no mathematical test was recorded."
                    )

            assessment = (
                "incomplete" if unchecked else "requires_validation"
            )

        except (TypeError, ValueError):
            output["unchecked_equations"] = None
            output["status"] = "incomplete"
            assessment = "unavailable"
            review_reasons.append(
                "The equation assessment records could not be interpreted."
            )

        # This integration runner does not independently validate proof
        # correctness, even when the agent reports a test or high confidence.
        review_reasons.append(
            "The agent's assessment has not been independently validated."
        )

        output["orchestration_review"] = {
            "assessment": assessment,
            "needs_human": True,
            "reasons": list(dict.fromkeys(review_reasons)),
        }

    except asyncio.TimeoutError:
        output["status"] = "timeout"
        output["error"] = (
            f"Task exceeded {timeout:g} seconds."
        )

    except Exception as error:
        output["status"] = "failed"
        output["error_type"] = type(error).__name__
        output["error"] = str(error)

    finally:
        client.close()
        output["elapsed_seconds"] = time.perf_counter() - started
        output["model_calls"] = client.metrics()
        output["active_slots_after_run"] = pool.active_total
        output["queued_calls_after_run"] = sum(
            pool.queue_counts.values()
        )

    if "orchestration_review" not in output:
        output["orchestration_review"] = {
            "assessment": "unavailable",
            "needs_human": True,
            "reasons": [
                "Execution did not produce a usable equation assessment."
            ],
        }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run a bounded real math-agent integration check."
    )

    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--model",
        choices=["deepseek-r1:7b", "qwen3:14b"],
        default="deepseek-r1:7b",
    )

    parser.add_argument(
        "--number",
        default=None,
        help="Optional equation number; otherwise check the first eligible one.",
    )

    args = parser.parse_args()

    if not args.paper.is_file():
        parser.error(f"Paper not found: {args.paper}")

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    paper = from_json(
        ParsedPaper,
        args.paper.read_text(encoding="utf-8"),
    )

    numbers = [args.number] if args.number is not None else None

    output = asyncio.run(
        execute(
            paper=paper,
            model=args.model,
            numbers=numbers,
            timeout=900.0,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"\nStatus: {output['status']}")
    print(f"Elapsed: {output['elapsed_seconds']:.3f} seconds")
    print(
        "Model calls completed:",
        output["model_calls"]["calls_completed"],
    )
    print("Unchecked equations:", output.get("unchecked_equations"))
    print("Active slots remaining:", output["active_slots_after_run"])

    result = output.get("agent_result")

    if result is not None:
        print("Agent-reported needs human review:", result["needs_human"])
        print("Findings:")
        print(json.dumps(result["findings"], indent=2))
        review = output["orchestration_review"]
        print("Orchestration assessment:", review["assessment"])
        print("Needs human review:", review["needs_human"])

    for reason in review["reasons"]:
        print(f"  - {reason}")

    if "error" in output:
        print("Error:", output["error"])

    print(f"Saved: {args.output}")

    if output["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()