# Filename: src/orchestrator/agent_adapter.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Provide an injectable mock agent boundary for testing
#          orchestration without implementing teammate-owned agents.

import json

from src.common.schemas import AgentResult


class MockAgentAdapter:
    """Exercise model calls and dependency flow.

    This adapter does not perform real evidence verification or
    image inspection. It returns no fabricated specialist findings.

    Future adapters must implement the same async execute method
    and return (AgentResult, ModelResponse).
    """

    async def execute(
        self,
        paper,
        task,
        decision,
        dependency_results,
        pool,
    ):
        evidence = {
            "paper_id": paper.id,
            "title": paper.title,
            "agent": task.agent,
            "target": task.target,
            "sections": paper.sections,
            "equations": paper.equations,
            "figure_paths": paper.figures,
            "references": paper.references,
            "previous_outputs": {
                task_id: result["raw_output"]
                for task_id, result in dependency_results.items()
            },
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "This is an orchestration integration test. "
                    "Return a short mock response. "
                    "Do not claim to verify evidence or inspect images."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(evidence, ensure_ascii=False),
            },
        ]

        response = await pool.call_model(
            model=decision.model,
            messages=messages,
        )

        result = AgentResult(
            findings=[],
            needs_human=False,
            raw_output=response.text,
        )

        return result, response