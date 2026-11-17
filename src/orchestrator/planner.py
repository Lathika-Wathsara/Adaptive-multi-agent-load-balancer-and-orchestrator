# Filename: src/orchestrator/planner.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Build deterministic review plans with explicit
#          dependencies between summarization, specialists,
#          and final review.

from src.common.schemas import PaperDoc
from src.orchestrator.contracts import PlannedTask, ReviewPlan


class PaperPlanner:
    """Initial rule-based planner.

    use_mock=True supports the existing scheduler tests and demo.
    LLM-generated planning is not implemented in this version.

    The presence of equations, figures, and references determines
    which optional specialists are included.
    """

    def __init__(self, use_mock: bool = True):
        if not use_mock:
            raise NotImplementedError(
                "LLM planning is not implemented. "
                "Use PaperPlanner(use_mock=True) for rule-based planning."
            )

    async def plan(self, paper: PaperDoc) -> ReviewPlan:
        if not paper.id.strip():
            raise ValueError("Paper ID must not be empty.")

        tasks = [
            PlannedTask(
                task_id="summary",
                agent="summarizer",
                task_type="general",
                target="paper",
                priority=0,
                est_difficulty=0.3,
            )
        ]

        def add_specialist(
            task_id: str,
            agent: str,
            task_type: str,
            target: str,
            difficulty: float,
        ) -> None:
            tasks.append(
                PlannedTask(
                    task_id=task_id,
                    agent=agent,
                    task_type=task_type,
                    target=target,
                    priority=1,
                    est_difficulty=difficulty,
                    depends_on=["summary"],
                )
            )

        if paper.equations:
            add_specialist(
                "math",
                "math_checker",
                "math",
                "equations",
                0.8,
            )

        if paper.figures:
            add_specialist(
                "figures",
                "figure_checker",
                "vision",
                "figures",
                0.6,
            )

        if paper.references:
            add_specialist(
                "references",
                "ref_verifier",
                "general",
                "references",
                0.5,
            )

        add_specialist(
            "methodology",
            "methodology",
            "general",
            "sections",
            0.7,
        )

        add_specialist(
            "novelty",
            "novelty",
            "general",
            "paper",
            0.7,
        )

        tasks.append(
            PlannedTask(
                task_id="final",
                agent="final_reviewer",
                task_type="general",
                target="all_results",
                priority=2,
                est_difficulty=0.8,
                depends_on=[task.task_id for task in tasks],
            )
        )

        return ReviewPlan(paper_id=paper.id, tasks=tasks)