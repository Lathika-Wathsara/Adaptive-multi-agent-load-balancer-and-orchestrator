
import asyncio

from src.common.schemas import PaperDoc
from src.orchestrator.planner import PaperPlanner
from src.orchestrator.scheduler import TaskScheduler


async def main():

    paper = PaperDoc(
        id="mimo001",
        title="Massive MIMO Signal Processing",
        sections={
            "Abstract": (
                "This paper investigates massive MIMO "
                "signal processing and performance."
            ),
            "Methodology": (
                "We derive analytical expressions "
                "for system performance."
            ),
        },
        equations=[
            "y = Hx + n",
            "R = log2(1 + SINR)",
            "SINR = S / (I + N)",
        ],
        figures=[
            "figure1.png",
            "figure2.png",
        ],
        references=["Reference 1"],
    )

    planner = PaperPlanner(use_mock=True)
    scheduler = TaskScheduler()

    # Create the task plan
    plan = await planner.plan(paper)

    print("\nPLANNED TASKS")
    print("-" * 65)

    for task in plan.tasks:
        print(
            f"{task.agent:23s} "
            f"Priority: {task.priority}"
        )

    # Execute the plan
    result = await scheduler.execute(
        paper, plan
    )

    print("\nEXECUTION RESULTS")
    print("-" * 65)

    for task in result["results"]:
        print(
            f"Agent: {task['agent']:23s} "
            f"Model: {task['model']:20s} "
            f"Latency: {task['latency']:.3f}s"
        )

    print("\nTOTAL REVIEW LATENCY")
    print(f"{result['total_latency']:.3f} seconds")

    print("\nFINAL REVIEW")
    print(result["final_review"])


if __name__ == "__main__":
    asyncio.run(main())
