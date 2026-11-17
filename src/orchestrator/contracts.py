# Filename: src/orchestrator/contracts.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Define orchestration task and plan records without
#          changing the shared agent or ingestion schemas.

from dataclasses import dataclass, field
from typing import List


@dataclass
class PlannedTask:
    task_id: str
    agent: str
    task_type: str
    target: str
    priority: int
    est_difficulty: float
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ReviewPlan:
    paper_id: str
    tasks: List[PlannedTask] = field(default_factory=list)