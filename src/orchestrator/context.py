# Filename: src/orchestrator/context.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Build a fixed, versioned 16-feature context for LinUCB
#          using existing ingestion features and runtime task information.

import math
from typing import List

from src.common.schemas import PaperDoc
from src.ingest.features import compute_features


CONTEXT_VERSION = 1

# Fixed engineering scales for the initial implementation.
# These are not estimates learned from evaluation/test papers.
# Changing the order or scales requires a new context version
# and resetting or retraining the LinUCB state.
PAPER_FEATURE_SCALES = (
    ("n_tokens_est", 20000.0),
    ("n_sections", 30.0),
    ("n_equations", 100.0),
    ("n_numbered_equations", 100.0),
    ("n_figures", 30.0),
    ("n_tables", 20.0),
    ("n_references", 100.0),
    ("eq_per_1k_tokens", 20.0),
    ("fig_per_1k_tokens", 10.0),
    ("tab_per_1k_tokens", 10.0),
)

TASK_TYPES = ("general", "math", "vision")

CONTEXT_FEATURE_NAMES = (
    "bias",
    *(name for name, _ in PAPER_FEATURE_SCALES),
    "task_general",
    "task_math",
    "task_vision",
    "difficulty",
    "queue_utilization",
)

CONTEXT_DIMENSION = len(CONTEXT_FEATURE_NAMES)


class ContextBuilder:
    """Construct one context shared across eligible model arms.

    queue_utilization describes overall pending-plus-running load,
    divided by a configured capacity and capped at one.

    Per-model capacity and availability belong in the router/pool.
    """

    @staticmethod
    def _unit_value(value: float, name: str) -> float:
        number = float(value)

        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")

        return number

    def build(
        self,
        paper: PaperDoc,
        task_type: str,
        difficulty: float = 0.5,
        queue_utilization: float = 0.0,
    ) -> List[float]:
        if task_type not in TASK_TYPES:
            raise ValueError(
                f"Unknown task_type {task_type!r}; "
                f"expected one of {TASK_TYPES}."
            )

        difficulty = self._unit_value(difficulty, "difficulty")
        queue_utilization = self._unit_value(
            queue_utilization,
            "queue_utilization",
        )

        # Recompute from the supplied paper to avoid stale cached features.
        features = compute_features(paper)

        context = [1.0]

        for name, scale in PAPER_FEATURE_SCALES:
            value = float(features[name])

            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Paper feature {name!r} must be finite "
                    "and nonnegative."
                )

            context.append(min(value / scale, 1.0))

        context.extend(
            float(task_type == candidate)
            for candidate in TASK_TYPES
        )

        context.extend([difficulty, queue_utilization])

        return context