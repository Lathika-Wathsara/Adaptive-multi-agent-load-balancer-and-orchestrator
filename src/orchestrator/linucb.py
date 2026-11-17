# Filename: src/orchestrator/linucb.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Implement disjoint LinUCB for selecting an eligible model
#          and learning from explicitly supplied reward feedback.

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


class LinUCB:
    """One linear reward model per arm (model name).

    Contexts must use a fixed feature order and scaling.

    Rewards are expected in [0, 1], where larger is better.
    Selection never updates the policy automatically.
    """

    def __init__(
        self,
        arms: Iterable[str],
        context_dimension: int = 16,
        alpha: float = 1.0,
        regularization: float = 1.0,
    ):
        arm_names = list(arms)

        if not arm_names:
            raise ValueError("At least one arm is required.")

        if any(not isinstance(arm, str) or not arm for arm in arm_names):
            raise ValueError("Each arm must be a nonempty string.")

        if len(set(arm_names)) != len(arm_names):
            raise ValueError("Arm names must be unique.")

        if (
            isinstance(context_dimension, bool)
            or not isinstance(context_dimension, int)
            or context_dimension <= 0
        ):
            raise ValueError("context_dimension must be a positive integer.")

        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative.")

        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive.")

        self.arms = tuple(arm_names)
        self.context_dimension = context_dimension
        self.alpha = float(alpha)
        self.regularization = float(regularization)

        self.A: Dict[str, np.ndarray] = {
            arm: self.regularization * np.eye(context_dimension)
            for arm in self.arms
        }

        self.b: Dict[str, np.ndarray] = {
            arm: np.zeros(context_dimension)
            for arm in self.arms
        }

        self.update_counts: Dict[str, int] = {
            arm: 0 for arm in self.arms
        }

    def _validate_context(
        self,
        context: Sequence[float],
    ) -> np.ndarray:
        x = np.asarray(context, dtype=float)

        if x.shape != (self.context_dimension,):
            raise ValueError(
                f"Expected context shape ({self.context_dimension},), "
                f"received {x.shape}."
            )

        if not np.all(np.isfinite(x)):
            raise ValueError("Context must contain only finite values.")

        return x.copy()

    def _validate_eligible(
        self,
        eligible_arms: Optional[Iterable[str]],
    ) -> Tuple[str, ...]:
        candidates = (
            list(self.arms)
            if eligible_arms is None
            else list(eligible_arms)
        )

        if not candidates:
            raise ValueError("No eligible arms were provided.")

        if len(set(candidates)) != len(candidates):
            raise ValueError("Eligible arms must be unique.")

        unknown = set(candidates) - set(self.arms)

        if unknown:
            raise ValueError(
                f"Unknown eligible arms: {sorted(unknown)}"
            )

        return tuple(candidates)

    def scores(
        self,
        context: Sequence[float],
        eligible_arms: Optional[Iterable[str]] = None,
    ) -> Dict[str, float]:
        """Return upper-confidence scores for eligible arms."""
        x = self._validate_context(context)
        candidates = self._validate_eligible(eligible_arms)

        values = {}

        for arm in candidates:
            # Solve linear systems instead of explicitly inverting A.
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            solved_x = np.linalg.solve(self.A[arm], x)

            predicted_reward = float(x @ theta)
            uncertainty = np.sqrt(max(float(x @ solved_x), 0.0))

            values[arm] = float(
                predicted_reward + self.alpha * uncertainty
            )

        return values

    def select(
        self,
        context: Sequence[float],
        eligible_arms: Optional[Iterable[str]] = None,
    ) -> str:
        """Choose the highest-scoring eligible arm.

        Equal scores are resolved by the supplied eligibility order.
        The router can use that order to express a cold-start preference.
        """
        values = self.scores(context, eligible_arms)
        return max(values, key=values.get)

    def update(
        self,
        arm: str,
        context: Sequence[float],
        reward: float,
    ) -> None:
        """Update the selected arm using its original decision context."""
        if arm not in self.A:
            raise ValueError(f"Unknown arm: {arm}")

        x = self._validate_context(context)
        reward_value = float(reward)

        if (
            not np.isfinite(reward_value)
            or not 0.0 <= reward_value <= 1.0
        ):
            raise ValueError("Reward must be finite and in [0, 1].")

        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward_value * x
        self.update_counts[arm] += 1