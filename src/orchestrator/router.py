# Filename: src/orchestrator/router.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Select an eligible, available model using the fixed
#          context builder and LinUCB policy.

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple
from uuid import uuid4
from src.orchestrator.feedback import FeedbackManager

import yaml

from src.common.schemas import PaperDoc
from src.orchestrator.context import (
    CONTEXT_DIMENSION,
    CONTEXT_VERSION,
    ContextBuilder,
)
from src.orchestrator.linucb import LinUCB


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "models.yaml"
)


class NoAvailableModelError(RuntimeError):
    """No eligible model is currently available."""


@dataclass(frozen=True)
class RoutingDecision:
    decision_id: str
    paper_id: str
    task_type: str
    model: str
    context: Tuple[float, ...]
    context_version: int
    eligible_models: Tuple[str, ...]
    scores: Tuple[Tuple[str, float], ...]


class ModelRouter:
    def __init__(self, config_path=None, checkpoint_path=None):
        path = (
            DEFAULT_CONFIG_PATH
            if config_path is None
            else Path(config_path)
        )

        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("Model configuration must be a mapping.")

        models = config.get("models")
        eligibility = config.get("task_eligibility")
        routing = config.get("routing")

        if not isinstance(models, dict) or not models:
            raise ValueError("Configuration requires a models mapping.")

        if not isinstance(eligibility, dict):
            raise ValueError("Configuration requires task_eligibility.")

        if not isinstance(routing, dict):
            raise ValueError("Configuration requires routing settings.")

        if routing.get("policy") != "linucb":
            raise ValueError("Only the linucb policy is supported.")

        if routing.get("context_dimension") != CONTEXT_DIMENSION:
            raise ValueError("Configured context dimension is incompatible.")

        if routing.get("context_version") != CONTEXT_VERSION:
            raise ValueError("Configured context version is incompatible.")

        for model, settings in models.items():
            if not isinstance(model, str) or not model:
                raise ValueError("Model names must be nonempty strings.")

            if not isinstance(settings, dict):
                raise ValueError(f"Invalid settings for {model!r}.")

        enabled_models = [
            model
            for model, settings in models.items()
            if settings.get("enabled") is True
        ]

        if not enabled_models:
            raise ValueError("At least one model must be enabled.")

        self.eligibility = {}

        for task_type in ("general", "math", "vision"):
            candidates = eligibility.get(task_type)

            if not isinstance(candidates, list) or not candidates:
                raise ValueError(
                    f"Missing or empty eligibility for {task_type!r}."
                )

            if any(not isinstance(model, str) for model in candidates):
                raise ValueError("Eligible model names must be strings.")

            if len(set(candidates)) != len(candidates):
                raise ValueError(
                    f"Duplicate models for task type {task_type!r}."
                )

            active = []

            for model in candidates:
                if model not in models:
                    raise ValueError(f"Unknown model: {model!r}")

                if task_type not in models[model].get("supports", []):
                    raise ValueError(
                        f"{model!r} does not support {task_type!r}."
                    )

                if model in enabled_models:
                    active.append(model)

            self.eligibility[task_type] = tuple(active)

        self.context_builder = ContextBuilder()

        self.policy = LinUCB(
            arms=enabled_models,
            context_dimension=CONTEXT_DIMENSION,
            alpha=routing.get("alpha", 1.0),
            regularization=routing.get("regularization", 1.0),
        )

        self.feedback = FeedbackManager(
            policy=self.policy,
            learning_enabled=routing.get("learning_enabled", False),
        )
        self.checkpoint_path = (
            None
            if checkpoint_path is None
            else Path(checkpoint_path)
        )

        if (
            self.checkpoint_path is not None
            and self.checkpoint_path.exists()
        ):
            restored = FeedbackManager.load_checkpoint(
                self.checkpoint_path,
                learning_enabled=self.feedback.learning_enabled,
            )

            if restored.policy.arms != self.policy.arms:
                raise ValueError(
                    "Checkpoint model arms do not match configuration."
                )

            if (
                restored.policy.alpha != self.policy.alpha
                or restored.policy.regularization
                != self.policy.regularization
            ):
                raise ValueError(
                    "Checkpoint LinUCB settings do not match configuration."
                )

            self.feedback = restored
            self.policy = restored.policy

    def route(
        self,
        paper: PaperDoc,
        task_type: str,
        difficulty: float = 0.5,
        queue_utilization: float = 0.0,
        available_models: Optional[Iterable[str]] = None,
    ) -> RoutingDecision:
        """Select a model without reserving capacity or updating learning.

        available_models=None means all enabled models are available.
        An empty iterable means no models are available.

        The serving pool must reserve capacity after selection.
        """
        if task_type not in self.eligibility:
            raise ValueError(f"Unknown task type: {task_type!r}")

        candidates = self.eligibility[task_type]

        if available_models is not None:
            available = set(available_models)
            candidates = tuple(
                model for model in candidates
                if model in available
            )

        if not candidates:
            raise NoAvailableModelError(
                f"No available model supports {task_type!r}."
            )

        context = self.context_builder.build(
            paper=paper,
            task_type=task_type,
            difficulty=difficulty,
            queue_utilization=queue_utilization,
        )

        scores = self.policy.scores(
            context,
            eligible_arms=candidates,
        )

        # Preserve eligibility order when scores are tied.
        selected = max(scores, key=scores.get)

        decision = RoutingDecision(
            decision_id=uuid4().hex,
            paper_id=paper.id,
            task_type=task_type,
            model=selected,
            context=tuple(context),
            context_version=CONTEXT_VERSION,
            eligible_models=candidates,
            scores=tuple(scores.items()),
        )


        if self.feedback.learning_enabled:
            if self.checkpoint_path is None:
                self.feedback.register(decision)
            else:
                # Commit to disk before replacing live state.
                candidate = deepcopy(self.feedback)
                candidate.register(decision)
                candidate.save_checkpoint(self.checkpoint_path)

                self.feedback = candidate
                self.policy = candidate.policy

        return decision

    def save_checkpoint(self):
        if self.checkpoint_path is None:
            raise ValueError("No checkpoint path was configured.")

        self.feedback.save_checkpoint(self.checkpoint_path)

    def apply_feedback(self, decision_id, reward):
        """Apply feedback and persist it before reporting success.

        When no checkpoint path is supplied, feedback remains
        in memory for the current runtime only.
        """
        if self.checkpoint_path is None:
            self.feedback.apply(decision_id, reward)
            return

        candidate = deepcopy(self.feedback)
        candidate.apply(decision_id, reward)
        candidate.save_checkpoint(self.checkpoint_path)

        self.feedback = candidate
        self.policy = candidate.policy