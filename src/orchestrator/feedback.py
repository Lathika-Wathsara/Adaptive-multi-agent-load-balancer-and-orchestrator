# Filename: src/orchestrator/feedback.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Apply explicit reward feedback to recorded routing
#          decisions and prevent duplicate updates within a run.

import math
from dataclasses import dataclass
from typing import Tuple
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.orchestrator.policy_state import (
    context_signature,
    _decode_policy,
)


@dataclass(frozen=True)
class RecordedDecision:
    model: str
    context: Tuple[float, ...]


class FeedbackManager:
    """In-memory feedback tracking for one runtime.

    Rewards must be supplied by an external evaluator or user.
    No rewards are inferred from mock outputs or model confidence.

    Use save_checkpoint/load_checkpoint to persist the policy,
    pending decisions, and applied-feedback IDs together.
    """

    def __init__(self, policy, learning_enabled=False):
        if not isinstance(learning_enabled, bool):
            raise ValueError("learning_enabled must be a boolean.")

        self.policy = policy
        self.learning_enabled = learning_enabled
        self._pending = {}
        self._applied = set()

    @property
    def pending_count(self):
        return len(self._pending)

    def register(self, decision):
        decision_id = decision.decision_id

        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("Decision ID must be a nonempty string.")

        if (
            decision_id in self._pending
            or decision_id in self._applied
        ):
            raise ValueError("Decision ID is already registered.")

        if decision.model not in self.policy.arms:
            raise ValueError("Decision model is not a policy arm.")

        context = tuple(float(value) for value in decision.context)

        if len(context) != self.policy.context_dimension:
            raise ValueError("Decision context has the wrong dimension.")

        if not all(math.isfinite(value) for value in context):
            raise ValueError("Decision context must contain finite values.")

        self._pending[decision_id] = RecordedDecision(
            model=decision.model,
            context=context,
        )

    def apply(self, decision_id, reward):
        if not self.learning_enabled:
            raise RuntimeError("Policy learning is disabled.")

        if decision_id in self._applied:
            raise ValueError("Feedback was already applied.")

        if decision_id not in self._pending:
            raise ValueError("Unknown decision ID.")

        decision = self._pending[decision_id]

        # Update first: invalid rewards must leave the decision pending.
        self.policy.update(
            arm=decision.model,
            context=decision.context,
            reward=reward,
        )

        self._applied.add(decision_id)
        del self._pending[decision_id]

    def save_checkpoint(self, path):
        """Save policy and feedback bookkeeping in one atomic file.

        Use from one runtime process, between feedback updates.
        Concurrent processes must not share this checkpoint.
        """
        policy = self.policy

        policy_payload = {
            "format_version": 1,
            "context": context_signature(),
            "arms": list(policy.arms),
            "alpha": policy.alpha,
            "regularization": policy.regularization,
            "A": {
                arm: policy.A[arm].tolist()
                for arm in policy.arms
            },
            "b": {
                arm: policy.b[arm].tolist()
                for arm in policy.arms
            },
            "update_counts": dict(policy.update_counts),
        }

        # Reuse the existing policy-state validation.
        _decode_policy(policy_payload)

        payload = {
            "checkpoint_version": 1,
            "policy": policy_payload,
            "pending": {
                decision_id: {
                    "model": decision.model,
                    "context": list(decision.context),
                }
                for decision_id, decision in self._pending.items()
            },
            "applied": sorted(self._applied),
        }

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)

                json.dump(
                    payload,
                    file,
                    indent=2,
                    allow_nan=False,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_path, destination)

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load_checkpoint(cls, path, learning_enabled=False):
        """Restore state; learning must be explicitly enabled again."""
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if (
            not isinstance(payload, dict)
            or payload.get("checkpoint_version") != 1
        ):
            raise ValueError("Unsupported feedback checkpoint.")

        policy = _decode_policy(payload.get("policy"))

        manager = cls(
            policy,
            learning_enabled=learning_enabled,
        )

        pending = payload.get("pending")
        applied = payload.get("applied")

        if not isinstance(pending, dict):
            raise ValueError("Invalid pending-decision records.")

        if (
            not isinstance(applied, list)
            or any(
                not isinstance(value, str) or not value
                for value in applied
            )
        ):
            raise ValueError("Invalid applied-decision IDs.")

        if len(applied) != len(set(applied)):
            raise ValueError("Duplicate applied-decision IDs.")

        if set(pending) & set(applied):
            raise ValueError(
                "A decision cannot be both pending and applied."
            )

        if len(applied) > sum(policy.update_counts.values()):
            raise ValueError(
                "Feedback records exceed policy update counts."
            )

        for decision_id, record in pending.items():
            if (
                not isinstance(record, dict)
                or "model" not in record
                or not isinstance(record.get("context"), list)
            ):
                raise ValueError("Invalid pending-decision record.")

            manager.register(
                SimpleNamespace(
                    decision_id=decision_id,
                    model=record["model"],
                    context=record["context"],
                )
            )

        manager._applied = set(applied)
        return manager

        self._applied.add(decision_id)
        del self._pending[decision_id]