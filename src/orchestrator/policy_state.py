# Filename: src/orchestrator/policy_state.py
# Created: 2026-11-17
# Created by: Thulashan
# Purpose: Save and restore LinUCB state using validated JSON,
#          including the feature definition used during training.

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from src.orchestrator.context import (
    CONTEXT_DIMENSION,
    CONTEXT_FEATURE_NAMES,
    CONTEXT_VERSION,
    PAPER_FEATURE_SCALES,
)
from src.orchestrator.linucb import LinUCB


def context_signature():
    return {
        "version": CONTEXT_VERSION,
        "dimension": CONTEXT_DIMENSION,
        "feature_names": list(CONTEXT_FEATURE_NAMES),
        "paper_feature_scales": [
            [name, scale]
            for name, scale in PAPER_FEATURE_SCALES
        ],
    }


def save_policy(policy: LinUCB, path):
    """Atomically write a policy checkpoint.

    Replacing an existing checkpoint is intentional.
    Call between updates; this is not a multi-process state store.
    """
    if policy.context_dimension != CONTEXT_DIMENSION:
        raise ValueError("Policy context dimension is incompatible.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
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

    # Validate before replacing a previously valid checkpoint.
    _decode_policy(payload)

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


def _decode_policy(payload):
    if not isinstance(payload, dict):
        raise ValueError("Policy checkpoint must be a JSON object.")

    if payload.get("format_version") != 1:
        raise ValueError("Unsupported policy checkpoint format.")

    if payload.get("context") != context_signature():
        raise ValueError(
            "Checkpoint context definition does not match this code."
        )

    try:
        arms = payload["arms"]

        if not isinstance(arms, list):
            raise ValueError("Checkpoint arms must be a list.")

        policy = LinUCB(
            arms=arms,
            context_dimension=CONTEXT_DIMENSION,
            alpha=payload["alpha"],
            regularization=payload["regularization"],
        )

        for field in ("A", "b", "update_counts"):
            values = payload[field]

            if (
                not isinstance(values, dict)
                or set(values) != set(policy.arms)
            ):
                raise ValueError(
                    f"Checkpoint {field} has incompatible model keys."
                )

        for arm in policy.arms:
            matrix = np.asarray(payload["A"][arm], dtype=float)
            vector = np.asarray(payload["b"][arm], dtype=float)
            count = payload["update_counts"][arm]

            if matrix.shape != (
                CONTEXT_DIMENSION,
                CONTEXT_DIMENSION,
            ):
                raise ValueError(f"Invalid matrix shape for {arm!r}.")

            if vector.shape != (CONTEXT_DIMENSION,):
                raise ValueError(f"Invalid vector shape for {arm!r}.")

            if (
                not np.all(np.isfinite(matrix))
                or not np.all(np.isfinite(vector))
            ):
                raise ValueError("Checkpoint contains nonfinite values.")

            if not np.allclose(
                matrix, matrix.T, rtol=1e-10, atol=1e-12
            ):
                raise ValueError("Checkpoint matrix is not symmetric.")

            # Positive definiteness is required for LinUCB solves.
            np.linalg.cholesky(matrix)

            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("Invalid policy update count.")

            policy.A[arm] = matrix.copy()
            policy.b[arm] = vector.copy()
            policy.update_counts[arm] = count

        return policy

    except (
        KeyError,
        TypeError,
        OverflowError,
        np.linalg.LinAlgError,
    ) as exc:
        raise ValueError("Invalid policy checkpoint.") from exc


def load_policy(path) -> LinUCB:
    """Load a checkpoint only if its feature definition matches."""
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return _decode_policy(payload)