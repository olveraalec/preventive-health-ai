"""Deterministic, constrained two-pathway controller.

No LangGraph, LLM, database, or network dependencies.
Uses PSD error moments and aligned evaluation penalties.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ControllerConfig:
    complexity_penalty: float = 0.5
    stability_penalty: float = 0.5
    memory_penalty: float = 0.0

    reference_weight: float = 0.5
    min_weight: float = 0.0
    max_weight: float = 1.0

    def __post_init__(self):
        penalties = (
            self.complexity_penalty,
            self.stability_penalty,
            self.memory_penalty,
        )
        if not all(np.isfinite(x) and x >= 0 for x in penalties):
            raise ValueError("Penalties must be finite and nonnegative")

        bounds = (
            self.min_weight,
            self.max_weight,
            self.reference_weight,
        )
        if not all(np.isfinite(x) for x in bounds):
            raise ValueError("Weights must be finite")

        if not (0 <= self.min_weight <= self.reference_weight <= self.max_weight <= 1):
            raise ValueError("Invalid allocation bounds")


@dataclass(frozen=True)
class Decision:
    weight: float
    estimated_risk: float
    estimated_objective: float
    used_fallback: bool


def project_covariance(v1, v4, cross):
    """Project an estimated 2x2 covariance matrix onto PSD."""
    matrix = np.array(
        [
            [v1, cross],
            [cross, v4],
        ],
        dtype=float,
    )

    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite covariance estimate")

    matrix = 0.5 * (matrix + matrix.T)

    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 0.0)

    projected = vectors @ np.diag(values) @ vectors.T

    return (
        float(projected[0, 0]),
        float(projected[1, 1]),
        float(projected[0, 1]),
    )


def estimated_risk(weight, moments):
    """Expected squared error of the weighted prediction."""
    v1, v4, cross = moments
    w = float(weight)

    return float((1 - w) ** 2 * v1 + w**2 * v4 + 2 * w * (1 - w) * cross)


def objective_terms(
    weight,
    previous_weight,
    moments,
    config,
    memory_cost=1.0,
):
    """Controller objective; also used for evaluation."""
    if not np.isfinite(memory_cost) or memory_cost < 0:
        raise ValueError("Invalid memory cost")

    w = float(weight)
    previous = float(previous_weight)

    risk = estimated_risk(w, moments)

    complexity = config.complexity_penalty * (w - config.reference_weight) ** 2
    stability = config.stability_penalty * (w - previous) ** 2
    resource = config.memory_penalty * memory_cost * w

    return {
        "risk": risk,
        "complexity": complexity,
        "stability": stability,
        "resource": resource,
        "total": risk + complexity + stability + resource,
    }


def optimize_weight(
    moments,
    previous_weight,
    config=None,
    memory_cost=1.0,
):
    """Minimize the convex quadratic over bounded weights."""
    config = config or ControllerConfig()

    if not np.isfinite(previous_weight):
        raise ValueError("Invalid previous weight")

    if not np.isfinite(memory_cost) or memory_cost < 0:
        raise ValueError("Invalid memory cost")

    moments = project_covariance(*moments)
    v1, v4, cross = moments

    variation = max(v1 + v4 - 2 * cross, 0.0)

    denominator = variation + config.complexity_penalty + config.stability_penalty

    numerator = (
        v1
        - cross
        + config.complexity_penalty * config.reference_weight
        + config.stability_penalty * previous_weight
        - 0.5 * config.memory_penalty * memory_cost
    )

    fallback = denominator <= 1e-12

    if fallback:
        proposed = config.reference_weight
    else:
        proposed = numerator / denominator

    weight = float(
        np.clip(
            proposed,
            config.min_weight,
            config.max_weight,
        )
    )

    terms = objective_terms(
        weight,
        previous_weight,
        moments,
        config,
        memory_cost,
    )

    return Decision(
        weight=weight,
        estimated_risk=terms["risk"],
        estimated_objective=terms["total"],
        used_fallback=fallback,
    )


def realized_objective(
    y_true,
    y_pred,
    weights,
    config,
    memory_costs=None,
):
    """Observed objective with matching decision penalties.

    Previous weight resets to reference at episode start.
    All arrays must represent one chronological episode.
    """
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(weights, dtype=float)

    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Expected nonempty 1D arrays")

    if pred.shape != y.shape or w.shape != y.shape:
        raise ValueError("Array shapes do not match")

    if not all(np.isfinite(x).all() for x in (y, pred, w)):
        raise ValueError("Nonfinite evaluation input")

    if np.any(w < config.min_weight) or np.any(w > config.max_weight):
        raise ValueError("Weights outside allowed bounds")

    if memory_costs is None:
        costs = np.ones(len(y))
    else:
        costs = np.asarray(memory_costs, dtype=float)

    if costs.shape != y.shape or not np.isfinite(costs).all():
        raise ValueError("Invalid memory costs")

    if np.any(costs < 0):
        raise ValueError("Memory costs cannot be negative")

    previous = np.concatenate(
        (
            [config.reference_weight],
            w[:-1],
        )
    )

    terms = {
        "realized_mse": float(np.mean((y - pred) ** 2)),
        "complexity": float(
            np.mean(config.complexity_penalty * (w - config.reference_weight) ** 2)
        ),
        "stability": float(np.mean(config.stability_penalty * (w - previous) ** 2)),
        "resource": float(np.mean(config.memory_penalty * costs * w)),
    }

    terms["total"] = sum(terms.values())
    return terms
