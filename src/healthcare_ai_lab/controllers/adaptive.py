"""Bounded continuous predictive-pathway controller."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ControllerConfig:
    fallback_weight: float = 0.0
    max_deviation: float = 0.25
    max_step: float = 0.10


def adaptive_allocation(
    q_anticipatory: float,
    q_reactive: float,
    reliability: float,
    anticipatory_share: float,
    feedback_available: bool,
    config: ControllerConfig = ControllerConfig(),
    previous_q: float | None = None,
) -> float:
    """Compute a bounded continuous allocation.

    All inputs must be known at decision time.
    """

    values = [
        q_anticipatory,
        q_reactive,
        reliability,
        anticipatory_share,
        config.fallback_weight,
        config.max_deviation,
        config.max_step,
    ]
    if previous_q is not None:
        values.append(previous_q)

    if not all(np.isfinite(v) for v in values):
        raise ValueError("Controller inputs must be finite")

    if not all(
        0 <= v <= 1
        for v in [
            q_anticipatory,
            q_reactive,
            reliability,
            anticipatory_share,
            config.fallback_weight,
        ]
    ):
        raise ValueError("Controller weights must be in [0, 1]")

    if not (0 <= config.max_deviation <= 1 and 0 <= config.max_step <= 1):
        raise ValueError("Invalid controller constraints")

    if previous_q is not None and not 0 <= previous_q <= 1:
        raise ValueError("previous_q must be in [0, 1]")

    # Without verified feedback, only anticipation is eligible.
    if not feedback_available:
        anticipatory_share = 1.0

    proposal = (
        anticipatory_share * q_anticipatory + (1 - anticipatory_share) * q_reactive
    )

    fallback = config.fallback_weight

    # Reliability-controlled exposure to adaptation.
    q = fallback + reliability * (proposal - fallback)

    # Constrain departure from fallback.
    q = np.clip(
        q,
        max(0, fallback - config.max_deviation),
        min(1, fallback + config.max_deviation),
    )

    # Constrain consecutive changes.
    # At the first observation, do not apply this constraint.
    if previous_q is not None:
        q = np.clip(q, previous_q - config.max_step, previous_q + config.max_step)

    return float(np.clip(q, 0.0, 1.0))


def blend_probabilities(
    p_static: float, p_temporal: float, temporal_weight: float
) -> float:
    values = (p_static, p_temporal, temporal_weight)

    if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Probabilities and weights must be in [0, 1]")

    return float((1 - temporal_weight) * p_static + temporal_weight * p_temporal)
