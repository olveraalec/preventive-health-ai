"""Typed contracts connecting estimation, control, and agents."""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class ControllerInput:
    # Identifies a single independently evaluated episode.
    episode_id: str
    step: int

    # Error moments for two candidate prediction pathways.
    variance_1: float
    variance_2: float
    covariance_12: float

    # Previous controller allocation.
    previous_weight: float = 0.5

    # Estimated incremental resource cost.
    # Units must be consistent with the objective.
    resource_cost: float = 1.0

    # Optional signals for future hybrid controllers.
    state_entropy: Optional[float] = None
    information_value: Optional[float] = None

    def __post_init__(self):
        if not self.episode_id:
            raise ValueError("episode_id is required")

        if self.step < 0:
            raise ValueError("step must be nonnegative")

        values = (
            self.variance_1,
            self.variance_2,
            self.covariance_12,
            self.previous_weight,
            self.resource_cost,
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite controller input")

        if self.variance_1 < 0 or self.variance_2 < 0:
            raise ValueError("Negative variance")

        if not 0 <= self.previous_weight <= 1:
            raise ValueError("Invalid previous weight")

        if self.resource_cost < 0:
            raise ValueError("Negative resource cost")

        for optional in (
            self.state_entropy,
            self.information_value,
        ):
            if optional is not None and not math.isfinite(optional):
                raise ValueError("Nonfinite optional signal")


@dataclass(frozen=True)
class ControllerOutput:
    episode_id: str
    step: int

    weight_1: float
    weight_2: float

    estimated_risk: float
    estimated_objective: float

    state_entropy: Optional[float]
    information_value: Optional[float]

    used_fallback: bool
    policy_name: str
