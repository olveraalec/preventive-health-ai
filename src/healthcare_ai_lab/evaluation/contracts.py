"""Frozen, task-specific evaluation contracts; outcome labels are not predictions."""

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[Finite, Field(ge=0, le=1)]
Identifier = Annotated[str, Field(min_length=1)]
PolicyKind = Literal[
    "static",
    "temporal",
    "constant_mixture",
    "hard_switch",
    "continuous_allocation",
    "hybrid",
]
METRIC_VERSION = "sepsis-classification.v1"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationConfig(Contract):
    epsilon: Annotated[Finite, Field(gt=0, lt=0.5)] = 1e-10
    calibration_bins: int = Field(default=10, ge=1, strict=True)
    n_bootstrap: int = Field(default=2000, ge=1, strict=True)
    n_permutations: int = Field(default=2000, ge=1, strict=True)
    seed: int = Field(default=12345, ge=0, strict=True)
    switching_threshold: Probability = 0.5
    allocation_change_tolerance: Probability = 0.0
    near_static_tolerance: Probability = 0.01

    @field_validator("epsilon")
    @classmethod
    def representable_score_clip(cls, value):
        if 1 - value == 1:
            raise ValueError(
                "epsilon must produce a representable upper clipping endpoint"
            )
        return value


class PredictionRecord(Contract):
    """A frozen output, with all times expressed on the episode's ICU-hour clock."""

    dataset: Literal["sepsis2019"] = "sepsis2019"
    split: Identifier
    episode_id: Identifier
    icu_hour: Annotated[Finite, Field(gt=0)]
    decision_time: Annotated[Finite, Field(ge=0)]
    observation_cutoff: Annotated[Finite, Field(ge=0)]
    replay_marker: Literal["retrospective"] = "retrospective"
    target_semantics: Literal["shifted_challenge_label"] = "shifted_challenge_label"
    policy_id: Identifier
    policy_kind: PolicyKind
    policy_version: Identifier
    artifact_version: Identifier
    feature_schema_version: Identifier
    static_probability: Probability
    temporal_probability: Probability | None
    final_probability: Probability
    q_temporal: Probability
    constant_weight: Probability | None = None
    status: Literal["ok", "static_only_fallback"] = "ok"
    fallback_reason: Identifier | None = None
    feedback_hours: tuple[Annotated[Finite, Field(gt=0)], ...] = ()
    feedback_protocol: Literal[
        "none", "verified_availability", "oracle_delay_simulation"
    ] = "none"

    @model_validator(mode="after")
    def validate_frozen_prediction(self):
        if self.policy_kind == "constant_mixture" and self.constant_weight is None:
            raise ValueError("Constant policy requires its configured reference weight")
        if (
            self.icu_hour > self.decision_time
            or self.observation_cutoff > self.decision_time
        ):
            raise ValueError("Prediction uses a future observation/decision clock")
        if self.temporal_probability is None and self.q_temporal != 0:
            raise ValueError("Unavailable temporal expert requires zero allocation")
        expected = (
            self.static_probability
            if self.q_temporal == 0
            else (
                (1 - self.q_temporal) * self.static_probability
                + self.q_temporal * self.temporal_probability
            )
        )
        if not np.isclose(self.final_probability, expected, rtol=0, atol=1e-12):
            raise ValueError("Final probability does not match the declared mixture")
        if self.status == "static_only_fallback":
            if self.q_temporal != 0 or self.fallback_reason is None:
                raise ValueError(
                    "Emergency fallback requires static allocation and a reason"
                )
        elif self.policy_kind == "static" and self.q_temporal != 0:
            raise ValueError("Static policy must allocate zero temporal weight")
        elif self.policy_kind == "temporal" and self.q_temporal != 1:
            raise ValueError("Temporal policy must allocate unit temporal weight")
        elif self.policy_kind == "hard_switch" and self.q_temporal not in (0, 1):
            raise ValueError("Hard switch requires an endpoint allocation")
        elif self.policy_kind == "constant_mixture":
            if self.constant_weight is None or self.q_temporal != self.constant_weight:
                raise ValueError("Constant policy must match its declared constant")
        if len(set(self.feedback_hours)) != len(self.feedback_hours):
            raise ValueError("Duplicate feedback reference")
        if self.feedback_hours and (
            self.feedback_protocol == "none"
            or any(hour >= self.icu_hour for hour in self.feedback_hours)
        ):
            raise ValueError(
                "Feedback must reference earlier observations with an explicit protocol"
            )
        return self


class OutcomeRecord(Contract):
    """Evaluation-only labels; availability is a declared source assertion."""

    dataset: Literal["sepsis2019"] = "sepsis2019"
    split: Identifier
    episode_id: Identifier
    icu_hour: Annotated[Finite, Field(gt=0)]
    y_true: Literal[0, 1]
    outcome_time: Annotated[Finite, Field(ge=0)]
    available_at: Annotated[Finite, Field(ge=0)]
    availability_basis: Literal["retrospective_only", "verified_independent"] = (
        "retrospective_only"
    )

    @model_validator(mode="after")
    def ordered_availability(self):
        if self.outcome_time < self.icu_hour or self.available_at < self.outcome_time:
            raise ValueError("Outcome availability precedes its target/event time")
        return self


class MetricDefinition(Contract):
    metric_id: str
    task: Literal["classification", "controller"] = "classification"
    distribution: Literal["Bernoulli", "allocation"] = "Bernoulli"
    direction: Literal["minimize", "maximize", "target", "none"]
    role: Literal["objective", "monitoring"]
    unit: str
    aggregation: str
    valid_input_shape: str = "aligned nonempty 1D arrays"
    probability_clip: float | None = None
    undefined_behavior: str = "null value with explicit reason"
    metric_version: str = METRIC_VERSION


class MetricValue(Contract):
    metric: MetricDefinition
    value: Finite | None
    status: Literal["defined", "undefined"]
    reason: str | None
    n_observations: int = Field(ge=0)
    n_patients: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def consistent_status(self):
        if (self.status == "defined") != (self.value is not None):
            raise ValueError("Metric status/value disagree")
        if self.status == "undefined" and not self.reason:
            raise ValueError("Undefined metrics require a reason")
        return self


def metric_value(
    name,
    value,
    *,
    n,
    patients=None,
    direction="none",
    role="monitoring",
    unit="unitless",
    aggregation="hour_weighted",
    reason=None,
    epsilon=None,
    controller=False,
):
    return MetricValue(
        metric=MetricDefinition(
            metric_id=name,
            direction=direction,
            role=role,
            unit=unit,
            aggregation=aggregation,
            probability_clip=epsilon,
            task="controller" if controller else "classification",
            distribution="allocation" if controller else "Bernoulli",
        ),
        value=None if value is None else float(value),
        status="undefined" if value is None else "defined",
        reason=reason if value is None else None,
        n_observations=int(n),
        n_patients=patients,
    )
