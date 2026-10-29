"""Additive dashboard transport; no outcomes, clinical orders or future inputs."""

from typing import Annotated, Literal

from pydantic import Field

from healthcare_ai_lab.trajectories.contracts import (
    Contract,
    FeatureProvenance,
    Finite,
    HourPrediction,
    Identifier,
    ModelMetadata,
    Probability,
)

Nonnegative = Annotated[Finite, Field(ge=0)]


class PredictionChange(Contract):
    status: Literal["first_observation", "exact", "partial_temporal_unavailable"]
    since: Literal["previous_observed_hour"] = "previous_observed_hour"
    elapsed_hours: Nonnegative | None = None
    total: Finite | None = None
    static_movement: Finite | None = None
    disagreement_movement: Finite | None = None
    controller_reallocation: Finite | None = None
    identity_residual: Finite | None = None
    unsupported_terms: list[str] = Field(default_factory=list)


class Measurement(Contract):
    current_value: Finite | None
    missing: bool
    last_value: Finite | None = None
    age_hours: Nonnegative | None = None
    last_observed_hour: Nonnegative | None = None
    freshness_status: Literal["observed_hour_proxy", "never_observed", "not_tracked"]
    used_by_feature_builder: bool


class InformationQuality(Contract):
    observed_vitals: int = Field(ge=0, le=7)
    total_vitals: Literal[7] = 7
    vital_completeness: Probability
    missing_vitals: list[str]
    availability_status: Literal["source_units_and_availability_unverified"] = (
        "source_units_and_availability_unverified"
    )


class PredictionDiagnostics(Contract):
    predictive_entropy_nats: Nonnegative
    expert_gap: Finite | None
    absolute_expert_gap: Nonnegative | None
    interpretation: Literal["descriptive_not_calibrated_reliability"] = (
        "descriptive_not_calibrated_reliability"
    )


class ControllerInterpretation(Contract):
    policy: Literal["bounded_anticipatory"] = "bounded_anticipatory"
    mode: Literal["bounded_anticipatory", "static_only_fallback"]
    reason: Literal["no_verified_feedback", "secondary_inference_failed"]
    emergency_fallback: bool
    step_limit_overridden: bool
    active_constraints: list[str]
    evidence: Literal["learned_gain_configured_bounds_no_validated_superiority"] = (
        "learned_gain_configured_bounds_no_validated_superiority"
    )
    feedback: Literal["unavailable"] = "unavailable"


class LogOddsContribution(Contract):
    status: Literal["available", "unavailable"]
    reason: str | None = None
    intercept: Finite | None = None
    terms: dict[str, Finite] | None = None
    total_log_odds: Finite | None = None
    reconstructed_probability: Probability | None = None
    interpretation: Literal["fitted_preprocessing_log_odds_not_causal"] = (
        "fitted_preprocessing_log_odds_not_causal"
    )


class NextStep(Contract):
    status: Literal["await_next_valid_observation"] = "await_next_valid_observation"
    then: tuple[
        Literal["update_causal_temporal_features"],
        Literal["reevaluate_inference_and_allocation"],
    ] = ("update_causal_temporal_features", "reevaluate_inference_and_allocation")


class UnavailableCapability(Contract):
    status: Literal["unavailable"] = "unavailable"
    reason: str


class UnavailableForecast(UnavailableCapability):
    horizon_hours: Literal[1] = 1
    target: Literal["shifted_challenge_label_at_t_plus_1"] = (
        "shifted_challenge_label_at_t_plus_1"
    )
    probability: None = None
    model_version: None = None


class Capabilities(Contract):
    next_hour_forecast: UnavailableForecast = Field(
        default_factory=lambda: UnavailableForecast(
            reason="no_separately_validated_forecast"
        )
    )
    learned_hybrid: UnavailableCapability = Field(
        default_factory=lambda: UnavailableCapability(
            reason="no_validated_serving_supervisor"
        )
    )
    clinician_review: UnavailableCapability = Field(
        default_factory=lambda: UnavailableCapability(
            reason="no_validated_recommendation_policy_abstain"
        )
    )


class DecisionRecord(Contract):
    current: HourPrediction
    previous: HourPrediction | None
    change: PredictionChange
    diagnostics: PredictionDiagnostics
    information_quality: InformationQuality
    measurements: dict[str, Measurement] | None
    controller: ControllerInterpretation
    next_step: NextStep = Field(default_factory=NextStep)
    log_odds_contributions: (
        dict[Literal["static", "temporal"], LogOddsContribution] | None
    ) = None


class DecisionSummaryResponse(Contract):
    schema_version: Literal["sepsis-decision-summary.v1"] = "sepsis-decision-summary.v1"
    dataset: Literal["sepsis2019"] = "sepsis2019"
    episode_id: Identifier
    target_semantics: Literal["shifted_challenge_label"] = "shifted_challenge_label"
    model: ModelMetadata
    artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None
    artifact_integrity: Literal["checksum_bound_manifest", "declared_only"]
    provenance: FeatureProvenance
    feature_transform_sha256: str
    model_feature_hashes: dict[str, str]
    configured_parameters: dict[str, Finite]
    records: list[DecisionRecord] = Field(min_length=1)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    payload_kind: Literal[
        "api", "synthetic_example", "sanitized_retrospective_demo"
    ] = "api"
    research_only: Literal[True] = True
    clinical_use: Literal[False] = False
    limitations: tuple[str, ...] = (
        "Shifted challenge label, not validated clinical onset risk.",
        "Arithmetic and log-odds attribution are not physiological causality.",
        "Measurement units and real availability timestamps are source assertions.",
        "Controller bounds are configured; adaptive superiority is not established.",
        "No future observations, outcomes or clinical treatment recommendations are used.",
    )
