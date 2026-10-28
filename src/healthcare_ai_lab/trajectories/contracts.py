"""Transport contracts; provenance declarations are not preprocessing verification."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[Finite, Field(ge=0, le=1)]
Identifier = Annotated[str, Field(min_length=1)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeatureProvenance(Contract):
    feature_schema: Identifier
    preprocessing_version: Identifier
    pipeline_id: Identifier
    declaration: Literal["current_and_past_only"]


class ModelMetadata(Contract):
    artifact_id: Identifier
    model_version: Identifier


class HourObservation(Contract):
    episode_id: Identifier
    icu_hour: Annotated[Finite, Field(gt=0)]
    features: dict[str, Finite | None]


class TrajectoryRequest(Contract):
    dataset: Literal["sepsis2019"]
    episode_id: Identifier
    provenance: FeatureProvenance
    model: ModelMetadata
    observations: list[HourObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_history(self):
        previous = None
        reserved = {"ICULOS", "episode_id", "patient_id", "SepsisLabel", "split"}
        for row in self.observations:
            if row.episode_id != self.episode_id:
                raise ValueError("Mixed episode identifiers")
            if previous is not None and row.icu_hour <= previous:
                raise ValueError("ICU hours must be strictly increasing and unique")
            if reserved.intersection(row.features):
                raise ValueError("Reserved identity, time or evaluation feature")
            previous = row.icu_hour
        return self


class HourPrediction(Contract):
    icu_hour: Finite
    static_probability: Probability
    temporal_probability: Probability | None
    final_probability: Probability
    temporal_weight: Probability
    static_weight: Probability
    status: Literal["bounded_anticipatory", "static_only_fallback"]
    reason: Literal["no_verified_feedback", "secondary_inference_failed"]


class UnavailableCapability(Contract):
    status: Literal["unavailable"] = "unavailable"
    value: None = None
    reason: Identifier


class DecisionState(HourPrediction):
    """Already computed values, never a second inference or a forecast."""

    static_contribution: Probability
    temporal_contribution: Probability | None


class ProbabilityChange(Contract):
    status: Literal["available", "unavailable"]
    reason: Literal["first_observation", "secondary_probability_unavailable"] | None
    interval: Literal["since_previous_observed_hour"] = "since_previous_observed_hour"
    probability_delta: Finite | None
    static_delta: Finite | None
    disagreement_change_at_previous_allocation: Finite | None
    reallocation_at_current_disagreement: Finite | None
    reconstruction_residual: Finite | None
    interpretation: Literal["arithmetic_attribution_not_medical_causality"] = (
        "arithmetic_attribution_not_medical_causality"
    )


class MeasurementAvailability(Contract):
    observed: bool
    age_hours: Annotated[Finite, Field(ge=0)] | None


class DataQuality(Contract):
    status: Literal["available", "unavailable"]
    reason: Literal["raw_measurements_not_supplied"] | None
    scope: Literal["sepsis_vitals"] = "sepsis_vitals"
    source_availability_verified: Literal[False] = False
    measurements: dict[str, MeasurementAvailability] = Field(default_factory=dict)
    missing_count: int | None = Field(default=None, ge=0)
    missing_fraction: Probability | None = None
    engineered_feature_count: int = Field(ge=0)
    engineered_missing_count: int = Field(ge=0)


class DecisionDiagnostics(Contract):
    signed_model_disagreement: Finite | None
    absolute_model_disagreement: Probability | None
    final_predictive_entropy_nats: Annotated[Finite, Field(ge=0)]
    entropy_interpretation: Literal["predictive_distribution_not_reliability"] = (
        "predictive_distribution_not_reliability"
    )


class ControllerExplanation(Contract):
    policy_id: Literal["bounded_anticipatory_research"] = "bounded_anticipatory_research"
    allocation_basis: Literal["learned_gain_with_heuristic_exposure_and_bounds"] = (
        "learned_gain_with_heuristic_exposure_and_bounds"
    )
    empirically_validated: Literal[False] = False
    exposure: Probability = 0.25
    exposure_empirically_calibrated: Literal[False] = False
    anticipatory_proposal: Probability | None
    estimated_gain: Finite | None
    gain_interpretation: Literal["training_gain_surrogate_not_verified_live_benefit"] = (
        "training_gain_surrogate_not_verified_live_benefit"
    )
    fallback_active: bool
    fallback_reason: Literal["secondary_inference_failed"] | None
    emergency_overrides_step_limit: bool
    configured_constraints: dict[str, float]
    constraint_trace: Literal["binding_constraints_not_recorded"] = (
        "binding_constraints_not_recorded"
    )
    reactive: UnavailableCapability
    covariance: UnavailableCapability


class DecisionProvenance(Contract):
    dataset: Literal["sepsis2019"] = "sepsis2019"
    target: Literal["shifted_challenge_label"] = "shifted_challenge_label"
    model: ModelMetadata
    artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None
    artifact_status: Literal["manifest_validated", "manifest_unavailable"]
    features: FeatureProvenance
    feature_status: Literal[
        "declared_compatible_not_independently_verified",
        "runtime_generated_source_availability_unverified",
    ] = "declared_compatible_not_independently_verified"


class LiveEvaluationAvailability(UnavailableCapability):
    reason: Literal["no_recorded_evaluation_bound_to_this_model"] = (
        "no_recorded_evaluation_bound_to_this_model"
    )
    recorded_summary_path: Literal["/api/v1/research/evaluation-summary"] = (
        "/api/v1/research/evaluation-summary"
    )


class DecisionSummary(Contract):
    schema_version: Literal["decision-summary.v0.6"] = "decision-summary.v0.6"
    episode_id: Identifier
    current: DecisionState
    previous: DecisionState | None
    elapsed_hours: Annotated[Finite, Field(gt=0)] | None
    change: ProbabilityChange
    diagnostics: DecisionDiagnostics
    data_quality: DataQuality
    controller: ControllerExplanation
    provenance: DecisionProvenance
    evaluation: LiveEvaluationAvailability = Field(default_factory=LiveEvaluationAvailability)
    next_computational_step: Literal[
        "await_next_valid_observation_then_update_causal_features_and_reevaluate"
    ] = "await_next_valid_observation_then_update_causal_features_and_reevaluate"
    next_hour_forecast: UnavailableCapability
    learned_hybrid: UnavailableCapability
    workflow_recommendation: UnavailableCapability
    research_only: Literal[True] = True
    clinical_use: Literal[False] = False


class TrajectoryResponse(Contract):
    dataset: Literal["sepsis2019"] = "sepsis2019"
    episode_id: Identifier
    provenance: FeatureProvenance
    provenance_status: Literal["declared_compatible_not_independently_verified"] = (
        "declared_compatible_not_independently_verified"
    )
    model: ModelMetadata
    predictions: list[HourPrediction]
    decision_summaries: list[DecisionSummary] = Field(default_factory=list)
    probability_type: Literal["shifted_challenge_label"] = "shifted_challenge_label"
    research_only: Literal[True] = True
    clinical_use: Literal[False] = False
    configured_parameters: dict[str, float]
    parameters_empirically_calibrated: Literal[False] = False
