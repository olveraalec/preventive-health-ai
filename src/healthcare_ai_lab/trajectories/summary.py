"""Read-only v0.6 arithmetic attribution of existing inference outputs.

No labels, metric evaluators, inference calls, retained episode state or forecasts.
"""

import math

from healthcare_ai_lab.trajectories.contracts import (
    ControllerExplanation,
    DataQuality,
    DecisionDiagnostics,
    DecisionProvenance,
    DecisionState,
    DecisionSummary,
    HourPrediction,
    ProbabilityChange,
    TrajectoryRequest,
    UnavailableCapability,
)


def probability_change(
    previous: HourPrediction | None, current: HourPrediction
) -> ProbabilityChange:
    """Exact ordered identity from v0.6; do not impute an unavailable expert."""
    values = {
        "status": "unavailable",
        "reason": "first_observation",
        "probability_delta": None,
        "static_delta": None,
        "disagreement_change_at_previous_allocation": None,
        "reallocation_at_current_disagreement": None,
        "reconstruction_residual": None,
    }
    if previous is None:
        return ProbabilityChange(**values)
    delta = current.final_probability - previous.final_probability
    static_delta = current.static_probability - previous.static_probability
    values.update(probability_delta=delta, static_delta=static_delta)
    if previous.temporal_probability is None or current.temporal_probability is None:
        return ProbabilityChange(
            **{**values, "reason": "secondary_probability_unavailable"}
        )
    old_d = previous.temporal_probability - previous.static_probability
    new_d = current.temporal_probability - current.static_probability
    disagreement = previous.temporal_weight * (new_d - old_d)
    reallocation = (current.temporal_weight - previous.temporal_weight) * new_d
    return ProbabilityChange(
        status="available",
        reason=None,
        probability_delta=delta,
        static_delta=static_delta,
        disagreement_change_at_previous_allocation=disagreement,
        reallocation_at_current_disagreement=reallocation,
        reconstruction_residual=delta
        - math.fsum((static_delta, disagreement, reallocation)),
    )


def decision_state(prediction: HourPrediction) -> DecisionState:
    return DecisionState(
        **prediction.model_dump(),
        static_contribution=prediction.static_weight * prediction.static_probability,
        temporal_contribution=(
            prediction.temporal_weight * prediction.temporal_probability
            if prediction.temporal_probability is not None
            else None
        ),
    )


def decision_summaries(
    bundle: dict,
    request: TrajectoryRequest,
    predictions: list[HourPrediction],
    controller_details: list[dict],
) -> list[DecisionSummary]:
    """Use only the current prefix and verified/declared existing metadata."""
    manifest = bundle.get("artifact_manifest")
    provenance = DecisionProvenance(
        model=request.model,
        artifact_sha256=manifest["artifact_sha256"] if manifest else None,
        artifact_status="manifest_validated" if manifest else "manifest_unavailable",
        features=request.provenance,
    )
    summaries = []
    previous = None
    for observation, prediction, details in zip(
        request.observations, predictions, controller_details, strict=True
    ):
        p = prediction.final_probability
        entropy = -sum(v * math.log(v) for v in (p, 1 - p) if v > 0)
        d = (
            prediction.temporal_probability - prediction.static_probability
            if prediction.temporal_probability is not None
            else None
        )
        fallback = prediction.status == "static_only_fallback"
        summaries.append(
            DecisionSummary(
                episode_id=request.episode_id,
                current=decision_state(prediction),
                previous=decision_state(previous) if previous else None,
                elapsed_hours=prediction.icu_hour - previous.icu_hour
                if previous
                else None,
                change=probability_change(previous, prediction),
                diagnostics=DecisionDiagnostics(
                    signed_model_disagreement=d,
                    absolute_model_disagreement=abs(d) if d is not None else None,
                    final_predictive_entropy_nats=entropy,
                ),
                data_quality=DataQuality(
                    status="unavailable",
                    reason="raw_measurements_not_supplied",
                    engineered_feature_count=len(observation.features),
                    engineered_missing_count=sum(
                        v is None for v in observation.features.values()
                    ),
                ),
                controller=ControllerExplanation(
                    anticipatory_proposal=details.get("anticipatory_proposal"),
                    estimated_gain=details.get("estimated_gain"),
                    fallback_active=fallback,
                    fallback_reason="secondary_inference_failed" if fallback else None,
                    emergency_overrides_step_limit=fallback,
                    configured_constraints={
                        "fallback_weight": 0.0,
                        "max_deviation": 0.25,
                        "max_step": 0.10,
                    },
                    reactive=UnavailableCapability(
                        reason="no_verified_outcome_feedback"
                    ),
                    covariance=UnavailableCapability(
                        reason="not_integrated_in_this_policy"
                    ),
                ),
                provenance=provenance,
                next_hour_forecast=UnavailableCapability(
                    reason="no_validated_next_hour_forecaster"
                ),
                learned_hybrid=UnavailableCapability(
                    reason="not_integrated_or_validated"
                ),
                workflow_recommendation=UnavailableCapability(
                    reason="no_supported_recommendation_layer"
                ),
            )
        )
        previous = prediction
    return summaries
