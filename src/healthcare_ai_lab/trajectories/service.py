"""Stateless application service for declared engineered Sepsis histories."""

import math

import pandas as pd

from healthcare_ai_lab.controllers.adaptive import ControllerConfig
from healthcare_ai_lab.models.artifact import ArtifactManifest, validate_manifest
from healthcare_ai_lab.models.sepsis_inference import predict_engineered_trajectory
from healthcare_ai_lab.trajectories.contracts import (
    HourPrediction,
    TrajectoryRequest,
    TrajectoryResponse,
)
from healthcare_ai_lab.trajectories.summary import decision_summaries


class ArtifactCompatibilityError(ValueError):
    """The installed artifact cannot support the requested contract."""


def predict_trajectory(bundle: dict, request: TrajectoryRequest) -> TrajectoryResponse:
    """Check declared compatibility; never certify how client features were built.

    The trusted artifact must contain trajectory_metadata with dataset, provenance
    and model declarations matching the request. Legacy row inference is unaffected.
    No patient/controller state is retained by this service.
    """
    if "artifact_manifest" in bundle:
        validate_manifest(
            bundle, ArtifactManifest.model_validate(bundle["artifact_manifest"])
        )
    expected = {
        "dataset": request.dataset,
        "provenance": request.provenance.model_dump(),
        "model": request.model.model_dump(),
    }
    if bundle.get("trajectory_metadata") != expected:
        raise ArtifactCompatibilityError("Model/feature provenance incompatibility")
    temperature = float(bundle.get("temperature", 0.01))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ArtifactCompatibilityError("Invalid configured temperature")
    # Fix the static fallback for this vertical slice. Both feasible intervals
    # contain zero, including after an immediate failure reset.
    if float(bundle.get("fallback_q", 0.0)) != 0.0:
        raise ArtifactCompatibilityError("Trajectory requires static fallback_q=0")
    required = (
        set(bundle["static"]["features"])
        | set(bundle["temporal"]["features"])
        | set(bundle["meta_features"])
    ) - {"ICULOS", "entropy_static"}
    forbidden = {"episode_id", "patient_id", "SepsisLabel", "split"}
    if required & forbidden or "entropy_static" in (
        set(bundle["static"]["features"]) | set(bundle["temporal"]["features"])
    ):
        raise ArtifactCompatibilityError("Invalid model feature schema")
    for row in request.observations:
        if required - row.features.keys():
            raise ValueError("Missing engineered features required by model schema")
    frame = pd.DataFrame(
        [{**row.features, "ICULOS": row.icu_hour} for row in request.observations]
    ).astype(float)
    config = ControllerConfig(fallback_weight=0.0, max_deviation=0.25, max_step=0.10)
    results = predict_engineered_trajectory(
        bundle,
        frame,
        reliability=0.25,
        config=config,
        static_on_secondary_failure=True,
    )
    predictions = []
    for result in results:
        controller = result["controller"]
        predictions.append(
            HourPrediction(
                icu_hour=result["icu_hour"],
                **{
                    k: v
                    for k, v in result["prediction"].items()
                    if k != "probability_type"
                },
                temporal_weight=controller["temporal_weight"],
                static_weight=controller["static_weight"],
                status=controller.get("status", "bounded_anticipatory"),
                reason=controller.get("reason", "no_verified_feedback"),
            )
        )
    return TrajectoryResponse(
        episode_id=request.episode_id,
        provenance=request.provenance,
        model=request.model,
        predictions=predictions,
        decision_summaries=decision_summaries(
            bundle, request, predictions, [result["controller"] for result in results]
        ),
        configured_parameters={
            "reliability_exposure": 0.25,
            "fallback_weight": 0.0,
            "max_deviation": 0.25,
            "max_step": 0.10,
            "temperature": temperature,
        },
    )
