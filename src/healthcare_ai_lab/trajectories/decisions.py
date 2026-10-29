"""Read-only interpretation of the existing raw-hourly numerical service."""

import math

import numpy as np
import pandas as pd
from scipy.special import expit, xlogy
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.models.artifact import feature_hash
from healthcare_ai_lab.preprocessing.sepsis import (
    RAW_REQUIRED,
    VITALS,
    build_sepsis_features,
)
from healthcare_ai_lab.trajectories.contracts import HourPrediction
from healthcare_ai_lab.trajectories.decision_contracts import (
    ControllerInterpretation,
    DecisionRecord,
    DecisionSummaryResponse,
    InformationQuality,
    LogOddsContribution,
    Measurement,
    PredictionChange,
    PredictionDiagnostics,
)
from healthcare_ai_lab.trajectories.raw import (
    RawTrajectoryRequest,
    predict_raw_trajectory,
)


def prediction_change(
    current: HourPrediction, previous: HourPrediction | None
) -> PredictionChange:
    """Exact additive v0.6 identity; no invented gap for unavailable experts."""
    if previous is None:
        return PredictionChange(status="first_observation")
    if current.icu_hour <= previous.icu_hour:
        raise ValueError("Expected a strictly earlier observed hour")
    total = current.final_probability - previous.final_probability
    static = current.static_probability - previous.static_probability
    values = {
        "elapsed_hours": current.icu_hour - previous.icu_hour,
        "total": total,
        "static_movement": static,
    }
    if current.temporal_probability is None or previous.temporal_probability is None:
        return PredictionChange(
            status="partial_temporal_unavailable",
            **values,
            unsupported_terms=["disagreement_movement", "controller_reallocation"],
        )
    gap = current.temporal_probability - current.static_probability
    old_gap = previous.temporal_probability - previous.static_probability
    disagreement = previous.temporal_weight * (gap - old_gap)
    reallocation = (current.temporal_weight - previous.temporal_weight) * gap
    residual = total - (static + disagreement + reallocation)
    if abs(residual) > 1e-12:
        raise ValueError("Prediction change identity mismatch")
    return PredictionChange(
        status="exact",
        **values,
        disagreement_movement=disagreement,
        controller_reallocation=reallocation,
        identity_residual=residual,
    )


def logistic_contributions(entry, row, probability) -> LogOddsContribution:
    """Explain only a supported fitted binary pipeline and its served probability.

    Terms are beta_j * scaled(imputed(x))_j, not probability changes or causal
    effects. Optional explanation failures never invalidate a healthy prediction.
    """

    def unavailable(reason):
        return LogOddsContribution(status="unavailable", reason=reason)

    if probability is None:
        return unavailable("prediction_unavailable")
    try:
        model, names = entry["model"], entry["features"]
        if not isinstance(model, Pipeline) or len(model.steps) != 3:
            return unavailable("unsupported_model_or_preprocessing")
        imputer, scaler, classifier = (step[1] for step in model.steps)
        if (
            not isinstance(imputer, SimpleImputer)
            or imputer.add_indicator
            or not isinstance(scaler, StandardScaler)
            or not isinstance(classifier, LogisticRegression)
            or list(classifier.classes_) != [0, 1]
            or list(model.feature_names_in_) != names
        ):
            return unavailable("unsupported_model_or_preprocessing")
        transformed = model[:-1].transform(row[names])
        coefficients = classifier.coef_
        if transformed.shape != (1, len(names)) or coefficients.shape != (
            1,
            len(names),
        ):
            return unavailable("feature_shape_mismatch")
        terms = transformed[0] * coefficients[0]
        intercept = float(classifier.intercept_[0])
        logit = intercept + float(terms.sum())
        reconstructed = float(expit(logit))
        if not np.isfinite(terms).all() or not math.isfinite(logit):
            return unavailable("nonfinite_explanation")
        if not math.isclose(reconstructed, probability, rel_tol=0, abs_tol=1e-12):
            return unavailable("served_probability_mismatch")
        return LogOddsContribution(
            status="available",
            intercept=intercept,
            terms=dict(zip(names, terms.tolist(), strict=True)),
            total_log_odds=logit,
            reconstructed_probability=reconstructed,
        )
    except Exception:  # noqa: BLE001 -- optional model introspection must not break inference
        return unavailable("fitted_preprocessing_unavailable")


def summarize_raw_trajectory(
    bundle: dict, request: RawTrajectoryRequest, *, include_contributions=False
) -> DecisionSummaryResponse:
    # Sole prediction authority; this adapter does not implement another model path.
    trajectory = predict_raw_trajectory(bundle, request)
    max_step = trajectory.configured_parameters["max_step"]
    deviation_upper = min(
        1.0,
        trajectory.configured_parameters["fallback_weight"]
        + trajectory.configured_parameters["max_deviation"],
    )
    raw = pd.DataFrame(
        [
            {**r.measurements, "ICULOS": r.icu_hour, "patient_id": request.episode_id}
            for r in request.observations
        ]
    )
    # Reuse the exact causal builder for explanations rather than reimplement ages.
    engineered = build_sepsis_features(raw)
    records = []
    previous = None
    for index, (observation, current) in enumerate(
        zip(request.observations, trajectory.predictions, strict=True)
    ):
        features = engineered.iloc[index]
        measurements = {}
        for name, value in observation.measurements.items():
            fields = {
                "current_value": value,
                "missing": value is None,
                "used_by_feature_builder": name in RAW_REQUIRED,
            }
            if name in VITALS:
                age, last = features[f"{name}_age"], features[f"{name}_last"]
                known = not pd.isna(age)
                measurements[name] = Measurement(
                    **fields,
                    age_hours=float(age) if known else None,
                    last_value=float(last) if not pd.isna(last) else None,
                    last_observed_hour=current.icu_hour - float(age) if known else None,
                    freshness_status="observed_hour_proxy"
                    if known
                    else "never_observed",
                )
            else:
                measurements[name] = Measurement(
                    **fields, freshness_status="not_tracked"
                )
        missing = [name for name in VITALS if observation.measurements[name] is None]
        p, pt = current.final_probability, current.temporal_probability
        gap = None if pt is None else pt - current.static_probability
        fallback = current.status == "static_only_fallback"
        step = (
            None
            if previous is None
            else abs(current.temporal_weight - previous.temporal_weight)
        )
        active = []
        if fallback:
            active.append("emergency_static_override")
        else:
            if current.temporal_weight == 0:
                active.append("allocation_lower_bound")
            if math.isclose(
                current.temporal_weight, deviation_upper, rel_tol=0, abs_tol=1e-12
            ):
                active.append("fallback_deviation_bound")
            if step is not None and math.isclose(
                step, max_step, rel_tol=0, abs_tol=1e-12
            ):
                active.append("step_bound")
        contributions = None
        if include_contributions:
            contributions = {
                role: logistic_contributions(
                    bundle[role], engineered.iloc[[index]], probability
                )
                for role, probability in (
                    ("static", current.static_probability),
                    ("temporal", pt),
                )
            }
        records.append(
            DecisionRecord(
                current=current,
                previous=previous,
                change=prediction_change(current, previous),
                measurements=measurements,
                information_quality=InformationQuality(
                    observed_vitals=7 - len(missing),
                    vital_completeness=(7 - len(missing)) / 7,
                    missing_vitals=missing,
                ),
                diagnostics=PredictionDiagnostics(
                    predictive_entropy_nats=float(-xlogy(p, p) - xlogy(1 - p, 1 - p)),
                    expert_gap=gap,
                    absolute_expert_gap=None if gap is None else abs(gap),
                ),
                controller=ControllerInterpretation(
                    mode=current.status,
                    reason=current.reason,
                    emergency_fallback=fallback,
                    step_limit_overridden=bool(
                        fallback and step is not None and step > max_step + 1e-12
                    ),
                    active_constraints=active,
                ),
                log_odds_contributions=contributions,
            )
        )
        previous = current
    manifest = bundle.get("artifact_manifest")
    return DecisionSummaryResponse(
        episode_id=request.episode_id,
        model=trajectory.model,
        provenance=trajectory.provenance,
        artifact_sha256=manifest["artifact_sha256"] if manifest else None,
        artifact_integrity="checksum_bound_manifest" if manifest else "declared_only",
        feature_transform_sha256=trajectory.feature_engineering.notebook_transform_sha256,
        model_feature_hashes={
            role: feature_hash(bundle[role]["features"])
            for role in ("static", "temporal")
        },
        configured_parameters=trajectory.configured_parameters,
        records=records,
    )
