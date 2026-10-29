"""Synthetic causal dashboard interpretation; no local model or patient data."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.api.main import app
from healthcare_ai_lab.experiments.export_decisions import synthetic_example
from healthcare_ai_lab.preprocessing.sepsis import RAW_REQUIRED
from healthcare_ai_lab.trajectories.contracts import HourPrediction
from healthcare_ai_lab.trajectories.decision_contracts import DecisionSummaryResponse
from healthcare_ai_lab.trajectories.decisions import (
    logistic_contributions,
    prediction_change,
    summarize_raw_trajectory,
)
from healthcare_ai_lab.trajectories.raw import (
    RawTrajectoryRequest,
    predict_raw_trajectory,
    runtime_provenance,
)

ROUTE = "/api/v1/research/sepsis/trajectory/decision-summary"


class Gain:
    def predict(self, frame):
        return np.ones(len(frame))


@pytest.fixture
def fixture():
    payload = synthetic_example()["request"]
    payload["observations"][0]["measurements"]["HR"] = None
    bundle = {
        "controller": Gain(),
        "meta_features": ["ICULOS", "entropy_static"],
        "trajectory_metadata": {
            "dataset": "sepsis2019",
            "model": deepcopy(payload["model"]),
            "provenance": runtime_provenance().model_dump(),
        },
    }
    for role, names in (
        ("static", ["Age", "HR_current"]),
        ("temporal", ["ICULOS", "HR_last", "HR_age"]),
    ):
        X = pd.DataFrame(
            {
                name: np.arange(8, dtype=float) + index
                for index, name in enumerate(names)
            }
        )
        X.iloc[0, -1] = np.nan
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.01, random_state=12345)),
            ]
        )
        model.fit(X, [0, 0, 0, 1, 0, 1, 1, 1])
        bundle[role] = {"features": names, "model": model}
    return bundle, payload


def score(fixture, **kwargs):
    bundle, payload = fixture
    return summarize_raw_trajectory(
        bundle, RawTrajectoryRequest.model_validate(payload), **kwargs
    )


def point(hour, static, temporal, q):
    return HourPrediction(
        icu_hour=hour,
        static_probability=static,
        temporal_probability=temporal,
        final_probability=static
        if temporal is None
        else (1 - q) * static + q * temporal,
        temporal_weight=q,
        static_weight=1 - q,
        status="static_only_fallback" if temporal is None else "bounded_anticipatory",
        reason="secondary_inference_failed"
        if temporal is None
        else "no_verified_feedback",
    )


def test_exact_decomposition_randomized_and_endpoints():
    rng = np.random.default_rng(12345)
    for a, b, q0, c, d, q1 in [
        *rng.random((500, 6)),
        [0, 0, 0, 1, 1, 1],
        [0.2, 0.8, 0, 0.3, 0.6, 1],
    ]:
        before, current = point(1, a, b, q0), point(5, c, d, q1)
        result = prediction_change(current, before)
        assert result.status == "exact" and result.elapsed_hours == 4
        assert result.total == pytest.approx(
            result.static_movement
            + result.disagreement_movement
            + result.controller_reallocation,
            abs=1e-14,
        )
        assert abs(result.identity_residual) < 1e-14
    assert prediction_change(point(1, 0.2, 0.6, 0.2), None).total is None
    with pytest.raises(ValueError):
        prediction_change(before, current)


@pytest.mark.parametrize(
    "old_missing,new_missing", [(True, False), (False, True), (True, True)]
)
def test_missing_temporal_terms_unsupported(old_missing, new_missing):
    previous = point(1, 0.2, None if old_missing else 0.6, 0 if old_missing else 0.2)
    current = point(4, 0.3, None if new_missing else 0.7, 0 if new_missing else 0.2)
    change = prediction_change(current, previous)
    assert change.total == current.final_probability - previous.final_probability
    assert change.static_movement == pytest.approx(0.1)
    assert change.status == "partial_temporal_unavailable"
    assert (
        change.disagreement_movement
        is change.controller_reallocation
        is change.identity_residual
        is None
    )
    assert set(change.unsupported_terms) == {
        "disagreement_movement",
        "controller_reallocation",
    }


def test_service_prediction_parity_freshness_and_unavailable_capabilities(fixture):
    bundle, payload = fixture
    reference = predict_raw_trajectory(
        bundle, RawTrajectoryRequest.model_validate(payload)
    )
    result = score(fixture)
    assert [r.current for r in result.records] == reference.predictions
    first, middle, last = result.records
    assert first.previous is None and first.change.status == "first_observation"
    assert first.measurements["HR"].freshness_status == "never_observed"
    assert middle.previous == first.current and last.previous == middle.current
    assert last.change.elapsed_hours == 2
    assert last.measurements["HR"].current_value is None
    assert last.measurements["HR"].last_value == 90
    assert last.measurements["HR"].age_hours == 2
    assert last.measurements["HR"].last_observed_hour == 2
    assert last.measurements["Age"].freshness_status == "not_tracked"
    assert last.next_step == first.next_step
    assert (
        result.artifact_sha256 is None and result.artifact_integrity == "declared_only"
    )
    assert result.capabilities.next_hour_forecast.probability is None
    assert result.capabilities.next_hour_forecast.horizon_hours == 1
    assert result.capabilities.learned_hybrid.status == "unavailable"
    assert result.capabilities.clinician_review.status == "unavailable"
    assert result.research_only and not result.clinical_use
    assert (
        DecisionSummaryResponse.model_validate_json(result.model_dump_json()) == result
    )


def test_prefix_patient_and_request_isolation(fixture):
    bundle, payload = fixture
    result = score(fixture, include_contributions=True)
    for n in (1, 2):
        prefix = deepcopy(payload)
        prefix["observations"] = prefix["observations"][:n]
        assert (
            score((bundle, prefix), include_contributions=True).records
            == result.records[:n]
        )
    other = deepcopy(payload)
    other["episode_id"] = "synthetic-other"
    for row in other["observations"]:
        row["episode_id"] = "synthetic-other"
    assert score((bundle, other), include_contributions=True).records == result.records
    assert score(fixture, include_contributions=True) == result


def test_optional_measurements_do_not_leak_back_into_prefix(fixture):
    bundle, payload = fixture
    before = score(fixture)
    payload["observations"][-1]["measurements"]["Lactate"] = 1.5
    after = score((bundle, payload))
    assert after.records[:2] == before.records[:2]
    measurement = after.records[-1].measurements["Lactate"]
    assert measurement.current_value == 1.5
    assert not measurement.used_by_feature_builder
    assert (
        measurement.age_hours is None and measurement.freshness_status == "not_tracked"
    )


def test_active_constraints_describe_actual_bounded_path(fixture):
    class ChangingGain:
        def predict(self, frame):
            return np.where(frame.ICULOS.to_numpy() == 1, 0.0, 1.0)

    fixture[0]["controller"] = ChangingGain()
    result = score(fixture)
    assert result.records[0].current.temporal_weight == 0.125
    assert result.records[1].current.temporal_weight == 0.225
    assert "step_bound" in result.records[1].controller.active_constraints
    assert "fallback_deviation_bound" in result.records[2].controller.active_constraints


def test_actual_fitted_log_odds_and_imputation(fixture):
    result = score(fixture, include_contributions=True)
    for record in result.records:
        for role, attribution in record.log_odds_contributions.items():
            assert attribution.status == "available"
            assert set(attribution.terms) == set(fixture[0][role]["features"])
            assert attribution.total_log_odds == pytest.approx(
                attribution.intercept + sum(attribution.terms.values()), abs=1e-13
            )
            assert expit(attribution.total_log_odds) == pytest.approx(
                getattr(record.current, f"{role}_probability"), abs=1e-12
            )
    assert result.records[0].measurements["HR"].current_value is None


def test_emergency_fallback_overrides_step_and_retains_partial_explanation(fixture):
    bundle, _ = fixture
    original = bundle["temporal"]["model"].predict_proba

    def predict(frame):
        if frame.ICULOS.iloc[0] == 4:
            raise RuntimeError("private diagnostic")
        return original(frame)

    with patch.object(
        bundle["temporal"]["model"], "predict_proba", side_effect=predict
    ):
        result = score(fixture, include_contributions=True)
    last = result.records[-1]
    assert last.previous.temporal_weight == 0.25
    assert last.current.temporal_weight == 0
    assert last.current.final_probability == last.current.static_probability
    assert last.current.temporal_probability is None
    assert last.change.status == "partial_temporal_unavailable"
    assert last.controller.emergency_fallback and last.controller.step_limit_overridden
    assert last.log_odds_contributions["temporal"].reason == "prediction_unavailable"
    assert last.log_odds_contributions["static"].status == "available"
    assert "private diagnostic" not in result.model_dump_json()


def test_optional_explanation_failure_preserves_predictions(fixture):
    result = score(fixture)
    unavailable = logistic_contributions({}, None, 0.3)
    with patch(
        "healthcare_ai_lab.trajectories.decisions.logistic_contributions",
        return_value=unavailable,
    ):
        explained = score(fixture, include_contributions=True)
    assert [r.current for r in explained.records] == [r.current for r in result.records]
    assert all(
        r.log_odds_contributions["static"].status == "unavailable"
        for r in explained.records
    )


@pytest.mark.parametrize("probability", [0, 1])
def test_entropy_endpoints_and_all_missing_sources(fixture, probability):
    class Endpoint:
        def predict_proba(self, frame):
            return np.tile([1 - probability, probability], (len(frame), 1))

    bundle, payload = fixture
    bundle["static"]["model"] = bundle["temporal"]["model"] = Endpoint()
    for row in payload["observations"]:
        row["measurements"] = dict.fromkeys(RAW_REQUIRED, None)
    result = score(fixture)
    for r in result.records:
        assert r.diagnostics.predictive_entropy_nats == 0
        assert r.diagnostics.expert_gap == 0
        assert r.information_quality.vital_completeness == 0
        assert all(
            r.measurements[name].age_hours is None
            for name in r.information_quality.missing_vitals
        )


def test_new_api_openapi_and_old_raw_response_regression(fixture):
    bundle, payload = fixture
    client = TestClient(app)
    with patch("healthcare_ai_lab.api.main.get_research_bundle", return_value=bundle):
        old = client.post("/api/v1/research/sepsis/trajectory/raw-hourly", json=payload)
        new = client.post(ROUTE + "?include_contributions=true", json=payload)
    assert old.status_code == new.status_code == 200
    assert (
        old.json()
        == predict_raw_trajectory(
            bundle, RawTrajectoryRequest.model_validate(payload)
        ).model_dump()
    )
    assert [r["current"] for r in new.json()["records"]] == old.json()["predictions"]
    schema = client.get("/openapi.json").json()
    assert (
        "DecisionSummaryResponse" in schema["components"]["schemas"]
        and ROUTE in schema["paths"]
    )


@pytest.mark.parametrize(
    "mutation", ["duplicate", "mixed", "outcome", "nan", "out_of_order"]
)
def test_new_endpoint_rejects_bad_history(fixture, mutation):
    bundle, payload = fixture
    if mutation == "duplicate":
        payload["observations"][1]["icu_hour"] = 1
    elif mutation == "mixed":
        payload["observations"][1]["episode_id"] = "other"
    elif mutation == "outcome":
        payload["observations"][1]["measurements"]["SepsisLabel"] = 1
    elif mutation == "nan":
        payload["observations"][1]["measurements"]["HR"] = "NaN"
    else:
        payload["observations"].reverse()
    with patch("healthcare_ai_lab.api.main.get_research_bundle", return_value=bundle):
        assert TestClient(app).post(ROUTE, json=payload).status_code == 422


def test_api_artifact_and_static_failures_are_safe(fixture):
    bundle, payload = fixture
    client = TestClient(app)
    with patch(
        "healthcare_ai_lab.api.main.get_research_bundle",
        side_effect=FileNotFoundError("private-path"),
    ):
        response = client.post(ROUTE, json=payload)
        assert response.status_code == 503 and "private-path" not in response.text
    with (
        patch("healthcare_ai_lab.api.main.get_research_bundle", return_value=bundle),
        patch.object(
            bundle["static"]["model"],
            "predict_proba",
            side_effect=RuntimeError("private-data"),
        ),
    ):
        response = client.post(ROUTE, json=payload)
        assert response.status_code == 503 and "private-data" not in response.text
    bundle["trajectory_metadata"]["model"]["model_version"] = "incompatible"
    with patch("healthcare_ai_lab.api.main.get_research_bundle", return_value=bundle):
        assert client.post(ROUTE, json=payload).status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities_rejected(value):
    with pytest.raises(ValidationError):
        point(1, value, 0.2, 0.1)
