"""Deterministic v0.6 identity, typed transport and time-isolation checks."""

import json
import math
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from healthcare_ai_lab.api.main import app
from healthcare_ai_lab.trajectories.contracts import (
    DecisionSummary,
    HourPrediction,
    TrajectoryRequest,
)
from healthcare_ai_lab.trajectories.service import predict_trajectory
from healthcare_ai_lab.trajectories.summary import probability_change


def prediction(s, t, q, hour=1):
    return HourPrediction(
        icu_hour=hour,
        static_probability=s,
        temporal_probability=t,
        final_probability=(1 - q) * s + q * t if t is not None else s,
        temporal_weight=q,
        static_weight=1 - q,
        status="bounded_anticipatory" if t is not None else "static_only_fallback",
        reason="no_verified_feedback"
        if t is not None
        else "secondary_inference_failed",
    )


def test_exact_ordered_identity_with_boundary_and_random_probabilities():
    rng = np.random.default_rng(741)
    examples = [*rng.random((250, 6)), [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]]
    for s0, t0, q0, s1, t1, q1 in examples:
        old, new = prediction(s0, t0, q0), prediction(s1, t1, q1, hour=4)
        change = probability_change(old, new)
        assert change.status == "available"
        assert change.static_delta == s1 - s0
        assert change.disagreement_change_at_previous_allocation == q0 * (
            (t1 - s1) - (t0 - s0)
        )
        assert change.reallocation_at_current_disagreement == (q1 - q0) * (t1 - s1)
        terms = [
            change.static_delta,
            change.disagreement_change_at_previous_allocation,
            change.reallocation_at_current_disagreement,
        ]
        assert math.fsum(terms) == pytest.approx(
            new.final_probability - old.final_probability, rel=0, abs=5e-16
        )
        assert abs(change.reconstruction_residual) <= 5e-16


@pytest.mark.parametrize("q", [0, 0.125, 0.5, 1])
def test_identical_experts_and_zero_change(q):
    old, new = prediction(0.3, 0.3, 1 - q), prediction(0.3, 0.3, q, 2)
    change = probability_change(old, new)
    assert change.reallocation_at_current_disagreement == 0
    assert change.probability_delta == pytest.approx(0, abs=1e-16)


@pytest.mark.parametrize(
    "old,new",
    [
        (prediction(0.2, 0.4, 0.25), prediction(0.3, None, 0, 2)),
        (prediction(0.2, None, 0), prediction(0.3, 0.5, 0.1, 2)),
        (prediction(0.2, None, 0), prediction(0.3, None, 0, 2)),
    ],
)
def test_missing_expert_never_fabricates_decomposition(old, new):
    change = probability_change(old, new)
    assert change.probability_delta == new.final_probability - old.final_probability
    assert change.static_delta == new.static_probability - old.static_probability
    assert change.status == "unavailable"
    assert change.reason == "secondary_probability_unavailable"
    assert change.disagreement_change_at_previous_allocation is None
    assert change.reallocation_at_current_disagreement is None


class SyntheticProbability:
    def predict_proba(self, frame):
        p = frame.iloc[:, 0].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


class SyntheticGain:
    def predict(self, frame):
        return 0.01 * (frame.ICULOS.to_numpy() - 2)


@pytest.fixture
def scenario():
    payload = {
        "dataset": "sepsis2019",
        "episode_id": "synthetic-summary",
        "model": {"artifact_id": "synthetic", "model_version": "1"},
        "provenance": {
            "feature_schema": "synthetic",
            "preprocessing_version": "1",
            "pipeline_id": "synthetic",
            "declaration": "current_and_past_only",
        },
        "observations": [
            {
                "episode_id": "synthetic-summary",
                "icu_hour": hour,
                "features": {"p_static": s, "p_temporal": t},
            }
            for hour, s, t in [(1, 0.1, 0.3), (4, 0.2, 0.5), (5, 0.3, 0.6)]
        ],
    }
    bundle = {
        "static": {"features": ["p_static"], "model": SyntheticProbability()},
        "temporal": {"features": ["p_temporal"], "model": SyntheticProbability()},
        "controller": SyntheticGain(),
        "meta_features": ["ICULOS"],
        "trajectory_metadata": {
            k: payload[k] for k in ("dataset", "model", "provenance")
        },
    }
    return bundle, payload


def score(bundle, payload):
    return predict_trajectory(bundle, TrajectoryRequest.model_validate(payload))


def test_summary_previous_current_gap_and_unavailable_capabilities(scenario):
    bundle, payload = scenario
    result = score(bundle, payload)
    first, second, _ = result.decision_summaries
    assert first.previous is None and first.elapsed_hours is None
    assert first.change.reason == "first_observation"
    assert second.previous == first.current
    assert second.elapsed_hours == 3
    assert second.change.interval == "since_previous_observed_hour"
    assert (
        second.change.probability_delta
        == second.current.final_probability - first.current.final_probability
    )
    for summary, row in zip(result.decision_summaries, result.predictions, strict=True):
        assert (
            summary.current.model_dump(
                exclude={"static_contribution", "temporal_contribution"}
            )
            == row.model_dump()
        )
        assert (
            summary.current.static_contribution + summary.current.temporal_contribution
            == row.final_probability
        )
        assert summary.provenance.artifact_sha256 is None
        assert summary.provenance.artifact_status == "manifest_unavailable"
        assert summary.data_quality.status == "unavailable"
        assert summary.data_quality.measurements == {}
        assert summary.evaluation.status == "unavailable"
        for name in ("next_hour_forecast", "learned_hybrid", "workflow_recommendation"):
            assert getattr(summary, name).value is None
        assert summary.controller.empirically_validated is False


def test_summary_prefix_and_patient_isolation(scenario):
    bundle, payload = scenario
    expected = score(bundle, payload).decision_summaries
    prefix = deepcopy(payload)
    prefix["observations"] = prefix["observations"][:2]
    assert score(bundle, prefix).decision_summaries == expected[:2]
    future = deepcopy(payload)
    future["observations"][-1]["features"]["p_static"] = 0.99
    assert score(bundle, future).decision_summaries[:2] == expected[:2]
    other = deepcopy(payload)
    other["episode_id"] = "synthetic-other"
    for row in other["observations"]:
        row["episode_id"] = other["episode_id"]
    other_summaries = score(bundle, other).decision_summaries
    assert other_summaries[0].previous is None
    assert [s.current for s in other_summaries] == [s.current for s in expected]
    assert score(bundle, payload).decision_summaries == expected


def test_fallback_and_recovery_summary(scenario):
    bundle, payload = scenario
    payload["observations"][1]["features"]["p_temporal"] = None
    result = score(bundle, payload)
    failed = result.decision_summaries[1]
    assert failed.current.final_probability == 0.2
    assert failed.current.temporal_weight == 0
    assert failed.controller.fallback_active
    assert failed.controller.fallback_reason == "secondary_inference_failed"
    assert failed.current.temporal_contribution is None
    assert failed.diagnostics.signed_model_disagreement is None
    assert failed.data_quality.engineered_missing_count == 1
    assert failed.change.status == "unavailable"
    recovery = result.decision_summaries[2]
    assert not recovery.controller.fallback_active
    assert recovery.current.temporal_weight <= 0.1
    assert recovery.previous == failed.current
    assert recovery.change.status == "unavailable"


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 0), (0.5, math.log(2))])
def test_entropy_boundaries_are_finite_diagnostics(scenario, value, expected):
    bundle, payload = scenario
    for row in payload["observations"]:
        row["features"] = {"p_static": value, "p_temporal": value}
    for summary in score(bundle, payload).decision_summaries:
        assert summary.diagnostics.final_predictive_entropy_nats == pytest.approx(
            expected
        )
        assert (
            summary.diagnostics.entropy_interpretation
            == "predictive_distribution_not_reliability"
        )


def test_api_typed_summary_and_no_label_evaluation(scenario):
    bundle, payload = scenario
    client = TestClient(app)
    with (
        patch("healthcare_ai_lab.api.main.get_research_bundle", return_value=bundle),
        patch(
            "healthcare_ai_lab.evaluation.registry.evaluate",
            side_effect=AssertionError("No live labels"),
        ),
    ):
        response = client.post("/api/v1/research/sepsis/trajectory", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == len(body["decision_summaries"]) == 3
        summary = DecisionSummary.model_validate(body["decision_summaries"][1])
        assert summary.schema_version == "decision-summary.v0.6"
        assert summary.research_only and not summary.clinical_use
        bad = deepcopy(payload)
        bad["observations"][0]["features"]["SepsisLabel"] = 1
        assert (
            client.post("/api/v1/research/sepsis/trajectory", json=bad).status_code
            == 422
        )
        payload["observations"][0]["features"]["p_static"] = None
        assert (
            client.post("/api/v1/research/sepsis/trajectory", json=payload).status_code
            == 422
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_summary_probabilities_are_rejected(scenario, value):
    bundle, payload = scenario
    summary = score(bundle, payload).decision_summaries[0].model_dump()
    summary["current"]["final_probability"] = value
    with pytest.raises(ValidationError):
        DecisionSummary.model_validate(summary)


def test_documented_frontend_example_matches_service(scenario):
    bundle, payload = scenario
    example = Path(__file__).resolve().parents[1] / "docs/examples/decision-summary.v0.6.synthetic.json"
    expected = score(bundle, payload).decision_summaries[1].model_dump(mode="json")
    assert json.loads(example.read_text()) == expected
