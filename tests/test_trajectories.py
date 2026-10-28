"""Synthetic engineered histories only; no artifact or database access."""

from copy import deepcopy
from itertools import pairwise

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from healthcare_ai_lab.api.main import app, get_research_bundle
from healthcare_ai_lab.trajectories.contracts import TrajectoryRequest
from healthcare_ai_lab.trajectories.service import predict_trajectory


class ProbabilityModel:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, frame):
        return np.tile([1 - self.probability, self.probability], (len(frame), 1))


class GainModel:
    def predict(self, frame):
        # Large first proposal, small second proposal exercises step limits.
        return np.where(frame.ICULOS == 1, 1.0, -1.0)


@pytest.fixture
def payload():
    return {
        "dataset": "sepsis2019",
        "episode_id": "synthetic-a",
        "provenance": {
            "feature_schema": "synthetic-v1",
            "preprocessing_version": "1",
            "pipeline_id": "synthetic-test",
            "declaration": "current_and_past_only",
        },
        "model": {"artifact_id": "synthetic", "model_version": "1"},
        "observations": [
            {
                "episode_id": "synthetic-a",
                "icu_hour": hour,
                "features": {"HR_current": 90.0, "HR_last": 88.0},
            }
            for hour in (1, 2, 3)
        ],
    }


@pytest.fixture
def bundle(payload):
    return {
        "static": {"features": ["HR_current"], "model": ProbabilityModel(0.1)},
        "temporal": {"features": ["HR_last"], "model": ProbabilityModel(0.3)},
        "controller": GainModel(),
        "meta_features": ["ICULOS", "entropy_static"],
        "trajectory_metadata": {
            key: deepcopy(payload[key]) for key in ("dataset", "provenance", "model")
        },
    }


def score(bundle, payload):
    return predict_trajectory(bundle, TrajectoryRequest.model_validate(payload))


def test_valid_trajectory(bundle, payload):
    result = score(bundle, payload)
    assert result.research_only and not result.clinical_use
    assert "not_independently_verified" in result.provenance_status
    assert [row.icu_hour for row in result.predictions] == [1, 2, 3]
    weights = [row.temporal_weight for row in result.predictions]
    assert all(0 <= q <= 0.25 for q in weights)
    assert max(abs(a - b) for a, b in pairwise(weights)) <= 0.100000001
    for row in result.predictions:
        assert row.static_probability == 0.1
        assert row.temporal_probability == 0.3
        assert row.static_weight + row.temporal_weight == pytest.approx(1)
        assert row.final_probability == pytest.approx(
            (1 - row.temporal_weight) * 0.1 + row.temporal_weight * 0.3
        )


@pytest.mark.parametrize(
    "hours",
    [(2, 1, 3), (1, 1, 3), (1, float("nan"), 3), (1, float("inf"), 3), (0, 1, 2)],
)
def test_invalid_hours(payload, hours):
    for row, hour in zip(payload["observations"], hours):
        row["icu_hour"] = hour
    with pytest.raises(ValidationError):
        TrajectoryRequest.model_validate(payload)


def test_mixed_patients(payload):
    payload["observations"][1]["episode_id"] = "other"
    with pytest.raises(ValidationError, match="Mixed episode"):
        TrajectoryRequest.model_validate(payload)


def test_isolation_and_prefix(bundle, payload):
    first = score(bundle, payload)
    other = deepcopy(payload)
    other["episode_id"] = "synthetic-b"
    for row in other["observations"]:
        row["episode_id"] = "synthetic-b"
    assert score(bundle, other).predictions == first.predictions
    assert score(bundle, payload) == first
    prefix = deepcopy(payload)
    prefix["observations"] = prefix["observations"][:2]
    assert score(bundle, prefix).predictions == first.predictions[:2]
    payload["observations"][-1]["features"]["HR_current"] = 999
    assert score(bundle, payload).predictions[:2] == first.predictions[:2]


@pytest.mark.parametrize("secondary", ["temporal", "controller"])
def test_secondary_failure_immediate_static_fallback(bundle, payload, secondary):
    class FailingModel:
        def predict(self, frame):
            if frame.ICULOS.iloc[0] == 2:
                raise RuntimeError("synthetic failure")
            return np.ones(len(frame))

        def predict_proba(self, frame):
            if frame.HR_last.iloc[0] == 89:
                raise RuntimeError("synthetic failure")
            return np.tile([0.7, 0.3], (len(frame), 1))

    if secondary == "controller":
        bundle["controller"] = FailingModel()
    else:
        bundle["temporal"]["model"] = FailingModel()
    payload["observations"][1]["features"]["HR_last"] = 89
    result = score(bundle, payload).predictions
    assert result[0].temporal_weight == pytest.approx(0.25)
    assert result[1].temporal_weight == 0
    assert result[1].final_probability == result[1].static_probability == 0.1
    assert result[1].temporal_probability is None
    assert result[1].status == "static_only_fallback"
    assert result[1].reason == "secondary_inference_failed"
    assert result[2].temporal_weight <= 0.10


@pytest.mark.parametrize(
    "field", ["feature_schema", "preprocessing_version", "pipeline_id"]
)
def test_provenance_incompatibility(bundle, payload, field):
    payload["provenance"][field] = "unknown"
    with pytest.raises(ValueError, match="incompatibility"):
        score(bundle, payload)


def test_model_and_feature_incompatibility(bundle, payload):
    bad = deepcopy(payload)
    bad["model"]["model_version"] = "unknown"
    with pytest.raises(ValueError):
        score(bundle, bad)
    del payload["observations"][0]["features"]["HR_last"]
    with pytest.raises(ValueError, match="Missing engineered"):
        score(bundle, payload)


def test_nonfinite_secondary_output_falls_back(bundle, payload):
    bundle["temporal"]["model"] = ProbabilityModel(float("nan"))
    assert all(
        row.status == "static_only_fallback"
        for row in score(bundle, payload).predictions
    )


def test_static_failure_is_not_fabricated(bundle, payload):
    bundle["static"]["model"] = ProbabilityModel(float("nan"))
    with pytest.raises(ValueError, match="Invalid model probability"):
        score(bundle, payload)


@pytest.mark.parametrize("field", ["SepsisLabel", "patient_id", "ICULOS"])
def test_reserved_features_rejected(payload, field):
    payload["observations"][0]["features"][field] = 1
    with pytest.raises(ValidationError, match="Reserved"):
        TrajectoryRequest.model_validate(payload)


def test_legacy_bundle_requires_explicit_provenance(bundle, payload):
    del bundle["trajectory_metadata"]
    with pytest.raises(ValueError, match="incompatibility"):
        score(bundle, payload)


def test_routes_and_legacy_compatibility(bundle, payload):
    from unittest.mock import patch

    with patch("healthcare_ai_lab.api.main.get_research_bundle", return_value=bundle):
        client = TestClient(app)
        result = client.post("/api/v1/research/sepsis/trajectory", json=payload)
        assert result.status_code == 200
        assert len(result.json()["predictions"]) == 3
        legacy = client.post(
            "/api/v1/research/sepsis/predict-engineered",
            json={
                "features": {"HR_current": 90, "HR_last": 88, "ICULOS": 1},
            },
        )
        assert legacy.status_code == 200
        assert legacy.json()["controller"]["mode"] == "bounded_anticipatory_research"
        invalid = deepcopy(payload)
        invalid["observations"].reverse()
        assert (
            client.post("/api/v1/research/sepsis/trajectory", json=invalid).status_code
            == 422
        )
    get_research_bundle.cache_clear()
