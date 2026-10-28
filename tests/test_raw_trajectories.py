"""Synthetic raw-hourly API/adapter integration; no model artifacts are loaded."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from healthcare_ai_lab.api.main import app
from healthcare_ai_lab.preprocessing.sepsis import RAW_REQUIRED
from healthcare_ai_lab.trajectories.raw import (
    RawTrajectoryRequest,
    predict_raw_trajectory,
    runtime_provenance,
)


class SyntheticProbability:
    def __init__(self, feature):
        self.feature = feature

    def predict_proba(self, frame):
        value = frame[self.feature].to_numpy(dtype=float)
        p = np.where(np.isnan(value), 0.1, 0.2)
        return np.column_stack([1 - p, p])


class SyntheticGain:
    def predict(self, frame):
        return np.zeros(len(frame))


@pytest.fixture
def raw_payload():
    measurements = dict.fromkeys(RAW_REQUIRED, None)
    measurements.update(Age=55, Gender=1, HospAdmTime=-12, Unit1=1)
    return {
        "dataset": "sepsis2019",
        "episode_id": "synthetic-raw",
        "model": {"artifact_id": "synthetic", "model_version": "1"},
        "observations": [
            {
                "episode_id": "synthetic-raw",
                "icu_hour": hour,
                "measurements": {**measurements, "HR": hr},
            }
            for hour, hr in [(1, None), (2, 90), (4, None)]
        ],
    }


@pytest.fixture
def raw_bundle(raw_payload):
    return {
        "static": {
            "features": ["HR_current"],
            "model": SyntheticProbability("HR_current"),
        },
        "temporal": {
            "features": ["HR_last", "HR_age", "HR_std6"],
            "model": SyntheticProbability("HR_last"),
        },
        "controller": SyntheticGain(),
        "meta_features": ["entropy_static", "ICULOS"],
        "trajectory_metadata": {
            "dataset": "sepsis2019",
            "model": raw_payload["model"],
            "provenance": runtime_provenance().model_dump(),
        },
    }


def score(bundle, payload):
    return predict_raw_trajectory(bundle, RawTrajectoryRequest.model_validate(payload))


def test_raw_features_predictions_and_prefix(raw_bundle, raw_payload):
    result = score(raw_bundle, raw_payload)
    assert result.provenance == runtime_provenance()
    assert (
        result.provenance_status == "runtime_generated_source_availability_unverified"
    )
    assert result.feature_engineering.source_columns["HR_age"] == ["HR", "ICULOS"]
    assert result.research_only and not result.clinical_use
    assert result.predictions[0].static_probability == 0.1
    assert result.predictions[0].temporal_probability == 0.1
    assert result.predictions[2].static_probability == 0.1
    assert result.predictions[2].temporal_probability == 0.2
    prefix = deepcopy(raw_payload)
    prefix["observations"] = prefix["observations"][:2]
    assert score(raw_bundle, prefix).predictions == result.predictions[:2]
    assert score(raw_bundle, raw_payload) == result
    other = deepcopy(raw_payload)
    other["episode_id"] = "synthetic-other"
    for row in other["observations"]:
        row["episode_id"] = "synthetic-other"
    assert score(raw_bundle, other).predictions == result.predictions


def test_model_schema_and_provenance_fail_closed(raw_bundle, raw_payload):
    raw_bundle["temporal"]["features"].append("unsupported_slope")
    with pytest.raises(ValueError, match="outside Notebook"):
        score(raw_bundle, raw_payload)
    raw_bundle["temporal"]["features"].remove("unsupported_slope")
    raw_bundle["trajectory_metadata"]["provenance"]["preprocessing_version"] = "unknown"
    with pytest.raises(ValueError, match="incompatibility"):
        score(raw_bundle, raw_payload)


@pytest.mark.parametrize("hour", [0, -1, 1.5, float("nan"), float("inf"), 1, True])
def test_bad_raw_hours(raw_payload, hour):
    raw_payload["observations"][1]["icu_hour"] = hour
    with pytest.raises(ValidationError):
        RawTrajectoryRequest.model_validate(raw_payload)


def test_order_identity_and_missing_sources(raw_payload):
    bad = deepcopy(raw_payload)
    bad["observations"].reverse()
    with pytest.raises(ValidationError):
        RawTrajectoryRequest.model_validate(bad)
    bad = deepcopy(raw_payload)
    bad["observations"][1]["episode_id"] = "other"
    with pytest.raises(ValidationError):
        RawTrajectoryRequest.model_validate(bad)
    del raw_payload["observations"][1]["measurements"]["HR"]
    with pytest.raises(ValidationError, match="Missing raw source"):
        RawTrajectoryRequest.model_validate(raw_payload)


@pytest.mark.parametrize("column", ["SepsisLabel", "HR_last", "patient_id", "split"])
def test_no_engineered_or_outcome_inputs(raw_payload, column):
    raw_payload["observations"][0]["measurements"][column] = 1
    with pytest.raises(ValidationError):
        RawTrajectoryRequest.model_validate(raw_payload)


def test_raw_route_and_artifact_gate(raw_bundle, raw_payload):
    client = TestClient(app)
    with patch(
        "healthcare_ai_lab.api.main.get_research_bundle", return_value=raw_bundle
    ):
        response = client.post(
            "/api/v1/research/sepsis/trajectory/raw-hourly", json=raw_payload
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 3
        del raw_bundle["trajectory_metadata"]
        response = client.post(
            "/api/v1/research/sepsis/trajectory/raw-hourly", json=raw_payload
        )
        assert response.status_code == 422
