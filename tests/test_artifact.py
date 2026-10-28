"""Synthetic fitted pipelines and temporary artifacts; no real patient/model access."""

from copy import deepcopy
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.special import xlogy
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.api.main import app, get_research_bundle
from healthcare_ai_lab.experiments.verify_sepsis_bundle import (
    VerificationError,
    training_consistency,
)
from healthcare_ai_lab.models.artifact import (
    META_FEATURES,
    PREPROCESSING,
    STATIC_FEATURES,
    TEMPORAL_FEATURES,
    ArtifactManifest,
    ArtifactValidationError,
    attach_manifest,
    expected_provenance,
    feature_hash,
    inspect_bundle,
    manifest_path,
    sha256_file,
    software_versions,
)
from healthcare_ai_lab.models.sepsis_inference import (
    load_bundle,
    predict_engineered_trajectory,
    static_entropy,
)
from healthcare_ai_lab.preprocessing.sepsis import RAW_REQUIRED
from healthcare_ai_lab.trajectories.contracts import TrajectoryRequest
from healthcare_ai_lab.trajectories.service import predict_trajectory


@pytest.fixture
def bundle():
    rng = np.random.default_rng(12345)
    features = list(dict.fromkeys([*TEMPORAL_FEATURES, *META_FEATURES]))
    frame = pd.DataFrame(rng.normal(size=(30, len(features))), columns=features)
    frame.loc[0, "HR_current"] = np.nan
    labels = np.tile([0, 1], 15)

    def fit(names, estimator, last):
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (last, estimator),
            ]
        ).fit(frame[names], labels)

    result = {
        "temperature": 0.01,
        "adaptive_exposure": 0.25,
        "fallback_q": 0.0,
        "status": "research_prototype",
    }
    for role, names in (("static", STATIC_FEATURES), ("temporal", TEMPORAL_FEATURES)):
        result[role] = {
            "features": list(names),
            "model": fit(
                names,
                LogisticRegression(C=0.01, max_iter=1000, random_state=12345),
                "model",
            ),
        }
    result["controller"] = fit(META_FEATURES, Ridge(alpha=10.0), "ridge")
    result["meta_features"] = list(META_FEATURES)
    return result


def manifest_for(bundle, digest="0" * 64):
    features = inspect_bundle(bundle)
    return ArtifactManifest(
        artifact_sha256=digest,
        model={
            "artifact_id": "notebook09-prototype",
            "model_version": f"sha256:{digest}",
        },
        provenance=expected_provenance(),
        source_hashes={},
        features=features,
        feature_hashes={k: feature_hash(v) for k, v in features.items()},
        preprocessing=PREPROCESSING,
        verified_runtime_versions=software_versions(),
        training_evidence={"passed": True},
        smoke_test={"passed": True},
    )


def test_sidecar_upgrade_and_binary_unchanged(bundle, tmp_path):
    path = tmp_path / "synthetic.joblib"
    joblib.dump(bundle, path)
    before = sha256_file(path)
    assert "trajectory_metadata" not in load_bundle(path)  # legacy supported
    manifest = manifest_for(bundle, before)
    manifest_path(path).write_text(manifest.model_dump_json())
    loaded = load_bundle(path)
    assert sha256_file(path) == before
    assert loaded["trajectory_metadata"] == manifest.trajectory_metadata()
    assert (
        loaded["artifact_manifest"]["historical_training_versions"]
        == "not_recorded_in_legacy_export"
    )


def test_checksum_rejected_before_deserialization(bundle, tmp_path):
    path = tmp_path / "synthetic.joblib"
    path.write_bytes(b"not an artifact")
    manifest_path(path).write_text(manifest_for(bundle).model_dump_json())
    with (
        patch("joblib.load") as deserialize,
        pytest.raises(ArtifactValidationError, match="checksum"),
    ):
        load_bundle(path)
    deserialize.assert_not_called()


@pytest.mark.parametrize(
    "change", ["order", "hash", "preprocessing", "version", "provenance", "evidence"]
)
def test_invalid_manifest_fails_closed(bundle, change):
    manifest = manifest_for(bundle)
    if change == "order":
        manifest.features["static"].reverse()
    elif change == "hash":
        manifest.feature_hashes["static"] = "bad"
    elif change == "preprocessing":
        manifest.preprocessing = {**manifest.preprocessing, "imputation": "future_fill"}
    elif change == "version":
        manifest.verified_runtime_versions = {
            **manifest.verified_runtime_versions,
            "scikit-learn": "unknown",
        }
    elif change == "evidence":
        manifest.training_evidence = {"passed": False}
    else:
        manifest.provenance.preprocessing_version = "unknown"
    with pytest.raises(ArtifactValidationError, match="compatibility"):
        attach_manifest(bundle, manifest)


def test_fitted_order_and_pipeline_semantics(bundle):
    altered = deepcopy(bundle)
    altered["static"]["features"][0:2] = altered["static"]["features"][1::-1]
    with pytest.raises(ArtifactValidationError, match="schema/order"):
        inspect_bundle(altered)
    bundle["temporal"]["model"].named_steps["imputer"].strategy = "mean"
    with pytest.raises(ArtifactValidationError, match="preprocessing"):
        inspect_bundle(bundle)


def test_missing_values_and_input_dictionary_order(bundle):
    row = dict.fromkeys(TEMPORAL_FEATURES, np.nan)
    row.update(
        ICULOS=1.0,
        HR_current=90,
        HR_age=0,
        Resp_age=1,
        HR_observed=1,
        Resp_observed=0,
        Temp_observed=0,
    )
    left = pd.DataFrame([row])
    right = left[list(reversed(left.columns))]
    a, b = (
        predict_engineered_trajectory(bundle, left),
        predict_engineered_trajectory(bundle, right),
    )
    assert a == b
    assert np.isfinite(a[0]["prediction"]["final_probability"])


@pytest.mark.parametrize(
    "role,method", [("temporal", "predict_proba"), ("controller", "predict")]
)
def test_manifested_model_failure_retains_static(bundle, role, method):
    manifest = manifest_for(bundle)
    active = attach_manifest(bundle, manifest)
    row = dict.fromkeys(TEMPORAL_FEATURES, None)
    row.update(
        HR_observed=0, Resp_observed=0, Temp_observed=0, HR_age=None, Resp_age=None
    )
    row.pop("ICULOS")
    request = TrajectoryRequest(
        dataset="sepsis2019",
        episode_id="synthetic",
        model=manifest.model,
        provenance=manifest.provenance,
        observations=[{"episode_id": "synthetic", "icu_hour": 1, "features": row}],
    )
    model = bundle[role] if role == "controller" else bundle[role]["model"]
    with patch.object(model, method, side_effect=RuntimeError("synthetic failure")):
        result = predict_trajectory(active, request).predictions[0]
    assert result.temporal_probability is None
    assert result.temporal_weight == 0
    assert result.final_probability == result.static_probability
    assert result.status == "static_only_fallback"


def test_entropy_matches_notebook_at_extreme_probabilities():
    p = np.array([0, 1e-14, 0.1, 0.5, 1 - 1e-14, 1.0])
    clipped = np.clip(p, 1e-10, 1 - 1e-10)
    expected = -(xlogy(p, clipped) + xlogy(1 - p, 1 - clipped))
    np.testing.assert_array_equal(static_entropy(p), expected)


def test_bad_sidecar_returns_service_unavailable(bundle, tmp_path, monkeypatch):
    path = tmp_path / "synthetic.joblib"
    joblib.dump(bundle, path)
    manifest_path(path).write_text('{"dataset": "wrong"}')
    monkeypatch.setenv("SEPSIS_MODEL_BUNDLE", str(path))
    get_research_bundle.cache_clear()
    try:
        response = TestClient(app).post(
            "/api/v1/research/sepsis/predict-engineered", json={"features": {}}
        )
        assert response.status_code == 503
        assert "wrong" not in response.text
    finally:
        get_research_bundle.cache_clear()


def test_activated_raw_api_uses_real_sidecar_loader(bundle, tmp_path, monkeypatch):
    path = tmp_path / "synthetic.joblib"
    joblib.dump(bundle, path)
    manifest = manifest_for(bundle, sha256_file(path))
    manifest_path(path).write_text(manifest.model_dump_json())
    monkeypatch.setenv("SEPSIS_MODEL_BUNDLE", str(path))
    get_research_bundle.cache_clear()
    try:
        response = TestClient(app).post(
            "/api/v1/research/sepsis/trajectory/raw-hourly",
            json={
                "dataset": "sepsis2019",
                "episode_id": "synthetic",
                "model": manifest.model.model_dump(),
                "observations": [
                    {
                        "episode_id": "synthetic",
                        "icu_hour": 1,
                        "measurements": dict.fromkeys(RAW_REQUIRED, None),
                    }
                ],
            },
        )
        assert response.status_code == 200
        assert response.json()["model"] == manifest.model.model_dump()
        summary = response.json()["decision_summaries"][0]
        assert summary["provenance"]["artifact_sha256"] == sha256_file(path)
        assert summary["provenance"]["artifact_status"] == "manifest_validated"
        assert summary["provenance"]["feature_status"] == (
            "runtime_generated_source_availability_unverified"
        )
        assert response.json()["predictions"][0]["status"] == "bounded_anticipatory"
    finally:
        get_research_bundle.cache_clear()


def test_verifier_rejects_split_overlap_before_fitting(bundle, tmp_path):
    # This is synthetic split metadata, not a real patient fixture.
    frame = pd.DataFrame(
        {
            "patient_id": [
                "synthetic-a",
                "synthetic-a",
                "synthetic-b",
                "synthetic-c",
                "synthetic-d",
            ],
            "split": ["train", "validation", "train", "test", "validation"],
        }
    )
    temporal, splits = tmp_path / "temporal.parquet", tmp_path / "splits.csv"
    frame.to_parquet(temporal)
    frame.drop_duplicates("patient_id").set_index("patient_id")[["split"]].to_csv(
        splits
    )
    with pytest.raises(VerificationError, match="split overlap"):
        training_consistency(bundle, temporal, tmp_path / "unused.parquet", splits)
