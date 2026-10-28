"""Checksum-bound, research-only metadata for trusted Notebook 09 artifacts.

A manifest records observed evidence; it is neither a signature nor independent
proof of historical training. Never load an untrusted joblib file.
"""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.preprocessing.sepsis import (
    FEATURE_SCHEMA,
    NOTEBOOK_TRANSFORM_SHA256,
    PIPELINE_ID,
    PREPROCESSING_VERSION,
    VITALS,
)
from healthcare_ai_lab.trajectories.contracts import (
    Contract,
    FeatureProvenance,
    ModelMetadata,
)

CONTEXT = ["Age", "Gender", "HospAdmTime", "ICULOS", "Unit1", "Unit2"]
STATIC_FEATURES = CONTEXT + [f"{v}_current" for v in VITALS]
TEMPORAL_FEATURES = STATIC_FEATURES + [
    f"{v}_{suffix}" for v in VITALS for suffix in ("last", "mean6", "std6", "delta3")
]
META_FEATURES = [
    "ICULOS",
    "HR_observed",
    "Resp_observed",
    "Temp_observed",
    "HR_age",
    "Resp_age",
    "entropy_static",
]
PREPROCESSING = {
    "imputation": "training_median",
    "missing_values": "NaN",
    "add_indicator": False,
    "keep_empty_features": False,
    "scaling": "training_mean_and_population_variance",
    "entropy": "Notebook09.binary_entropy",
    "entropy_probability_clip": 1e-10,
}


class ArtifactValidationError(ValueError):
    """Artifact integrity, order, semantics, or declared version is incompatible."""


class ArtifactManifest(Contract):
    manifest_version: Literal["sepsis-notebook09.v1"] = "sepsis-notebook09.v1"
    dataset: Literal["sepsis2019"] = "sepsis2019"
    prediction_target: Literal["shifted_challenge_label"] = "shifted_challenge_label"
    prototype_status: Literal["research_prototype_not_clinically_validated"] = (
        "research_prototype_not_clinically_validated"
    )
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: ModelMetadata
    provenance: FeatureProvenance
    notebook02_transform_sha256: str = NOTEBOOK_TRANSFORM_SHA256
    source_hashes: dict[str, str]
    features: dict[str, list[str]]
    feature_hashes: dict[str, str]
    preprocessing: dict
    verified_runtime_versions: dict[str, str]
    historical_training_versions: Literal["not_recorded_in_legacy_export"] = (
        "not_recorded_in_legacy_export"
    )
    training_provenance_status: Literal[
        "reconstructed_consistency_not_training_attestation"
    ] = "reconstructed_consistency_not_training_attestation"
    training_evidence: dict
    smoke_test: dict

    def trajectory_metadata(self) -> dict:
        return {
            "dataset": self.dataset,
            "model": self.model.model_dump(),
            "provenance": self.provenance.model_dump(),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_hash(features: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(features, separators=(",", ":")).encode()
    ).hexdigest()


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
        },
    }


def expected_provenance() -> FeatureProvenance:
    return FeatureProvenance(
        feature_schema=FEATURE_SCHEMA,
        preprocessing_version=PREPROCESSING_VERSION,
        pipeline_id=PIPELINE_ID,
        declaration="current_and_past_only",
    )


def inspect_bundle(bundle: dict) -> dict[str, list[str]]:
    """Verify fitted feature order and Notebook 09 pipeline semantics, without fitting."""
    expected = {
        "static": STATIC_FEATURES,
        "temporal": TEMPORAL_FEATURES,
        "controller": META_FEATURES,
    }
    actual = {}
    for role, canonical in expected.items():
        model = bundle[role] if role == "controller" else bundle[role]["model"]
        names = (
            bundle["meta_features"]
            if role == "controller"
            else bundle[role]["features"]
        )
        # Notebook 09 drops training-all-missing base features, retaining order.
        if (
            not names
            or len(names) != len(set(names))
            or names != [name for name in canonical if name in names]
            or (role == "controller" and names != canonical)
        ):
            raise ArtifactValidationError("Unexpected Notebook 09 feature schema/order")
        if not isinstance(model, Pipeline) or list(model.feature_names_in_) != names:
            raise ArtifactValidationError(
                "Declared feature order differs from fitted model"
            )
        final = "ridge" if role == "controller" else "model"
        if list(model.named_steps) != ["imputer", "scaler", final]:
            raise ArtifactValidationError("Unexpected pipeline structure")
        imputer, scaler, estimator = (
            model.named_steps[name] for name in ("imputer", "scaler", final)
        )
        if (
            not isinstance(imputer, SimpleImputer)
            or imputer.strategy != "median"
            or not np.isnan(imputer.missing_values)
            or imputer.add_indicator
            or imputer.keep_empty_features
            or not np.isfinite(imputer.statistics_).all()
            or not isinstance(scaler, StandardScaler)
            or not scaler.with_mean
            or not scaler.with_std
            or not np.isfinite(scaler.mean_).all()
            or not np.isfinite(scaler.var_).all()
        ):
            raise ArtifactValidationError("Unexpected fitted preprocessing semantics")
        if role == "controller":
            if not isinstance(estimator, Ridge) or estimator.alpha != 10.0:
                raise ArtifactValidationError("Unexpected controller model")
        elif (
            not isinstance(estimator, LogisticRegression)
            or estimator.C != 0.01
            or estimator.max_iter != 1000
            or estimator.random_state != 12345
            or list(model.classes_) != [0, 1]
        ):
            raise ArtifactValidationError("Unexpected base classifier")
        if (
            not np.isfinite(estimator.coef_).all()
            or not np.isfinite(estimator.intercept_).all()
        ):
            raise ArtifactValidationError("Nonfinite fitted estimator")
        actual[role] = list(names)
    if (
        bundle.get("status") != "research_prototype"
        or bundle.get("temperature") != 0.01
        or bundle.get("adaptive_exposure") != 0.25
        or bundle.get("fallback_q") != 0.0
    ):
        raise ArtifactValidationError("Unexpected prototype configuration")
    return actual


def validate_manifest(bundle: dict, manifest: ArtifactManifest) -> None:
    features = inspect_bundle(bundle)
    if (
        manifest.provenance != expected_provenance()
        or manifest.notebook02_transform_sha256 != NOTEBOOK_TRANSFORM_SHA256
        or manifest.features != features
        or manifest.feature_hashes != {k: feature_hash(v) for k, v in features.items()}
        or manifest.preprocessing != PREPROCESSING
        or manifest.verified_runtime_versions != software_versions()
        or manifest.model.artifact_id != "notebook09-prototype"
        or manifest.model.model_version != f"sha256:{manifest.artifact_sha256}"
        or manifest.training_evidence.get("passed") is not True
        or manifest.smoke_test.get("passed") is not True
    ):
        raise ArtifactValidationError("Artifact manifest compatibility mismatch")


def manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def attach_manifest(bundle: dict, manifest: ArtifactManifest) -> dict:
    validate_manifest(bundle, manifest)
    if (
        "trajectory_metadata" in bundle
        and bundle["trajectory_metadata"] != manifest.trajectory_metadata()
    ):
        raise ArtifactValidationError("Conflicting embedded and sidecar metadata")
    return {
        **bundle,
        "trajectory_metadata": manifest.trajectory_metadata(),
        "artifact_manifest": manifest.model_dump(),
    }
