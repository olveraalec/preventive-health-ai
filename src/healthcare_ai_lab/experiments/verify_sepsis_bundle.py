"""Local retrospective verification, never fitting models or printing patient records.

Run with --activate to write a checksum-bound sidecar after all checks pass.
All printed/persisted results are aggregate consistency/parity diagnostics.
"""

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.special import xlogy
from sklearn.model_selection import StratifiedGroupKFold

from healthcare_ai_lab.models.artifact import (
    PREPROCESSING,
    STATIC_FEATURES,
    TEMPORAL_FEATURES,
    ArtifactManifest,
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
from healthcare_ai_lab.preprocessing.sepsis import (
    NOTEBOOK_TRANSFORM_SHA256,
    RAW_REQUIRED,
    build_sepsis_features,
    feature_sources,
)
from healthcare_ai_lab.trajectories.contracts import ModelMetadata

NOTEBOOK09_SOURCE_SHA256 = (
    "304d743becef8a8e587899e7a55f2b42e4bea7a356d697cf30207b150fd18670"
)


class VerificationError(ValueError):
    """A static diagnostic safe to print without patient details."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def source_hash(path: Path, cells: tuple[int, ...]) -> str:
    notebook = json.loads(path.read_text())
    source = "\n".join("".join(notebook["cells"][i]["source"]) for i in cells)
    return hashlib.sha256(source.encode()).hexdigest()


def max_difference(left, right) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    require(a.shape == b.shape, "Comparison shape mismatch")
    require(np.array_equal(np.isnan(a), np.isnan(b)), "Missingness mismatch")
    valid = ~np.isnan(a)
    require(
        np.isfinite(a[valid]).all() and np.isfinite(b[valid]).all(),
        "Nonfinite comparison",
    )
    return float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0


def training_consistency(
    bundle: dict, temporal_path: Path, oof_path: Path, splits_path: Path
) -> dict:
    """Reconstruct preprocessing and weighted-ridge stationarity, without .fit()."""
    inspect_bundle(bundle)
    data = pd.read_parquet(temporal_path)
    split = pd.read_csv(splits_path, index_col=0).reset_index(names="patient_id")
    split = split[["patient_id", "split"]]
    require(not split.patient_id.duplicated().any(), "Duplicate split membership")
    require(
        set(split.split) == {"train", "validation", "test"}, "Unknown split membership"
    )
    require(
        data.groupby("patient_id").split.nunique().eq(1).all(), "Patient split overlap"
    )
    observed = data[["patient_id", "split"]].drop_duplicates()
    checked = observed.merge(
        split,
        on="patient_id",
        suffixes=("_data", "_split"),
        how="outer",
        validate="one_to_one",
    )
    require(checked.split_data.eq(checked.split_split).all(), "Split source mismatch")
    train = data[data.split.eq("train")].sort_values(["patient_id", "ICULOS"]).copy()
    require(
        not train.duplicated(["patient_id", "ICULOS"]).any(), "Duplicate training hour"
    )
    evidence = {
        "passed": True,
        "verification_fit_calls": 0,
        "split_patient_counts": split.groupby("split").size().to_dict(),
        "training_rows": len(train),
        "historical_fit_exclusion": "consistent_with_train_only_not_independently_attested",
        "controller_procedure": "Notebook09_training_patient_grouped_3_fold_OOF_gain_equal_patient_weight",
    }
    for role in ("static", "temporal"):
        entry = bundle[role]
        model, features = entry["model"], entry["features"]
        canonical = STATIC_FEATURES if role == "static" else TEMPORAL_FEATURES
        require(
            features == [c for c in canonical if train[c].notna().any()],
            "Training-only feature availability mismatch",
        )
        values = model.named_steps["imputer"].transform(train[features])
        diffs = {
            "median": max_difference(
                model.named_steps["imputer"].statistics_, train[features].median()
            ),
            "mean": max_difference(
                model.named_steps["scaler"].mean_, values.mean(axis=0)
            ),
            "variance": max_difference(
                model.named_steps["scaler"].var_, values.var(axis=0)
            ),
        }
        require(max(diffs.values()) <= 1e-8, "Training preprocessing inconsistency")
        evidence[f"{role}_preprocessing_max_abs_differences"] = diffs
    pred = pd.read_parquet(oof_path)
    pred = pred[pred.policy.isin(["static", "temporal"])].copy()
    require(
        set(pred.patient_id) == set(train.patient_id),
        "OOF includes non-training or misses training patients",
    )
    require(
        pred.groupby("patient_id").fold.nunique().eq(1).all(),
        "OOF fold isolation mismatch",
    )
    require(set(pred.fold) == {0, 1, 2}, "Unexpected OOF folds")
    assignments = train[["patient_id", "ICULOS", "SepsisLabel"]].copy()
    assignments["expected_fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=12345)
    for fold, (_, hold) in enumerate(
        splitter.split(train, train.SepsisLabel, groups=train.patient_id)
    ):
        assignments.iloc[hold, assignments.columns.get_loc("expected_fold")] = fold
    audited = pred.merge(
        assignments, on=["patient_id", "ICULOS"], how="left", validate="many_to_one"
    )
    require(
        audited.fold.eq(audited.expected_fold).all(),
        "Stored OOF fold assignment mismatch",
    )
    require(audited.y_true.eq(audited.SepsisLabel).all(), "Stored OOF outcome mismatch")
    wide = pred.pivot(
        index=["patient_id", "ICULOS"], columns="policy", values="p_pred"
    ).reset_index()
    joined = train.merge(
        wide, on=["patient_id", "ICULOS"], how="outer", validate="one_to_one"
    )
    require(len(joined) == len(train), "OOF row coverage mismatch")
    probabilities = joined[["static", "temporal"]].to_numpy()
    require(
        np.isfinite(probabilities).all()
        and ((probabilities >= 0) & (probabilities <= 1)).all(),
        "Invalid OOF probabilities",
    )
    joined["entropy_static"] = static_entropy(joined["static"])
    features = bundle["meta_features"]
    controller = bundle["controller"]
    imp, scaler, ridge = (
        controller.named_steps[k] for k in ("imputer", "scaler", "ridge")
    )
    imputed = imp.transform(joined[features])
    diffs = {
        "median": max_difference(imp.statistics_, joined[features].median()),
        "mean": max_difference(scaler.mean_, imputed.mean(axis=0)),
        "variance": max_difference(scaler.var_, imputed.var(axis=0)),
    }
    require(max(diffs.values()) <= 1e-8, "Controller preprocessing inconsistency")
    y = joined.SepsisLabel.to_numpy()

    def loss(probability):
        p = np.clip(np.asarray(probability), 1e-10, 1 - 1e-10)
        return -xlogy(y, p) - xlogy(1 - y, 1 - p)

    gain = loss(joined["static"]) - loss(joined["temporal"])
    weights = 1 / joined.groupby("patient_id").patient_id.transform("size").to_numpy()
    weights /= weights.mean()
    X = scaler.transform(imputed)
    residual = X @ ridge.coef_ + ridge.intercept_ - gain
    gradient = float(
        np.max(np.abs(X.T @ (weights * residual) + ridge.alpha * ridge.coef_))
    )
    intercept_gradient = float(abs(weights @ residual))
    require(
        max(gradient, intercept_gradient) < 1e-7,
        "Cross-fitted controller consistency failed",
    )
    evidence.update(
        controller_preprocessing_max_abs_differences=diffs,
        controller_stationarity_max_abs=gradient,
        controller_intercept_stationarity_abs=intercept_gradient,
        oof_training_rows=len(joined),
        oof_fold_assignments_reproduced=True,
        oof_limit="stored_predictions_and_fold_isolation_checked_no_historical_fit_logs",
    )
    return evidence


def smoke_test(
    bundle: dict, raw_path: Path, temporal_path: Path, artifact: Path | None = None
) -> dict:
    """One validation patient, complete matching history; output only maximum errors."""
    from healthcare_ai_lab.api.main import app, get_research_bundle

    cohort = pd.read_parquet(
        temporal_path, columns=["patient_id"], filters=[("split", "==", "validation")]
    )
    require(not cohort.empty, "No validation trajectory available")
    patient = min(cohort.patient_id.unique())
    filters = [("patient_id", "==", patient)]
    raw = pd.read_parquet(
        raw_path, columns=["patient_id", "ICULOS", *RAW_REQUIRED], filters=filters
    )
    raw = raw.sort_values("ICULOS").reset_index(drop=True)
    offline = pd.read_parquet(
        temporal_path, columns=["patient_id", *feature_sources()], filters=filters
    )
    offline = offline.sort_values("ICULOS").reset_index(drop=True)
    require(
        len(raw) > 0 and np.array_equal(raw.ICULOS, offline.ICULOS),
        "Patient-hour coverage mismatch",
    )
    built = build_sepsis_features(raw)
    names = list(feature_sources())
    feature_error = max_difference(built[names], offline[names])
    require(feature_error <= 1e-12, "Offline/raw engineered feature mismatch")
    # Replace source identifiers with a demo alias before invoking the API.
    observations = []
    for row in raw.to_dict("records"):
        measurements = {
            name: None if pd.isna(row[name]) else row[name] for name in RAW_REQUIRED
        }
        observations.append(
            {
                "episode_id": "retrospective-local-demo",
                "icu_hour": int(row["ICULOS"]),
                "measurements": measurements,
            }
        )
    payload = {
        "dataset": "sepsis2019",
        "episode_id": "retrospective-local-demo",
        "model": bundle["trajectory_metadata"]["model"],
        "observations": observations,
    }
    with ExitStack() as stack:
        if artifact is None:
            stack.enter_context(
                patch(
                    "healthcare_ai_lab.api.main.get_research_bundle",
                    return_value=bundle,
                )
            )
        else:
            stack.enter_context(
                patch.dict(
                    "os.environ", {"SEPSIS_MODEL_BUNDLE": str(artifact.resolve())}
                )
            )
            get_research_bundle.cache_clear()
        try:
            response = TestClient(app).post(
                "/api/v1/research/sepsis/trajectory/raw-hourly", json=payload
            )
        finally:
            if artifact is not None:
                get_research_bundle.cache_clear()
    require(response.status_code == 200, "Raw-hourly API smoke test failed")
    hourly = response.json()["predictions"]
    reference = predict_engineered_trajectory(
        bundle, offline.drop(columns="patient_id")
    )
    require(
        all(row["status"] == "bounded_anticipatory" for row in hourly),
        "Unexpected secondary fallback",
    )
    errors = {}
    for key in ("static_probability", "temporal_probability", "final_probability"):
        errors[key] = max_difference(
            [r[key] for r in hourly], [r["prediction"][key] for r in reference]
        )
    errors["temporal_weight"] = max_difference(
        [r["temporal_weight"] for r in hourly],
        [r["controller"]["temporal_weight"] for r in reference],
    )
    # Independent batch scoring checks that row inference retains model feature order.
    for role in ("static", "temporal"):
        entry = bundle[role]
        batch = entry["model"].predict_proba(offline[entry["features"]])[:, 1]
        errors[f"{role}_batch_probability"] = max_difference(
            batch, [r[f"{role}_probability"] for r in hourly]
        )
    require(max(errors.values()) <= 1e-12, "Probability/controller parity failed")
    return {
        "passed": True,
        "cohort": "validation",
        "patients": 1,
        "matching_hours": len(raw),
        "feature_max_abs_difference": feature_error,
        "prediction_max_abs_differences": errors,
        "tolerance": 1e-12,
        "patient_records_persisted": False,
        "api_loader": "checksum_bound_sidecar"
        if artifact is not None
        else "proposed_in_memory_metadata",
    }


def verify(root: Path, artifact: Path, activate: bool = False) -> ArtifactManifest:
    nb02 = root / "notebooks/02_temporal_feature_engineering.ipynb"
    nb09 = root / "notebooks/09_nested_ctonroller_benchmark.ipynb"
    raw = root / "data/processed/sepsis_development.parquet"
    temporal = root / "data/processed/sepsis_temporal_features.parquet"
    splits = root / "data/processed/sepsis_patient_splits.csv"
    oof = root / "results/experiments/nested_controller_predictions.parquet"
    paths = {
        "notebook02": nb02,
        "notebook09": nb09,
        "raw": raw,
        "temporal": temporal,
        "splits": splits,
        "oof": oof,
    }
    require(
        artifact.is_file() and all(p.is_file() for p in paths.values()),
        "Required local evidence missing",
    )
    require(
        source_hash(nb02, (9, 10)) == NOTEBOOK_TRANSFORM_SHA256,
        "Notebook 02 source mismatch",
    )
    require(
        source_hash(nb09, (0, 1, 2, 5)) == NOTEBOOK09_SOURCE_SHA256,
        "Notebook 09 source mismatch",
    )
    # No model fitting is performed. Loading remains limited to the trusted local artifact.
    bundle = load_bundle(artifact)
    features = inspect_bundle(bundle)
    evidence = training_consistency(bundle, temporal, oof, splits)
    digest = sha256_file(artifact)
    manifest = ArtifactManifest(
        artifact_sha256=digest,
        model=ModelMetadata(
            artifact_id="notebook09-prototype", model_version=f"sha256:{digest}"
        ),
        provenance=expected_provenance(),
        source_hashes={k: sha256_file(p) for k, p in paths.items()},
        features=features,
        feature_hashes={k: feature_hash(v) for k, v in features.items()},
        preprocessing=PREPROCESSING,
        verified_runtime_versions=software_versions(),
        training_evidence=evidence,
        smoke_test={},
    )
    candidate = {**bundle, "trajectory_metadata": manifest.trajectory_metadata()}
    # The real legacy binary is unchanged; only metadata is attached in memory.
    candidate.pop("artifact_manifest", None)
    manifest.smoke_test = smoke_test(
        candidate,
        raw,
        temporal,
        artifact if manifest_path(artifact).is_file() else None,
    )
    attach_manifest(bundle, manifest)
    require(digest == sha256_file(artifact), "Artifact changed during verification")
    require(
        manifest.source_hashes == {k: sha256_file(p) for k, p in paths.items()},
        "Evidence changed during verification",
    )
    if activate:
        destination = manifest_path(artifact)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(manifest.model_dump_json(indent=2) + "\n")
        temporary.replace(destination)
        loaded = load_bundle(artifact)
        require(
            loaded["trajectory_metadata"] == manifest.trajectory_metadata(),
            "Activated metadata mismatch",
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/models/prototype_bundle.joblib"),
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Write local sidecar only after verification",
    )
    args = parser.parse_args()
    try:
        artifact = (
            args.artifact if args.artifact.is_absolute() else args.root / args.artifact
        )
        manifest = verify(args.root, artifact, args.activate)
    except VerificationError as exc:
        raise SystemExit(f"Verification failed: {exc}") from None
    except Exception as exc:  # noqa: BLE001 -- avoid patient-bearing library tracebacks
        raise SystemExit(
            f"Verification failed ({type(exc).__name__}); no patient details displayed."
        ) from None
    print(
        json.dumps(
            {
                "activated": args.activate,
                "model": manifest.model.model_dump(),
                "training_provenance_status": manifest.training_provenance_status,
                "training_evidence": manifest.training_evidence,
                "smoke_test": manifest.smoke_test,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
