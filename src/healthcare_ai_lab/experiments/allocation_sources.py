"""Explicit synthetic/frozen-local sources. Never refit or write base artifacts."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold

from healthcare_ai_lab.experiments.allocation_comparison import (
    CONTEXT_COLUMNS,
    ExperimentError,
    ExperimentSource,
    FrozenCohort,
    OOFProof,
)
from healthcare_ai_lab.models.artifact import META_FEATURES, sha256_file
from healthcare_ai_lab.models.sepsis_inference import load_bundle, static_entropy


def synthetic_source(
    seed=12345, n_training=900, n_evaluation=300, hours=12, scenario="informative"
):
    if (
        n_training < 10
        or n_evaluation < 2
        or hours < 1
        or scenario not in ("informative", "uninformative", "shifted")
    ):
        raise ExperimentError("Invalid synthetic design")
    rng = np.random.default_rng(seed)
    context_rng = np.random.default_rng(seed + 1)

    def cohort(count, split):
        context, outcomes = [], []
        for patient in range(count):
            episode = f"synthetic-{split}-{patient:06}"
            severity = rng.normal()
            regime = int(rng.integers(0, 2))
            last_seen = {"HR": None, "Resp": None}
            for hour in range(1, hours + 1):
                if rng.random() < 0.15:
                    regime = 1 - regime
                cue = (
                    int(context_rng.integers(0, 2))
                    if scenario == "uninformative"
                    else regime
                )
                if scenario == "shifted" and split != "train":
                    cue = 1 - cue
                # Observable missingness/freshness is intentionally informative in
                # the favorable control, not an observed clinical relationship.
                observed = {
                    "HR": int(cue == 0),
                    "Resp": int(context_rng.random() < (0.85 if cue == 0 else 0.15)),
                }
                ages = {}
                for signal, previous_hour in last_seen.items():
                    if observed[signal]:
                        previous_hour = hour
                        last_seen[signal] = previous_hour
                    ages[signal] = (
                        hour - previous_hour if previous_hour is not None else np.nan
                    )
                logit = -2.5 + severity + 0.4 * np.sin(hour / 3)
                p_true = expit(logit)
                # Frozen stochastic experts never use labels or future observations.
                ps = expit(logit + rng.normal(scale=0.15 if regime == 0 else 1.6))
                pt = expit(logit + rng.normal(scale=1.6 if regime == 0 else 0.15))
                context.append(
                    {
                        "episode_id": episode,
                        "icu_hour": hour,
                        "p_static": ps,
                        "p_temporal": pt,
                        "HR_observed": observed["HR"],
                        "Resp_observed": observed["Resp"],
                        "Temp_observed": int(context_rng.random() < 0.7),
                        "HR_age": ages["HR"],
                        "Resp_age": ages["Resp"],
                    }
                )
                outcomes.append(
                    {
                        "episode_id": episode,
                        "icu_hour": hour,
                        "y_true": int(rng.random() < p_true),
                    }
                )
        return FrozenCohort(
            pd.DataFrame(context),
            pd.DataFrame(outcomes),
            "synthetic_train" if split == "train" else "synthetic_test",
            "synthetic-label-independent-experts-v1",
            "synthetic-causal-observability-v1",
        )

    training, evaluation = cohort(n_training, "train"), cohort(n_evaluation, "test")
    ids = sorted(training.context.episode_id.unique())
    proof = OOFProof(
        {patient: index % 3 for index, patient in enumerate(ids)},
        {fold: frozenset() for fold in range(3)},
        "synthetic_fixed_label_independent_experts_empty_fit_sets",
    )
    return ExperimentSource(
        training,
        evaluation,
        proof,
        {
            "origin": "synthetic",
            "scenario": scenario,
            "seed": seed,
            "fresh_holdout": True,
            "training_patients": n_training,
            "evaluation_patients": n_evaluation,
            "hours_per_patient": hours,
            "base_models_fitted": False,
            "context_information": "constructed_not_clinically_discovered",
            "existing_baseline": "Notebook09_gain_estimator_form_fit_on_synthetic_training_only",
            "source_adapter_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    )


def local_validation_source(root=None):
    """Existing training OOF cache + artifact validation predictions; never test rows."""
    root = Path.cwd() if root is None else Path(root)
    artifact = root / "artifacts/models/prototype_bundle.joblib"
    oof_path = root / "results/experiments/nested_controller_predictions.parquet"
    temporal_path = root / "data/processed/sepsis_temporal_features.parquet"
    if not all(
        p.is_file()
        for p in (
            artifact,
            artifact.with_suffix(".manifest.json"),
            oof_path,
            temporal_path,
        )
    ):
        raise ExperimentError(
            "Trusted artifact/manifest, training OOF cache or engineered source is missing"
        )
    bundle = load_bundle(artifact)
    manifest = bundle.get("artifact_manifest")
    if manifest is None:
        raise ExperimentError("Activated checksum-bound artifact manifest is required")
    if manifest["source_hashes"].get("oof") != sha256_file(oof_path) or manifest[
        "source_hashes"
    ].get("temporal") != sha256_file(temporal_path):
        raise ExperimentError(
            "Training evidence no longer matches the activated artifact"
        )
    columns = ["patient_id", "ICULOS", "SepsisLabel", *META_FEATURES[1:-1]]
    # Predicate-filtered projections do not select reserved test records.
    train = pd.read_parquet(
        temporal_path, columns=columns, filters=[("split", "==", "train")]
    )
    validation = pd.read_parquet(
        temporal_path, columns=columns, filters=[("split", "==", "validation")]
    )
    train = train.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    validation = validation.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    if (
        train.empty
        or validation.empty
        or set(train.patient_id) & set(validation.patient_id)
    ):
        raise ExperimentError("Training/validation cohort coverage invalid")
    pred = pd.read_parquet(oof_path)
    pred = pred[pred.policy.isin(["static", "temporal"])].copy()
    if (
        set(pred.policy) != {"static", "temporal"}
        or not pred.groupby(["patient_id", "ICULOS"]).policy.nunique().eq(2).all()
    ):
        raise ExperimentError("OOF table requires both experts at every patient-hour")
    if pred.duplicated(["patient_id", "ICULOS", "policy"]).any() or set(
        pred.patient_id
    ) != set(train.patient_id):
        raise ExperimentError(
            "OOF prediction coverage does not match original training patients"
        )
    if not pred.groupby("patient_id").fold.nunique().eq(1).all():
        raise ExperimentError("OOF patient-fold isolation failed")
    assignment = {}
    fitting = {}
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=12345)
    for fold, (fit_index, hold_index) in enumerate(
        splitter.split(train, train.SepsisLabel, groups=train.patient_id)
    ):
        hold_ids = set(train.iloc[hold_index].patient_id)
        if not pred[pred.patient_id.isin(hold_ids)].fold.eq(fold).all():
            raise ExperimentError("Recorded OOF folds do not reproduce Notebook09")
        fitting[fold] = frozenset(train.iloc[fit_index].patient_id)
        assignment.update(dict.fromkeys(hold_ids, fold))
    labels = pred[["patient_id", "ICULOS", "y_true"]].drop_duplicates()
    if labels.duplicated(["patient_id", "ICULOS"]).any():
        raise ExperimentError("OOF expert labels disagree")
    checked = train[["patient_id", "ICULOS", "SepsisLabel"]].merge(
        labels, on=["patient_id", "ICULOS"], how="outer", validate="one_to_one"
    )
    if not checked.SepsisLabel.eq(checked.y_true).all():
        raise ExperimentError("OOF targets do not match original training")
    wide = pred.pivot(
        index=["patient_id", "ICULOS"], columns="policy", values="p_pred"
    ).reset_index()
    train = train.merge(
        wide, on=["patient_id", "ICULOS"], how="left", validate="one_to_one"
    )
    # Project full model inputs only for original validation, with no .fit calls.
    required = list(
        dict.fromkeys(
            [
                "patient_id",
                "ICULOS",
                *bundle["static"]["features"],
                *bundle["temporal"]["features"],
            ]
        )
    )
    base_inputs = pd.read_parquet(
        temporal_path, columns=required, filters=[("split", "==", "validation")]
    )
    base_inputs = base_inputs.sort_values(["patient_id", "ICULOS"]).reset_index(
        drop=True
    )
    if not base_inputs[["patient_id", "ICULOS"]].equals(
        validation[["patient_id", "ICULOS"]]
    ):
        raise ExperimentError("Frozen validation base-input alignment failed")
    for name in ("static", "temporal"):
        entry = bundle[name]
        validation[name] = entry["model"].predict_proba(base_inputs[entry["features"]])[
            :, 1
        ]

    def package(frame, split):
        context = frame.rename(
            columns={
                "patient_id": "episode_id",
                "ICULOS": "icu_hour",
                "static": "p_static",
                "temporal": "p_temporal",
            }
        )
        return FrozenCohort(
            context[list(CONTEXT_COLUMNS)].copy(),
            context[["episode_id", "icu_hour", "SepsisLabel"]].rename(
                columns={"SepsisLabel": "y_true"}
            ),
            split,
            manifest["model"]["model_version"],
            manifest["provenance"]["feature_schema"],
        )

    # Only an aggregate consistency check is retained, never per-patient outputs.
    prototype_context = validation.copy()
    prototype_context["entropy_static"] = static_entropy(prototype_context["static"])
    if not np.isfinite(
        bundle["controller"].predict(prototype_context[META_FEATURES])
    ).all():
        raise ExperimentError("Existing anticipatory signal unavailable")
    return ExperimentSource(
        package(train, "train"),
        package(validation, "validation"),
        OOFProof(
            assignment,
            fitting,
            "Notebook09_fold_assignment_reconstructed_not_independently_attested",
        ),
        {
            "origin": "local_previously_inspected_validation",
            "fresh_holdout": False,
            "reserved_test_records_selected": False,
            "reserved_test_patient_count_reported_by_existing_manifest": manifest[
                "training_evidence"
            ]["split_patient_counts"]["test"],
            "artifact_sha256": manifest["artifact_sha256"],
            "artifact_manifest_sha256": sha256_file(
                artifact.with_suffix(".manifest.json")
            ),
            "oof_sha256": sha256_file(oof_path),
            "temporal_sha256": sha256_file(temporal_path),
            "historical_provenance": manifest["training_provenance_status"],
            "inner_reliability": "global_OOF_non_nested_training_selection_heuristic",
            "existing_baseline": "unchanged_trusted_artifact_controller",
            "source_adapter_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        bundle["controller"],
    )
