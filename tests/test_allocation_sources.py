"""Synthetic frozen-source fixtures: no patient records or model binaries."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from healthcare_ai_lab.experiments import allocation_comparison as experiment
from healthcare_ai_lab.experiments import allocation_sources as sources
from healthcare_ai_lab.models.artifact import sha256_file


def test_synthetic_context_ablation_preserves_experts_labels_and_patients():
    cohorts = [
        sources.synthetic_source(n_training=36, n_evaluation=8, hours=4, scenario=name)
        for name in ("informative", "uninformative", "shifted")
    ]
    for other in cohorts[1:]:
        for name in ("training", "evaluation"):
            expected, actual = getattr(cohorts[0], name), getattr(other, name)
            pd.testing.assert_frame_equal(expected.outcomes, actual.outcomes)
            pd.testing.assert_frame_equal(
                expected.context[["episode_id", "icu_hour", "p_static", "p_temporal"]],
                actual.context[["episode_id", "icu_hour", "p_static", "p_temporal"]],
            )
    pd.testing.assert_frame_equal(
        cohorts[0].training.context, cohorts[2].training.context
    )
    assert not cohorts[0].evaluation.context.equals(cohorts[2].evaluation.context)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_training": 9}, {"n_evaluation": 1}, {"hours": 0}, {"scenario": "unknown"}],
)
def test_invalid_synthetic_design(kwargs):
    with pytest.raises(experiment.ExperimentError):
        sources.synthetic_source(**kwargs)


@pytest.fixture
def local_fixture(tmp_path, monkeypatch):
    sample = sources.synthetic_source(n_training=36, n_evaluation=8, hours=4)
    frames = []
    for name, split in (("training", "train"), ("evaluation", "validation")):
        cohort = getattr(sample, name)
        frame = cohort.context.merge(cohort.outcomes, on=["episode_id", "icu_hour"])
        frame = frame.rename(
            columns={
                "episode_id": "patient_id",
                "icu_hour": "ICULOS",
                "y_true": "SepsisLabel",
            }
        )
        frame["split"] = split
        frames.append(frame)
    reserved = frames[1].copy()
    reserved["patient_id"] = "synthetic-reserved"
    reserved["split"] = "test"
    reserved["p_static"] = np.inf  # Inference must not select these rows.
    data = pd.concat([*frames, reserved], ignore_index=True)
    temporal_path = tmp_path / "data/processed/sepsis_temporal_features.parquet"
    temporal_path.parent.mkdir(parents=True)
    data.to_parquet(temporal_path, index=False)
    train = frames[0].sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=12345)
    pieces = []
    for fold, (_, hold) in enumerate(
        splitter.split(train, train.SepsisLabel, groups=train.patient_id)
    ):
        for policy in ("static", "temporal"):
            frame = train.iloc[hold][
                ["patient_id", "ICULOS", "SepsisLabel", f"p_{policy}"]
            ].copy()
            frame = frame.rename(
                columns={"SepsisLabel": "y_true", f"p_{policy}": "p_pred"}
            )
            frame["fold"], frame["policy"] = fold, policy
            pieces.append(frame)
    oof_path = tmp_path / "results/experiments/nested_controller_predictions.parquet"
    oof_path.parent.mkdir(parents=True)
    pd.concat(pieces).to_parquet(oof_path, index=False)
    artifact = tmp_path / "artifacts/models/prototype_bundle.joblib"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("synthetic placeholder; never deserialize")
    artifact.with_suffix(".manifest.json").write_text("{}")

    class FrozenExpert:
        def __init__(self, column):
            self.column = column

        def predict_proba(self, matrix):
            assert matrix.columns.tolist() == [self.column]
            p = matrix[self.column].to_numpy()
            assert np.isfinite(p).all()
            return np.c_[1 - p, p]

        def fit(self, *args, **kwargs):
            pytest.fail("Frozen base models must never be refitted")

    class ExistingController:
        def predict(self, matrix):
            return np.zeros(len(matrix))

        def fit(self, *args, **kwargs):
            pytest.fail("Existing controller must remain frozen")

    bundle = {
        "static": {"features": ["p_static"], "model": FrozenExpert("p_static")},
        "temporal": {"features": ["p_temporal"], "model": FrozenExpert("p_temporal")},
        "controller": ExistingController(),
        "artifact_manifest": {
            "source_hashes": {
                "oof": sha256_file(oof_path),
                "temporal": sha256_file(temporal_path),
            },
            "model": {"model_version": "synthetic-frozen-v1"},
            "provenance": {"feature_schema": "synthetic-features-v1"},
            "training_evidence": {"split_patient_counts": {"test": 1}},
            "training_provenance_status": "synthetic-fixture",
            "artifact_sha256": sha256_file(artifact),
        },
    }
    monkeypatch.setattr(sources, "load_bundle", lambda path: bundle)
    reads = []
    read_parquet = pd.read_parquet

    def observed_read(path, **kwargs):
        reads.append((path, kwargs))
        return read_parquet(path, **kwargs)

    monkeypatch.setattr(sources.pd, "read_parquet", observed_read)
    return tmp_path, bundle, reads, sample, artifact


def test_local_adapter_never_selects_test_rows_or_refits(local_fixture):
    root, bundle, reads, expected, artifact = local_fixture
    before = sha256_file(artifact)
    source = sources.local_validation_source(root)
    assert source.existing_gain_model is bundle["controller"]
    assert source.metadata["fresh_holdout"] is False
    assert source.metadata["reserved_test_records_selected"] is False
    assert (
        source.metadata["reserved_test_patient_count_reported_by_existing_manifest"]
        == 1
    )
    for path, kwargs in reads:
        if path.name == "sepsis_temporal_features.parquet":
            assert kwargs["filters"] in (
                [("split", "==", "train")],
                [("split", "==", "validation")],
            )
    for name in ("training", "evaluation"):
        pd.testing.assert_frame_equal(
            getattr(source, name).context,
            getattr(expected, name).context,
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            getattr(source, name).outcomes.reset_index(drop=True),
            getattr(expected, name).outcomes.reset_index(drop=True),
            check_dtype=False,
        )
    result = experiment.run_experiment(
        source, experiment.ExperimentConfig(n_bootstrap=16)
    )
    assert result["synthetic_support_for_adaptation"] is False
    assert sha256_file(artifact) == before


def test_local_source_defaults_to_callers_working_directory(local_fixture, monkeypatch):
    root, _, _, _, _ = local_fixture
    monkeypatch.chdir(root)
    assert sources.local_validation_source().metadata["fresh_holdout"] is False


@pytest.mark.parametrize(
    "mutation", ["checksum", "manifest", "artifact", "fold", "label", "missing_expert"]
)
def test_local_provenance_fails_closed(local_fixture, mutation):
    root, bundle, _, _, artifact = local_fixture
    oof_path = root / "results/experiments/nested_controller_predictions.parquet"
    if mutation == "checksum":
        bundle["artifact_manifest"]["source_hashes"]["oof"] = "0" * 64
    elif mutation == "manifest":
        bundle.pop("artifact_manifest")
    elif mutation == "artifact":
        artifact.unlink()
    else:
        frame = pd.read_parquet(oof_path)
        if mutation == "fold":
            frame["fold"] = (frame.fold + 1) % 3
        elif mutation == "label":
            frame["y_true"] = 1 - frame.y_true
        else:
            frame = frame[frame.policy == "static"]
        frame.to_parquet(oof_path, index=False)
        bundle["artifact_manifest"]["source_hashes"]["oof"] = sha256_file(oof_path)
    with pytest.raises(experiment.ExperimentError):
        source = sources.local_validation_source(root)
        experiment.training_rows(source.training)


def test_cli_aggregate_json_and_no_overwrite(tmp_path, monkeypatch, capsys):
    output = tmp_path / "aggregate.json"
    args = [
        "allocation-comparison",
        "--train-patients",
        "12",
        "--evaluation-patients",
        "2",
        "--hours",
        "2",
        "--bootstrap",
        "8",
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", args)
    experiment.main()
    result = json.loads(output.read_text())
    assert result["evaluation_patients"] == 2
    assert "synthetic-test-" not in capsys.readouterr().out
    before = output.read_bytes()
    with pytest.raises(SystemExit, match="FileExistsError"):
        experiment.main()
    assert output.read_bytes() == before


def test_missing_local_source_error_is_privacy_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "allocation-comparison",
            "--source",
            "local-validation",
            "--root",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit, match="source is missing"):
        experiment.main()
    assert capsys.readouterr().out == ""
