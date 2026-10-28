"""Notebook parity and temporal invariants, using synthetic records only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from healthcare_ai_lab.preprocessing.sepsis import (
    NOTEBOOK_TRANSFORM_SHA256,
    RAW_REQUIRED,
    IncrementalSepsisFeatures,
    build_sepsis_features,
    feature_sources,
)

FIXTURE = Path(__file__).parent / "fixtures/sepsis_notebook02_synthetic.json"


@pytest.fixture
def reference():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def patient(reference):
    raw = pd.DataFrame(reference["raw"])
    return raw.loc[raw.patient_id == "synthetic-a"].reset_index(drop=True)


def test_notebook_reference_parity(reference):
    assert reference["notebook_transform_sha256"] == NOTEBOOK_TRANSFORM_SHA256
    raw = pd.DataFrame(reference["raw"])
    actual = pd.concat(
        [
            build_sepsis_features(part)
            for _, part in raw.groupby("patient_id", sort=False)
        ],
        ignore_index=True,
    )
    expected = pd.DataFrame(reference["expected"])
    expected = expected.astype(
        {name: float for name in expected if name != "patient_id"}
    )
    assert list(actual.columns) == reference["expected_columns"]
    pd.testing.assert_frame_equal(
        actual, expected, check_dtype=False, rtol=1e-13, atol=1e-13
    )
    for col in (name for name in actual if name.endswith("_observed")):
        assert actual[col].dtype == np.dtype("int8")
    assert set(actual) - {"patient_id"} == set(feature_sources())


def test_local_notebook_source_fingerprint():
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks/02_temporal_feature_engineering.ipynb"
    )
    if not path.is_file():
        pytest.skip("Optional local notebook is not distributed")
    notebook = json.loads(path.read_text())
    source = "\n".join("".join(notebook["cells"][i]["source"]) for i in (9, 10))
    assert hashlib.sha256(source.encode()).hexdigest() == NOTEBOOK_TRANSFORM_SHA256


def test_local_processed_schema_only(reference):
    # Read schema metadata only; never read patient row groups.
    import pyarrow.parquet as pq

    path = (
        Path(__file__).resolve().parents[1]
        / "data/processed/sepsis_temporal_features.parquet"
    )
    if not path.is_file():
        pytest.skip("Optional local processed schema is unavailable")
    assert set(pq.read_schema(path).names) - {"SepsisLabel", "split"} == set(
        reference["expected_columns"]
    )


def test_incremental_matches_batch_and_all_prefixes(patient):
    batch = build_sepsis_features(patient)
    stream = IncrementalSepsisFeatures("synthetic-a")
    rows = []
    for i, row in enumerate(patient.to_dict("records")):
        rows.append(stream.update(row))
        prefix = build_sepsis_features(patient.iloc[: i + 1])
        pd.testing.assert_frame_equal(batch.iloc[: i + 1], prefix, check_exact=True)
    incremental = pd.concat(rows, ignore_index=True)
    pd.testing.assert_frame_equal(batch, incremental, check_exact=True)


def test_missingness_and_gap_semantics(patient):
    result = build_sepsis_features(patient)
    assert np.isnan(result.loc[0, "HR_last"])
    assert np.isnan(result.loc[0, "HR_age"])
    assert result.loc[3, "ICULOS"] == 5
    assert result.loc[3, "HR_age"] == 3  # last observed at hour 2
    assert result.loc[3, "HR_last"] == 81
    assert result.loc[3, "HR_mean6"] == 81
    assert result.loc[3, "HR_std6"] == 0
    assert result.Temp_last.isna().all()
    assert result.Temp_observed.eq(0).all()
    assert result.loc[4, "HR_delta3"] == 3  # three records back, not hour 3
    assert np.isnan(result.loc[0, "HRtrend_x_MAPtrend"])
    assert "SepsisLabel" not in result and "split" not in result


def test_outcomes_and_future_measurements_cannot_change_prefix(patient):
    original = build_sepsis_features(patient)
    changed = patient.copy()
    changed["SepsisLabel"] = 1
    changed["nominal_onset_hour"] = 2
    pd.testing.assert_frame_equal(original, build_sepsis_features(changed))
    changed.loc[7:, list(RAW_REQUIRED)] = 9999
    pd.testing.assert_frame_equal(
        original.iloc[:7], build_sepsis_features(changed).iloc[:7]
    )


def test_no_cross_patient_filling(reference):
    raw = pd.DataFrame(reference["raw"])
    with pytest.raises(ValueError, match="one nonempty patient"):
        build_sepsis_features(raw)
    first = IncrementalSepsisFeatures("synthetic-a")
    second = IncrementalSepsisFeatures("synthetic-b")
    first.update(raw[raw.patient_id == "synthetic-a"].iloc[1].to_dict())
    initial = second.update(raw[raw.patient_id == "synthetic-b"].iloc[0].to_dict())
    assert initial.HR_last.isna().all()


@pytest.mark.parametrize("hour", [0, -1, 1.5, np.nan, np.inf, 1])
def test_invalid_hours(patient, hour):
    patient["ICULOS"] = patient.ICULOS.astype(float)
    patient.loc[1, "ICULOS"] = hour
    with pytest.raises(ValueError, match="ICU hours"):
        build_sepsis_features(patient)


def test_invalid_sources_and_transactional_update(patient):
    with pytest.raises(ValueError, match="Missing raw source"):
        build_sepsis_features(patient.drop(columns="HR"))
    bad = patient.copy()
    bad.loc[0, "HR"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        build_sepsis_features(bad)
    stream = IncrementalSepsisFeatures("synthetic-a")
    records = patient.to_dict("records")
    stream.update(records[0])
    missing = dict(records[1])
    del missing["HR"]
    with pytest.raises(ValueError, match="Missing raw source"):
        stream.update(missing)
    with pytest.raises(ValueError, match="Mixed patient"):
        stream.update({**records[1], "patient_id": "synthetic-b"})
    actual = stream.update(records[1])
    pd.testing.assert_frame_equal(
        actual, build_sepsis_features(patient.iloc[:2]).iloc[[-1]]
    )
