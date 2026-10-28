"""Recorded evidence is immutable metadata, separate from live evaluation."""

import hashlib
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from healthcare_ai_lab.api.main import app
from healthcare_ai_lab.evaluation.registry import METRICS, SUMMARY_METRICS, evaluate
from healthcare_ai_lab.evaluation.summary import (
    EvaluationSummary,
    RecordedMetric,
    recorded_evaluation_summary,
)


def test_recorded_values_match_pinned_source_not_patient_model_performance():
    result = recorded_evaluation_summary()
    experiment = result.experiments[0]
    source = Path(__file__).resolve().parents[1] / experiment.provenance.source_path
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == experiment.provenance.source_sha256
    )
    rows = [
        line
        for line in source.read_text().splitlines()
        if re.match(
            r"\| (Static|Training-selected|Global Brier|Conditional two-regime|Shuffled-context)",
            line,
        )
    ]
    assert len(rows) == len(experiment.policies) == 5
    for row, policy in zip(rows, experiment.policies, strict=True):
        values = [
            float(value.replace("*", "").strip()) for value in row.split("|")[2:5]
        ]
        assert [m.value for m in policy.metrics] == values
    assert experiment.comparison.metric.value == 0.006495
    assert experiment.comparison.ci95 == (0.004191, 0.008847)
    assert experiment.dataset == "synthetic_binary_sequences"
    assert experiment.provenance.model_artifact_sha256 is None
    assert experiment.provenance.score_clipping_epsilon is None
    assert result.installed_sepsis_model_evaluation == "unavailable"


def test_evaluation_api_requires_no_bundle_database_or_label_calculation():
    client = TestClient(app)
    with (
        patch(
            "healthcare_ai_lab.api.main.get_research_bundle",
            side_effect=AssertionError("No model"),
        ),
        patch(
            "healthcare_ai_lab.evaluation.registry.evaluate",
            side_effect=AssertionError("No labels"),
        ),
    ):
        response = client.get("/api/v1/research/evaluation-summary")
    assert response.status_code == 200
    result = EvaluationSummary.model_validate(response.json())
    assert not result.live_labels_used
    assert result.research_only and not result.clinical_use
    assert "auroc" in result.unavailable_metrics
    for policy in result.experiments[0].policies:
        for metric in policy.metrics:
            spec = SUMMARY_METRICS[metric.metric_id]
            assert metric.direction == spec.direction
            assert metric.aggregation == spec.aggregation


def test_gaussian_registry_is_unchanged_and_independent():
    assert set(METRICS) == {
        "rmse",
        "mae",
        "bias",
        "p90_absolute_error",
        "nll",
        "crps",
        "coverage_90",
        "width_90",
    }
    scores = evaluate([0, 1], [0, 1], sigma=1)
    assert scores["nll"] == pytest.approx(0.9189385332046727)
    assert scores["rmse"] == 0
    assert not set(SUMMARY_METRICS).intersection(METRICS)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_metric_values_rejected(value):
    metric = (
        recorded_evaluation_summary().experiments[0].policies[0].metrics[0].model_dump()
    )
    metric["value"] = value
    with pytest.raises(ValidationError):
        RecordedMetric.model_validate(metric)


def test_catalog_requests_do_not_share_mutable_state():
    first = recorded_evaluation_summary()
    first.experiments.clear()
    assert len(recorded_evaluation_summary().experiments) == 1


def test_documented_frontend_example_matches_response():
    example = Path(__file__).resolve().parents[1] / "docs/examples/evaluation-summary.v0.6.synthetic.json"
    assert json.loads(example.read_text()) == recorded_evaluation_summary().model_dump(mode="json")
