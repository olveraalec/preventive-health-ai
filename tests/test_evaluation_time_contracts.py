"""Frozen prediction/outcome clocks and optional-provider independence."""

import json
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from healthcare_ai_lab.evaluation.contracts import EvaluationConfig, PredictionRecord
from healthcare_ai_lab.evaluation.patient_classification import evaluate_policies
from healthcare_ai_lab.evaluation.registry import classification_metric_definitions


def synthetic_pairs(n_patients=2):
    predictions, outcomes = [], []
    for index in range(n_patients):
        episode = f"synthetic-{index:03}"
        outcomes.append(
            {
                "split": "synthetic",
                "episode_id": episode,
                "icu_hour": 1,
                "y_true": 1,
                "outcome_time": 1,
                "available_at": 5,
            }
        )
        for name, kind, weight in (
            ("static", "static", 0),
            ("continuous", "continuous_allocation", 0.25),
        ):
            predictions.append(
                {
                    "split": "synthetic",
                    "episode_id": episode,
                    "icu_hour": 1,
                    "decision_time": 1,
                    "observation_cutoff": 1,
                    "policy_id": name,
                    "policy_kind": kind,
                    "policy_version": "v1",
                    "artifact_version": "synthetic-v1",
                    "feature_schema_version": "synthetic-v1",
                    "q_temporal": weight,
                    "static_probability": 0.2,
                    "temporal_probability": 0.8,
                    "final_probability": 0.2 + weight * 0.6,
                }
            )
    return predictions, outcomes


def test_monte_carlo_patient_permutation_deterministic_and_plus_one():
    inputs = synthetic_pairs(13)
    config = EvaluationConfig(n_bootstrap=31, n_permutations=31, seed=12)
    report = evaluate_policies(
        *inputs,
        evaluated_at=5,
        experiment_id="synthetic",
        reference="static",
        config=config,
    )
    again = evaluate_policies(
        inputs[0][::-1],
        inputs[1][::-1],
        evaluated_at=5,
        experiment_id="synthetic",
        reference="static",
        config=config,
    )
    assert report == again
    for comp in report.comparisons:
        assert comp.n_patients == 13
        assert comp.permutation_method == "monte_carlo_patient_sign_flip_plus_one"
        assert comp.permutation_replicates == 31
        assert comp.metrics["paired_patient_sign_flip_p_value"].value >= 1 / 32
        assert comp.metrics["paired_patient_win_rate"].value == 1


def test_clock_metadata_missing_outcomes_and_frozen_experts_rejected():
    predictions, outcomes = synthetic_pairs()
    with pytest.raises(ValueError):
        evaluate_policies(
            predictions, outcomes[:-1], evaluated_at=5, experiment_id="missing"
        )
    with pytest.raises(ValueError):
        evaluate_policies(
            predictions, outcomes, evaluated_at=np.nan, experiment_id="time"
        )
    predictions[0]["decision_time"] = 2
    with pytest.raises(ValueError, match="clocks"):
        evaluate_policies(predictions, outcomes, evaluated_at=5, experiment_id="clocks")


def test_missing_temporal_and_fallback_contract():
    predictions, outcomes = synthetic_pairs(1)
    row = predictions[1]
    row.update(
        q_temporal=0,
        final_probability=0.2,
        temporal_probability=None,
        status="static_only_fallback",
        fallback_reason="synthetic_secondary_failure",
    )
    report = evaluate_policies(
        predictions, outcomes, evaluated_at=5, experiment_id="fallback"
    )
    continuous = next(
        policy for policy in report.policies if policy.policy_id == "continuous"
    )
    assert continuous.metrics["emergency_fallback_count"].value == 1
    assert continuous.metrics["patient_mean_fallback_frequency"].value == 1
    assert (
        continuous.metrics["patient_mean_expert_disagreement_mean_squared"].value
        is None
    )
    assert (
        continuous.metrics["patient_mean_expert_disagreement_mean_squared"].n_patients
        == 0
    )
    row.pop("fallback_reason")
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(row)


def test_constant_reference_metadata_survives_emergency_fallback():
    row = synthetic_pairs(1)[0][0]
    row.update(
        policy_kind="constant_mixture",
        status="static_only_fallback",
        fallback_reason="synthetic",
    )
    with pytest.raises(ValidationError, match="configured reference"):
        PredictionRecord.model_validate(row)
    row["constant_weight"] = 0.25
    assert PredictionRecord.model_validate(row).q_temporal == 0


@pytest.mark.parametrize(
    "change",
    ["feedback_duplicate", "feedback_missing", "future_decision", "future_outcome"],
)
def test_temporal_contract_failures(change):
    predictions, outcomes = synthetic_pairs(1)
    row = predictions[0]
    if change.startswith("feedback"):
        row.update(
            icu_hour=3,
            decision_time=3,
            observation_cutoff=3,
            feedback_protocol="oracle_delay_simulation",
            feedback_hours=[2, 2] if change.endswith("duplicate") else [2],
        )
    elif change == "future_decision":
        row["decision_time"] = 10
    else:
        outcomes[0]["available_at"] = 10
    with pytest.raises(ValueError):
        evaluate_policies(
            predictions, outcomes, evaluated_at=5, experiment_id="time-failure"
        )


def test_empty_cohort_and_unknown_reference_fail():
    with pytest.raises(ValueError):
        evaluate_policies([], [], evaluated_at=5, experiment_id="empty")
    with pytest.raises(ValueError, match="Reference"):
        evaluate_policies(
            *synthetic_pairs(),
            evaluated_at=5,
            experiment_id="reference",
            reference="missing",
        )
    with pytest.raises(ValidationError):
        EvaluationConfig(n_bootstrap=0)


def test_registry_definition_metadata_is_json_safe_and_separate():
    specs = classification_metric_definitions()
    assert "nll" not in specs and "bernoulli_log_loss" in specs
    assert specs["bernoulli_log_loss"].distribution == "Bernoulli"
    assert specs["bernoulli_log_loss"].direction == "minimize"
    assert specs["ece_binned"].role == "monitoring"
    json.dumps(
        {name: spec.model_dump() for name, spec in specs.items()}, allow_nan=False
    )


def test_evaluation_imports_without_optional_agent_or_llm_components():
    code = (
        "import sys\n"
        "from healthcare_ai_lab.evaluation.registry import evaluate_classification\n"
        "from healthcare_ai_lab.evaluation.patient_classification import evaluate_policies\n"
        "assert evaluate_classification([0,1],[0.1,0.9]).metrics['brier_score'].value > 0\n"
        "assert not any(name.split('.')[0] in {'langgraph','langchain','openai','anthropic'} for name in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
