"""Synthetic patient-time contracts, exact scores, cluster inference and tails."""

import json
from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from healthcare_ai_lab.evaluation.contracts import (
    EvaluationConfig,
    OutcomeRecord,
    PredictionRecord,
)
from healthcare_ai_lab.evaluation.patient_classification import (
    CoverageError,
    empirical_cvar,
    evaluate_policies,
)
from healthcare_ai_lab.evaluation.registry import (
    METRICS,
    evaluate,
    evaluate_classification,
    evaluate_classification_policies,
)


@pytest.fixture
def frozen():
    predictions, outcomes = [], []
    for episode, labels, weights in (
        ("synthetic-a", [0], [0.25]),
        ("synthetic-b", [0, 1, 1], [0.25, 0.15, 0.25]),
    ):
        for hour, (label, weight) in enumerate(
            zip(labels, weights, strict=True), start=1
        ):
            outcomes.append(
                {
                    "split": "synthetic_holdout",
                    "episode_id": episode,
                    "icu_hour": hour,
                    "y_true": label,
                    "outcome_time": hour,
                    "available_at": hour + 6,
                }
            )
            for policy, kind, q in (
                ("static", "static", 0),
                ("continuous", "continuous_allocation", weight),
            ):
                predictions.append(
                    {
                        "split": "synthetic_holdout",
                        "episode_id": episode,
                        "icu_hour": hour,
                        "decision_time": hour,
                        "observation_cutoff": hour,
                        "policy_id": policy,
                        "policy_kind": kind,
                        "policy_version": "synthetic-v1",
                        "artifact_version": "synthetic-artifact",
                        "feature_schema_version": "synthetic-features",
                        "static_probability": 0.2,
                        "temporal_probability": 0.8,
                        "final_probability": (1 - q) * 0.2 + q * 0.8,
                        "q_temporal": q,
                    }
                )
    return predictions, outcomes


def score(frozen, **kwargs):
    predictions, outcomes = frozen
    return evaluate_policies(
        predictions,
        outcomes,
        evaluated_at=100,
        experiment_id="synthetic-test",
        reference="static",
        **kwargs,
    )


def policy(report, name):
    return next(row for row in report.policies if row.policy_id == name)


def comparison(report, loss="bernoulli_log_loss"):
    return next(row for row in report.comparisons if row.loss == loss)


def test_unequal_patient_durations_and_exact_tail_metrics(frozen):
    report = score(frozen)
    static = policy(report, "static")
    assert static.metrics["hourly_brier_score"].value == pytest.approx(0.34)
    assert static.metrics["patient_mean_brier_score"].value == pytest.approx(0.24)
    assert static.metrics["patient_cvar90_brier_score"].value == pytest.approx(0.44)
    assert static.metrics["patient_loss_max_brier_score"].value == pytest.approx(0.44)
    assert static.metrics["patient_loss_p90_brier_score"].value == pytest.approx(0.4)
    assert static.prediction_coverage == 1
    assert static.metrics["patient_mean_brier_score"].n_patients == 2
    assert static.metrics["patient_mean_brier_score"].n_observations == 4
    assert static.metrics["patient_mean_brier_score"].metric.role == "objective"
    assert static.metrics["patient_cvar90_brier_score"].metric.role == "monitoring"
    continuous = policy(report, "continuous")
    assert continuous.metrics["patient_mean_brier_score"].value == pytest.approx(0.2361)
    assert comparison(report, "brier_score").metrics[
        "paired_patient_gain_mean"
    ].value == pytest.approx(0.0039)
    # Empty steps for the one-hour patient are not replaced by artificial zeros.
    step = continuous.metrics["patient_mean_mean_absolute_change"]
    assert step.value == pytest.approx(0.1)
    assert step.n_patients == 1
    json.dumps(report.model_dump(), allow_nan=False)
    records = report.metric_records()
    assert records and all("episode_id" not in row for row in records)
    assert {row["scope"] for row in records} == {"policy", "comparison"}
    assert {row.get("subgroup") for row in records if row["scope"] == "policy"} == {
        "all",
        "eventual_positive_patients",
        "eventual_negative_patients",
        "positive_label_hours",
        "negative_label_hours",
    }
    assert static.retrospective_strata["eventual_positive_patients"][
        "patient_mean_brier_score"
    ].value == pytest.approx(0.44)
    assert static.retrospective_strata["positive_label_hours"][
        "hourly_brier_score"
    ].value == pytest.approx(0.64)


def test_patient_cluster_bootstrap_and_gain_polarity(frozen):
    config = EvaluationConfig(n_bootstrap=1000, seed=17)
    report = score(frozen, config=config)
    comp = comparison(report)
    ref, cand = policy(report, "static"), policy(report, "continuous")
    gains = np.array(
        [
            r.metrics["bernoulli_log_loss"].value
            - c.metrics["bernoulli_log_loss"].value
            for r, c in zip(ref.patients, cand.patients, strict=True)
        ]
    )
    samples = np.random.default_rng(17).choice(gains, size=(1000, 2)).mean(axis=1)
    expected = np.quantile(samples, [0.025, 0.975])
    assert comp.metrics["paired_patient_gain_ci_low"].value == pytest.approx(
        expected[0]
    )
    assert comp.metrics["paired_patient_gain_ci_high"].value == pytest.approx(
        expected[1]
    )
    assert comp.metrics["paired_patient_win_rate"].value == 0.5
    assert comp.metrics["paired_patient_loss_rate"].value == 0.5
    assert comp.metrics["paired_patient_worst_gain"].value == pytest.approx(gains.min())
    assert comp.metrics[
        "paired_patient_bottom_decile_mean_gain"
    ].value == pytest.approx(gains.min())
    reverse = evaluate_policies(
        *frozen,
        evaluated_at=100,
        experiment_id="reverse",
        reference="continuous",
        config=config,
    )
    backwards = comparison(reverse)
    assert backwards.metrics["paired_patient_gain_mean"].value == pytest.approx(
        -gains.mean()
    )
    assert backwards.metrics["paired_patient_gain_ci_low"].value == pytest.approx(
        -expected[1]
    )


def test_patient_and_hour_permutation_invariance(frozen):
    expected = score(frozen)
    predictions, outcomes = deepcopy(frozen)
    rng = np.random.default_rng(12345)
    rng.shuffle(predictions)
    rng.shuffle(outcomes)
    assert score((predictions, outcomes)) == expected


def test_patient_sign_flip_exact_and_single_class_monitoring(frozen):
    predictions, outcomes = deepcopy(frozen)
    for row in outcomes:
        row["y_true"] = 1
    report = score((predictions, outcomes))
    paired = comparison(report)
    assert paired.permutation_method == "exact_patient_sign_flip"
    assert paired.permutation_replicates == 4
    assert paired.metrics["paired_patient_sign_flip_p_value"].value == 0.5
    assert paired.metrics["paired_patient_win_rate"].value == 1
    assert policy(report, "static").metrics["auroc"].reason == "single_class_cohort"
    empty = policy(report, "static").retrospective_strata["eventual_negative_patients"][
        "patient_mean_brier_score"
    ]
    assert empty.value is None and empty.reason == "empty_retrospective_stratum"


def test_ties_and_one_patient_uncertainty(frozen):
    predictions, outcomes = deepcopy(frozen)
    predictions = [row for row in predictions if row["episode_id"] == "synthetic-a"]
    outcomes = outcomes[:1]
    for row in predictions:
        row["temporal_probability"] = row["static_probability"]
        row["final_probability"] = row["static_probability"]
    report = score((predictions, outcomes))
    comp = comparison(report)
    assert comp.metrics["paired_patient_gain_mean"].value == 0
    assert comp.metrics["paired_patient_win_rate"].value == 0
    assert comp.metrics["paired_patient_tie_rate"].value == 1
    assert (
        comp.metrics["paired_patient_gain_ci_low"].reason == "fewer_than_two_patients"
    )
    assert comp.bootstrap_replicates == 0


def test_cvar_fractional_boundary_and_ru_identity():
    assert empirical_cvar([0.3]) == pytest.approx(0.3)
    assert empirical_cvar([0.7] * 13) == pytest.approx(0.7)
    assert empirical_cvar(np.arange(20)) == pytest.approx(18.5)
    assert empirical_cvar(np.arange(11)) == pytest.approx((10 + 0.1 * 9) / 1.1)
    x = np.array([-2, -1, 0, 3, 5, 5, 8])
    objective = [eta + np.maximum(x - eta, 0).sum() / (0.1 * len(x)) for eta in x]
    assert empirical_cvar(x) == pytest.approx(min(objective))
    for bad in ([], [np.nan], [np.inf], [[1]]):
        with pytest.raises(ValueError):
            empirical_cvar(bad)


@pytest.mark.parametrize(
    "kind,q",
    [
        ("static", 0),
        ("temporal", 1),
        ("constant_mixture", 0.3),
        ("hard_switch", 1),
        ("continuous_allocation", 0.25),
        ("hybrid", 0.2),
    ],
)
def test_all_policy_families_share_contract(frozen, kind, q):
    source = frozen[0][0]
    row = {
        **source,
        "policy_kind": kind,
        "q_temporal": q,
        "final_probability": (1 - q) * 0.2 + q * 0.8,
    }
    if kind == "constant_mixture":
        row["constant_weight"] = q
    assert PredictionRecord.model_validate(row).q_temporal == q
    report = evaluate_policies(
        [row], [frozen[1][0]], evaluated_at=100, experiment_id="policy-families"
    )
    assert report.policies[0].policy_kind == kind
    assert report.policies[0].constant_weight == (
        q if kind == "constant_mixture" else None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_probability", np.nan),
        ("q_temporal", 1.1),
        ("decision_time", 0),
        ("observation_cutoff", 2),
        ("final_probability", 0.9),
        ("y_true", 1),
    ],
)
def test_invalid_or_label_bearing_predictions_rejected(frozen, field, value):
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate({**frozen[0][0], field: value})


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_prediction",
        "duplicate_prediction",
        "duplicate_outcome",
        "split",
        "artifact",
        "expert",
        "policy_version",
    ],
)
def test_alignment_and_versions_fail_closed(frozen, mutation):
    predictions, outcomes = deepcopy(frozen)
    if mutation == "missing_prediction":
        predictions.pop()
    elif mutation == "duplicate_prediction":
        predictions.append(predictions[0])
    elif mutation == "duplicate_outcome":
        outcomes.append(outcomes[0])
    elif mutation == "split":
        outcomes[0]["split"] = "other_holdout"
    elif mutation == "artifact":
        predictions[-1]["artifact_version"] = "other_artifact"
    elif mutation == "policy_version":
        predictions[-1]["policy_version"] = "other_policy"
    else:
        predictions[0].update(static_probability=0.3, final_probability=0.3)
    with pytest.raises(ValueError):
        score((predictions, outcomes))


def test_outcome_and_evaluation_clocks(frozen):
    predictions, outcomes = deepcopy(frozen)
    with pytest.raises(ValueError, match="not yet available"):
        evaluate_policies(
            predictions, outcomes, evaluated_at=3, experiment_id="too-early"
        )
    with pytest.raises(ValidationError):
        OutcomeRecord.model_validate({**outcomes[0], "available_at": 0})
    with pytest.raises(ValidationError):
        OutcomeRecord.model_validate({**outcomes[0], "y_true": 2})
    with pytest.raises(ValueError):
        score(frozen, expected_policy_ids=["static", "continuous", "missing"])


def test_feedback_maturity_and_explicit_independent_availability(frozen):
    predictions, outcomes = deepcopy(frozen)
    target = next(
        row
        for row in predictions
        if row["episode_id"] == "synthetic-b" and row["icu_hour"] == 2
    )
    target.update(feedback_hours=[1], feedback_protocol="verified_availability")
    with pytest.raises(ValueError, match="not matured"):
        score((predictions, outcomes))
    previous = next(
        row
        for row in outcomes
        if row["episode_id"] == "synthetic-b" and row["icu_hour"] == 1
    )
    previous["available_at"] = 2
    with pytest.raises(ValueError, match="cannot be called verified"):
        score((predictions, outcomes))
    previous["availability_basis"] = "verified_independent"
    assert not score((predictions, outcomes)).contains_oracle_delay_simulation
    previous["availability_basis"] = "retrospective_only"
    target["feedback_protocol"] = "oracle_delay_simulation"
    assert score((predictions, outcomes)).contains_oracle_delay_simulation
    target["feedback_hours"] = [2]
    with pytest.raises(ValidationError):
        score((predictions, outcomes))


def test_metric_prefix_is_independent_of_extra_ledger_and_inputs_not_mutated(frozen):
    before = deepcopy(frozen)
    report = score(frozen)
    outcomes = deepcopy(frozen[1])
    outcomes.append(
        {**outcomes[0], "icu_hour": 10, "outcome_time": 10, "available_at": 16}
    )
    keys = [(row["episode_id"], row["icu_hour"]) for row in frozen[1]]
    assert score((frozen[0], outcomes), expected_keys=keys) == report
    assert frozen == before


def test_gaussian_registry_namespace_unchanged(frozen):
    baseline = evaluate([0, 1], [0.2, 0.8], sigma=1)
    names = set(METRICS)
    evaluate_classification([0, 1], [0.2, 0.8])
    evaluate_classification_policies(
        *frozen, evaluated_at=100, experiment_id="namespace"
    )
    assert set(METRICS) == names
    assert evaluate([0, 1], [0.2, 0.8], sigma=1) == baseline


def test_coverage_failure_reports_matching_denominators(frozen):
    predictions, outcomes = deepcopy(frozen)
    predictions.pop()
    with pytest.raises(CoverageError) as failure:
        score((predictions, outcomes))
    assert failure.value.coverage["continuous"] == {
        "expected": 4,
        "supplied": 3,
        "matching": 3,
        "matching_fraction": 0.75,
    }
    assert failure.value.coverage["static"]["matching_fraction"] == 1
