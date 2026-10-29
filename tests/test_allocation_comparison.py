"""Synthetic allocation identities, causal replay and patient-disjoint selection."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from healthcare_ai_lab.evaluation.classification import binary_losses
from healthcare_ai_lab.experiments.allocation_comparison import (
    POLICIES,
    ExperimentConfig,
    ExperimentError,
    _evaluation_inputs,
    allocate,
    features,
    fit_allocations,
    mixture,
    normal_context,
    patient_loss,
    run_experiment,
    training_rows,
    validate_oof,
    variation_penalty,
)
from healthcare_ai_lab.experiments.allocation_sources import synthetic_source
from healthcare_ai_lab.models.artifact import META_FEATURES


@pytest.fixture(scope="module")
def source():
    return synthetic_source(n_training=36, n_evaluation=8, hours=4)


@pytest.fixture(scope="module")
def config():
    return ExperimentConfig(n_bootstrap=32, min_reliability_patients=5)


@pytest.fixture(scope="module")
def fitted(source, config):
    return fit_allocations(source.training, source.proof, config)


def test_full_experiment_reproducible_and_aggregate_only(source, config):
    result = run_experiment(source, config)
    assert result == run_experiment(source, config)
    assert set(result["summaries"]) == set(POLICIES)
    assert result["evaluation_patients"] == 8
    assert result["evaluation_hours"] == 32
    for row in result["summaries"].values():
        assert row["prediction_coverage"] == 1
        assert row["scored_patient_hours"] == 32
        assert len(row["calibration"]) == 10
        assert sum(bin["count"] for bin in row["calibration"]) == 32
    assert result["clinical_promotion"] is False
    assert result["research_only"] is True
    assert {row["reference"] for row in result["comparisons"]} == {
        "constant_mixture",
        "static",
        "hard_switch",
    }
    serialized = json.dumps(result, allow_nan=False)
    assert "synthetic-train-" not in serialized
    assert "synthetic-test-" not in serialized
    assert "episode_id" not in serialized
    assert len(result["runner_sha256"]) == 64
    assert len(result["protocol_sha256"]) == 64
    assert {
        row["reference"]
        for row in result["metric_records"]
        if row["scope"] == "comparison"
    } == {"constant_mixture", "hard_switch", "static"}


def test_fitting_roles_are_training_only_and_disjoint(source, fitted):
    roles = list(fitted.roles.values())
    assert set.union(*(set(role) for role in roles)) == set(
        source.training.context.episode_id
    )
    assert all(not a & b for i, a in enumerate(roles) for b in roles[i + 1 :])
    assert not set(source.evaluation.context.episode_id) & set.union(
        *(set(r) for r in roles)
    )
    assert features(source.training.context).columns.tolist() == META_FEATURES


def test_evaluation_labels_cannot_change_allocations_or_selection(source, config):
    original = run_experiment(source, config)
    outcomes = source.evaluation.outcomes.copy()
    outcomes["y_true"] = 1 - outcomes.y_true
    changed = run_experiment(
        replace(source, evaluation=replace(source.evaluation, outcomes=outcomes)),
        config,
    )
    for field in ("parameters", "role_counts", "reliability", "allocation_digest"):
        assert original[field] == changed[field]
    assert original["summaries"] != changed["summaries"]


def test_reliability_outcomes_do_not_refit_or_reselect_policy(source, config, fitted):
    outcomes = source.training.outcomes.copy()
    mask = outcomes.episode_id.isin(fitted.roles["reliability"])
    outcomes.loc[mask, "y_true"] = 1 - outcomes.loc[mask, "y_true"]
    altered = fit_allocations(
        replace(source.training, outcomes=outcomes), source.proof, config
    )
    assert altered.reference_weight == fitted.reference_weight
    assert altered.hard_parameters == fitted.hard_parameters
    assert altered.continuous_parameters == fitted.continuous_parameters
    np.testing.assert_array_equal(
        altered.gain_model.predict(features(source.evaluation.context)),
        fitted.gain_model.predict(features(source.evaluation.context)),
    )


def test_brier_incremental_gain_and_log_loss_convexity(source):
    frame = source.evaluation.context
    y = source.evaluation.outcomes.y_true.to_numpy()
    ps, pt = frame.p_static.to_numpy(), frame.p_temporal.to_numpy()
    q0, rho = 0.3, 0.4
    p0, delta = mixture(frame, q0), (pt - ps) * (0.8 - q0)
    p1 = mixture(frame, q0 + rho * (0.8 - q0))
    gain = (y - p0) ** 2 - (y - p1) ** 2
    np.testing.assert_allclose(
        gain, 2 * rho * (y - p0) * delta - rho**2 * delta**2, atol=1e-16
    )
    q = np.linspace(0, 1, len(frame))
    assert np.all(
        binary_losses(y, mixture(frame, q))
        <= (1 - q) * binary_losses(y, ps) + q * binary_losses(y, pt) + 1e-15
    )


def test_patient_and_hour_permutation_invariance(source, config):
    def shuffled(cohort):
        return replace(
            cohort,
            context=cohort.context.sample(frac=1, random_state=11),
            outcomes=cohort.outcomes.sample(frac=1, random_state=12),
        )

    result = run_experiment(source, config)
    permuted = run_experiment(
        replace(
            source,
            training=shuffled(source.training),
            evaluation=shuffled(source.evaluation),
        ),
        config,
    )
    assert result == permuted


@pytest.mark.parametrize(
    "bad_lineage",
    [
        "own_patient",
        "evaluation_patient",
        "one_fold",
        "missing_patient",
        "missing_witness",
    ],
)
def test_oof_leakage_rejected(source, bad_lineage):
    training = training_rows(source.training)
    mapping = dict(source.proof.fold_by_patient)
    fit_sets = dict(source.proof.fit_patients_by_fold)
    patient = next(iter(mapping))
    if bad_lineage == "own_patient":
        fit_sets[mapping[patient]] = frozenset([patient])
    elif bad_lineage == "evaluation_patient":
        fit_sets[mapping[patient]] = frozenset(
            [source.evaluation.context.episode_id.iloc[0]]
        )
    elif bad_lineage == "one_fold":
        mapping = dict.fromkeys(mapping, 0)
    elif bad_lineage == "missing_patient":
        mapping.pop(patient)
    else:
        fit_sets.pop(mapping[patient])
    with pytest.raises(ExperimentError):
        validate_oof(
            training,
            source.evaluation.context,
            replace(
                source.proof, fold_by_patient=mapping, fit_patients_by_fold=fit_sets
            ),
        )


def test_outer_patient_overlap_rejected(source, config):
    with pytest.raises(ExperimentError, match="overlap"):
        run_experiment(replace(source, evaluation=source.training), config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("icu_hour", 0),
        ("icu_hour", np.inf),
        ("p_static", np.nan),
        ("p_static", 1.1),
        ("p_temporal", np.inf),
        ("p_temporal", -0.1),
        ("HR_observed", 2),
        ("HR_age", -1),
        ("Resp_age", np.inf),
    ],
)
def test_invalid_context_rejected(source, field, value):
    frame = source.evaluation.context.astype({field: float}).copy()
    frame.loc[0, field] = value
    with pytest.raises(ExperimentError):
        normal_context(frame)


def test_label_fields_and_duplicate_context_rejected(source, fitted):
    frame = source.evaluation.context
    with pytest.raises(ExperimentError):
        allocate(fitted, frame.assign(y_true=0))
    with pytest.raises(ExperimentError):
        normal_context(pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ExperimentError):
        normal_context(frame.iloc[:0])


def test_numeric_hour_normalization_precedes_order_and_uniqueness(source):
    frame = source.evaluation.context.iloc[:4].copy()
    frame["icu_hour"] = ["10", "2", "1", "3"]
    assert normal_context(frame).icu_hour.tolist() == [1, 2, 3, 10]
    frame["icu_hour"] = [1, "1", 2, 3]
    with pytest.raises(ExperimentError, match="Duplicate"):
        normal_context(frame)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "invalid"])
def test_training_outcome_alignment_rejected(source, mutation):
    ledger = source.training.outcomes.copy()
    if mutation == "missing":
        ledger = ledger.iloc[1:]
    elif mutation == "duplicate":
        ledger = pd.concat([ledger, ledger.iloc[:1]])
    else:
        ledger.loc[0, "y_true"] = 2
    with pytest.raises(ExperimentError):
        training_rows(replace(source.training, outcomes=ledger))


def test_prefix_causality_and_patient_state_isolation(source, fitted):
    context, weights, _ = allocate(fitted, source.evaluation.context)
    prefix, prefix_weights, _ = allocate(fitted, context[context.icu_hour <= 2])
    mask = context.icu_hour.le(2).to_numpy()
    assert len(prefix) == mask.sum()
    for name in POLICIES:
        np.testing.assert_allclose(
            prefix_weights[name], weights[name][mask], rtol=0, atol=1e-15
        )
    # A fresh replay of any one patient must equal their trajectory in the batch.
    for episode in context.episode_id.unique():
        selected = context.episode_id.eq(episode).to_numpy()
        _, independent, _ = allocate(fitted, context[selected])
        for name in POLICIES:
            np.testing.assert_allclose(
                independent[name], weights[name][selected], rtol=0, atol=1e-15
            )
    _, replayed, _ = allocate(fitted, source.evaluation.context)
    for name in POLICIES:
        np.testing.assert_array_equal(replayed[name], weights[name])


def test_feasibility_endpoints_and_bounded_step(source, fitted):
    context, weights, _ = allocate(fitted, source.evaluation.context)
    np.testing.assert_array_equal(mixture(context, weights["static"]), context.p_static)
    np.testing.assert_array_equal(
        mixture(context, weights["temporal"]), context.p_temporal
    )
    assert set(weights["hard_switch"]) <= {0, 1}
    for name, q in weights.items():
        assert np.isfinite(q).all() and np.all((q >= 0) & (q <= 1))
        if name in ("bounded_anticipatory", "continuous_step_0_1"):
            delta = pd.Series(q).groupby(context.episode_id).diff().dropna()
            assert delta.abs().max() <= 0.1 + 1e-15
    assert weights["bounded_anticipatory"].max() <= 0.25


@pytest.mark.parametrize("q", [[[0.5]], [0, 1], np.nan, 1.1, -0.1])
def test_mixture_rejects_invalid_weights(source, q):
    with pytest.raises(ExperimentError):
        mixture(source.evaluation.context, q)


def test_agreeing_experts_do_not_force_adaptation(source, config):
    context = source.training.context.copy()
    context["p_temporal"] = context.p_static
    fitted = fit_allocations(
        replace(source.training, context=context), source.proof, config
    )
    assert fitted.reference_weight == 0
    assert fitted.hard_parameters["mode"] == "static"
    assert all(p["exposure"] == 0 for p in fitted.continuous_parameters.values())
    assert fitted.reliability["enabled"] is False
    for q in (0, 0.15, 0.5, 1):
        np.testing.assert_allclose(mixture(context, q), context.p_static, atol=1e-16)


def test_patient_weighted_loss_and_variation_identity(source):
    frame = normal_context(source.evaluation.context).iloc[:5].copy()
    frame["y_true"] = [0, 0, 1, 0, 1]
    losses = binary_losses(frame.y_true.to_numpy(), frame.p_static.to_numpy())
    assert patient_loss(frame, frame.p_static.to_numpy()) == pytest.approx(
        (losses[:4].mean() + losses[4]) / 2
    )
    assert variation_penalty(frame, [0, 0.1, 0.3, 0.6, 1]) == pytest.approx(
        (0.01 + 0.04 + 0.09) / 3 / 2
    )


def test_insufficient_support_abstains(source, config):
    fitted = fit_allocations(
        source.training,
        source.proof,
        replace_config(config, min_reliability_patients=100),
    )
    assert not fitted.reliability["enabled"]
    assert fitted.reliability["reason"] == "insufficient_patient_support"
    _, weights, _ = allocate(fitted, source.evaluation.context)
    np.testing.assert_array_equal(weights["continuous"], weights["constant_mixture"])


def replace_config(config, **kwargs):
    return ExperimentConfig(**{**config.model_dump(), **kwargs})


def test_secondary_failure_overrides_step_and_retains_constant_metadata(source, fitted):
    # Construct a high previous allocation so the immediate drop violates the
    # ordinary 0.1 step cap deliberately, as required by emergency precedence.
    class HighGain:
        def predict(self, matrix):
            return np.full(len(matrix), 10.0)

    selected = replace(
        fitted,
        reference_weight=0.8,
        gain_model=HighGain(),
        baseline_gain_model=HighGain(),
        reliability={"enabled": True},
        continuous_parameters={
            key: {"exposure": 1, "threshold": 0, "temperature": 0.02}
            for key in (0.0, 0.01, 0.1)
        },
    )
    frame = normal_context(source.evaluation.context).iloc[:4].copy()
    frame.loc[1, "p_temporal"] = np.nan
    context, weights, reasons = allocate(selected, frame)
    assert weights["continuous_step_0_1"][0] > 0.7
    assert weights["continuous_step_0_1"][1] == 0
    for name in POLICIES:
        assert weights[name][1] == 0
        assert mixture(context, weights[name])[1] == context.p_static.iloc[1]
        if name != "static":
            assert reasons[name][1] == "secondary_prediction_unavailable"
    frame["p_temporal"] = np.nan
    context, weights, reasons = allocate(selected, frame)
    predictions, _, _ = _evaluation_inputs(source, context, weights, reasons, 0.8)
    assert all(
        row["constant_weight"] == 0.8
        for row in predictions
        if row["policy_id"] == "constant_mixture"
    )
    with pytest.raises(ExperimentError):
        mixture(context, 0.1)


@pytest.mark.parametrize("failure", ["raise", "shape", "nan", "inf"])
def test_optional_controller_failure_returns_static(source, fitted, failure):
    class FailedGain:
        def predict(self, matrix):
            if failure == "raise":
                raise RuntimeError("synthetic failure")
            return np.full(
                (len(matrix), 1) if failure == "shape" else len(matrix),
                np.inf if failure == "inf" else np.nan,
            )

    selected = replace(
        fitted,
        gain_model=FailedGain(),
        baseline_gain_model=FailedGain(),
        hard_parameters={"mode": "threshold", "threshold": 0},
        reliability={"enabled": True},
        continuous_parameters={
            key: {"exposure": 1, "threshold": 0, "temperature": 0.02}
            for key in (0.0, 0.01, 0.1)
        },
    )
    _, weights, reasons = allocate(selected, source.evaluation.context)
    for name in set(POLICIES) - {"temporal", "static", "constant_mixture"}:
        assert np.all(weights[name] == 0)
        assert set(reasons[name]) == {"controller_signal_unavailable"}
