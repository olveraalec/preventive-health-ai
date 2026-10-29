"""Aligned retrospective policy evaluation with patient-cluster inference.

Only frozen predictions and a separate outcome ledger are consumed. This module
does not predict, fit, allocate, query a database, or call an agent/LLM.
"""

from itertools import product
from typing import Literal

import numpy as np

from healthcare_ai_lab.evaluation.classification import (
    CalibrationBin,
    binary_losses,
    evaluate_classification,
)
from healthcare_ai_lab.evaluation.contracts import (
    METRIC_VERSION,
    Contract,
    EvaluationConfig,
    MetricValue,
    OutcomeRecord,
    PredictionRecord,
    metric_value,
)
from healthcare_ai_lab.evaluation.controller_diagnostics import controller_diagnostics


def empirical_cvar(values, alpha=0.9):
    """Exact empirical upper-tail RU risk, including fractional boundary mass."""
    x = np.asarray(values, dtype=float)
    if (
        x.ndim != 1
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(alpha)
        or not 0 <= alpha < 1
    ):
        raise ValueError("Invalid empirical CVaR input")
    eta = float(np.quantile(x, alpha, method="inverted_cdf"))
    return float(eta + np.maximum(x - eta, 0).sum() / ((1 - alpha) * len(x)))


class PatientMetrics(Contract):
    episode_id: str
    n_hours: int
    eventual_positive: bool
    metrics: dict[str, MetricValue]
    controller: dict[str, MetricValue]


class PolicyEvaluation(Contract):
    policy_id: str
    policy_kind: str
    policy_version: str
    artifact_version: str
    feature_schema_version: str
    constant_weight: float | None
    n_expected_patient_hours: int
    n_scored_patient_hours: int
    prediction_coverage: float
    metrics: dict[str, MetricValue]
    calibration: tuple[CalibrationBin, ...]
    patients: tuple[PatientMetrics, ...]
    retrospective_strata: dict[str, dict[str, MetricValue]]


class PairedComparison(Contract):
    reference: str
    candidate: str
    loss: str
    n_patients: int
    bootstrap_replicates: int
    seed: int
    permutation_method: str
    permutation_replicates: int
    metrics: dict[str, MetricValue]
    uncertainty_scope: str = (
        "patient_sampling_conditional_on_frozen_policies_not_tuning_bias"
    )


class EvaluationReport(Contract):
    experiment_id: str
    dataset: str
    split: str
    target_semantics: str = "shifted_challenge_label"
    time_unit: str = "episode_icu_hours"
    evaluated_at: float
    metric_version: str = METRIC_VERSION
    config: EvaluationConfig
    policies: tuple[PolicyEvaluation, ...]
    comparisons: tuple[PairedComparison, ...]
    research_only: Literal[True] = True
    clinical_use: Literal[False] = False
    availability_status: str = "declared_timestamps_checked_not_independently_verified"
    contains_oracle_delay_simulation: bool

    def metric_records(self):
        """JSON-safe long-form cohort/comparison records; no patient IDs exported."""
        rows = []
        context = {
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "split": self.split,
            "target_semantics": self.target_semantics,
            "evaluated_at": self.evaluated_at,
            "calibration_bins": self.config.calibration_bins,
        }
        for policy in self.policies:
            for subgroup, metrics in [
                ("all", policy.metrics),
                *policy.retrospective_strata.items(),
            ]:
                for result in metrics.values():
                    rows.append(
                        {
                            **context,
                            "scope": "policy",
                            "subgroup": subgroup,
                            "policy_id": policy.policy_id,
                            "policy_version": policy.policy_version,
                            "artifact_version": policy.artifact_version,
                            "feature_schema_version": policy.feature_schema_version,
                            "constant_weight": policy.constant_weight,
                            **result.model_dump(),
                        }
                    )
        for comparison in self.comparisons:
            reference = next(
                policy
                for policy in self.policies
                if policy.policy_id == comparison.reference
            )
            candidate = next(
                policy
                for policy in self.policies
                if policy.policy_id == comparison.candidate
            )
            for result in comparison.metrics.values():
                rows.append(
                    {
                        **context,
                        "scope": "comparison",
                        "reference": comparison.reference,
                        "candidate": comparison.candidate,
                        "loss": comparison.loss,
                        "reference_policy_version": reference.policy_version,
                        "candidate_policy_version": candidate.policy_version,
                        "artifact_version": reference.artifact_version,
                        "feature_schema_version": reference.feature_schema_version,
                        "seed": comparison.seed,
                        "bootstrap_replicates": comparison.bootstrap_replicates,
                        **result.model_dump(),
                    }
                )
        return rows


def _key(record):
    return record.episode_id, record.icu_hour


class CoverageError(ValueError):
    """Reject scoring while exposing counts, not patient keys, for coverage reporting."""

    def __init__(self, expected, groups, policies=None):
        expected = set(expected)
        self.coverage = {}
        for name in sorted(set(groups) | set(policies or ())):
            keys = set(groups.get(name, {}))
            self.coverage[name] = {
                "expected": len(expected),
                "supplied": len(keys),
                "matching": len(keys & expected),
                "matching_fraction": len(keys & expected) / len(expected),
            }
        super().__init__(
            "Unaligned policy coverage; no patient-hours were silently dropped"
        )


def _validate_records(
    predictions, outcomes, evaluated_at, expected_keys, expected_policy_ids
):
    predictions = [PredictionRecord.model_validate(row) for row in predictions]
    outcomes = [OutcomeRecord.model_validate(row) for row in outcomes]
    if not predictions or not outcomes:
        raise ValueError("Nonempty frozen predictions and outcome ledger are required")
    if not np.isfinite(evaluated_at):
        raise ValueError("Evaluation cutoff must be finite")
    contexts = {(row.dataset, row.split) for row in [*predictions, *outcomes]}
    if len(contexts) != 1:
        raise ValueError("Mixed datasets/splits are not a single evaluation partition")
    ledger = {_key(row): row for row in outcomes}
    if len(ledger) != len(outcomes):
        raise ValueError("Duplicate patient-hour outcomes")
    if any(row.available_at > evaluated_at for row in outcomes):
        raise ValueError("Outcome not yet available at evaluation cutoff")
    expected = list(ledger) if expected_keys is None else list(expected_keys)
    if (
        not expected
        or len(set(expected)) != len(expected)
        or not set(expected).issubset(ledger)
    ):
        raise ValueError("Invalid or missing expected patient-hour keys")
    groups = {}
    shared = {}
    for row in predictions:
        if row.decision_time > evaluated_at:
            raise ValueError("Prediction is after evaluation cutoff")
        key = _key(row)
        for hour in row.feedback_hours:
            feedback = ledger.get((row.episode_id, hour))
            if feedback is None or feedback.available_at > row.decision_time:
                raise ValueError("Feedback is missing or not matured at decision time")
            if (
                row.feedback_protocol == "verified_availability"
                and feedback.availability_basis != "verified_independent"
            ):
                raise ValueError(
                    "Retrospective challenge labels cannot be called verified online feedback"
                )
        group = groups.setdefault(row.policy_id, {})
        if key in group:
            raise ValueError("Duplicate policy patient-hour prediction")
        group[key] = row
        if key in shared:
            other = shared[key]
            if (
                row.artifact_version != other.artifact_version
                or row.feature_schema_version != other.feature_schema_version
                or row.decision_time != other.decision_time
                or row.observation_cutoff != other.observation_cutoff
                or row.static_probability != other.static_probability
                or (
                    row.temporal_probability is not None
                    and other.temporal_probability is not None
                    and row.temporal_probability != other.temporal_probability
                )
            ):
                raise ValueError(
                    "Policies disagree on frozen experts, versions, or observation clocks"
                )
        # Retain a supplied temporal value for checking later policies even if
        # another policy legitimately did not have that expert available.
        if key not in shared or row.temporal_probability is not None:
            shared[key] = row
    if expected_policy_ids is not None:
        requested = list(expected_policy_ids)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or set(requested) != set(groups)
        ):
            raise CoverageError(expected, groups, requested)
    for group in groups.values():
        if set(group) != set(expected):
            raise CoverageError(expected, groups)
        versions = {
            (
                r.policy_kind,
                r.policy_version,
                r.artifact_version,
                r.feature_schema_version,
                r.constant_weight,
            )
            for r in group.values()
        }
        if len(versions) != 1:
            raise ValueError("Policy kind/versions/constant changed inside evaluation")
    return groups, ledger, contexts.pop()


def _policy_report(policy_id, group, ledger, config):
    ordered = [group[key] for key in sorted(group)]
    y = np.array([ledger[_key(row)].y_true for row in ordered])
    p = np.array([row.final_probability for row in ordered])
    classification = evaluate_classification(y, p, config=config)
    patient_rows = []
    by_episode = {}
    for row in ordered:
        by_episode.setdefault(row.episode_id, []).append(row)
    for episode, part in sorted(by_episode.items()):
        target = [ledger[_key(row)].y_true for row in part]
        scores = evaluate_classification(
            target, [row.final_probability for row in part], config=config
        )
        patient_rows.append(
            PatientMetrics(
                episode_id=episode,
                n_hours=len(part),
                eventual_positive=any(target),
                metrics={
                    name: value.model_copy(
                        update={
                            "n_patients": 1,
                            "metric": value.metric.model_copy(
                                update={"aggregation": "within_patient"}
                            ),
                        }
                    )
                    for name, value in scores.metrics.items()
                },
                controller=controller_diagnostics(
                    [row.q_temporal for row in part],
                    static_probability=[row.static_probability for row in part],
                    temporal_probability=[row.temporal_probability for row in part],
                    emergency_fallback=[
                        row.status == "static_only_fallback" for row in part
                    ],
                    config=config,
                ),
            )
        )
    count, n = len(patient_rows), len(ordered)
    metrics = {}
    for name, result in classification.metrics.items():
        output_name = (
            f"hourly_{name}" if name in ("bernoulli_log_loss", "brier_score") else name
        )
        metrics[output_name] = result.model_copy(
            update={
                "n_patients": count,
                "metric": result.metric.model_copy(update={"metric_id": output_name}),
            }
        )
    for name in ("bernoulli_log_loss", "brier_score"):
        losses = np.array([row.metrics[name].value for row in patient_rows])
        unit = "nats" if name == "bernoulli_log_loss" else "squared_probability"
        for prefix, value in (
            ("patient_mean", losses.mean()),
            ("patient_loss_p90", np.quantile(losses, 0.9, method="linear")),
            ("patient_loss_max", losses.max()),
            ("patient_cvar90", empirical_cvar(losses)),
        ):
            metric_name = f"{prefix}_{name}"
            metrics[metric_name] = metric_value(
                metric_name,
                value,
                n=n,
                patients=count,
                direction="minimize",
                role="objective" if prefix == "patient_mean" else "monitoring",
                aggregation="patient_weighted",
                unit=unit,
                epsilon=config.epsilon if name == "bernoulli_log_loss" else None,
            )
    for name in patient_rows[0].controller:
        values = [row.controller[name] for row in patient_rows]
        valid = [value for value in values if value.value is not None]
        metric_name = f"patient_mean_{name}"
        source = values[0].metric
        metrics[metric_name] = metric_value(
            metric_name,
            np.mean([v.value for v in valid]) if valid else None,
            n=sum(v.n_observations for v in valid),
            patients=len(valid),
            controller=True,
            aggregation="patient_weighted_defined_patients",
            unit=source.unit,
            reason="no_patients_with_defined_diagnostic",
        )
    for name in ("emergency_fallback_count", "allocation_step_count"):
        metrics[name] = metric_value(
            name,
            sum(row.controller[name].value for row in patient_rows),
            n=n,
            patients=count,
            controller=True,
            aggregation="sum_of_within_patient_counts",
            unit="count",
        )
    metrics["score_clipped_probability_count"] = metric_value(
        "score_clipped_probability_count",
        np.sum((p < config.epsilon) | (p > 1 - config.epsilon)),
        n=n,
        patients=count,
        unit="count",
    )
    sample = ordered[0]
    return PolicyEvaluation(
        policy_id=policy_id,
        policy_kind=sample.policy_kind,
        policy_version=sample.policy_version,
        artifact_version=sample.artifact_version,
        feature_schema_version=sample.feature_schema_version,
        constant_weight=sample.constant_weight,
        n_expected_patient_hours=n,
        n_scored_patient_hours=n,
        prediction_coverage=1,
        metrics=metrics,
        calibration=classification.calibration,
        patients=tuple(patient_rows),
        retrospective_strata=_retrospective_strata(patient_rows, y, p, ordered, config),
    )


def _retrospective_strata(patients, y, p, ordered, config):
    """Descriptive evaluation only; eventual outcomes never become policy inputs."""
    strata = {}
    for positive in (False, True):
        selected = [row for row in patients if row.eventual_positive == positive]
        name = (
            "eventual_positive_patients" if positive else "eventual_negative_patients"
        )
        strata[name] = {}
        for loss in ("bernoulli_log_loss", "brier_score"):
            metric_name = f"patient_mean_{loss}"
            strata[name][metric_name] = metric_value(
                metric_name,
                np.mean([row.metrics[loss].value for row in selected])
                if selected
                else None,
                n=sum(row.n_hours for row in selected),
                patients=len(selected),
                direction="minimize",
                unit="nats" if loss == "bernoulli_log_loss" else "squared_probability",
                aggregation="patient_weighted_retrospective_stratum",
                reason="empty_retrospective_stratum",
                epsilon=config.epsilon if loss == "bernoulli_log_loss" else None,
            )
    log_losses = binary_losses(y, p, config.epsilon)
    for label in (0, 1):
        mask = y == label
        name = "positive_label_hours" if label else "negative_label_hours"
        strata[name] = {}
        for loss, values in (
            ("bernoulli_log_loss", log_losses),
            ("brier_score", (y - p) ** 2),
        ):
            metric_name = f"hourly_{loss}"
            strata[name][metric_name] = metric_value(
                metric_name,
                values[mask].mean() if mask.any() else None,
                n=int(mask.sum()),
                patients=len(
                    {
                        row.episode_id
                        for row, use in zip(ordered, mask, strict=True)
                        if use
                    }
                ),
                direction="minimize",
                unit="nats" if loss == "bernoulli_log_loss" else "squared_probability",
                aggregation="hour_weighted_retrospective_stratum",
                reason="empty_retrospective_stratum",
                epsilon=config.epsilon if loss == "bernoulli_log_loss" else None,
            )
    return strata


def _paired_comparison(reference, candidate, loss, config):
    ref = {row.episode_id: row for row in reference.patients}
    cand = {row.episode_id: row for row in candidate.patients}
    gains = np.array(
        [
            ref[key].metrics[loss].value - cand[key].metrics[loss].value
            for key in sorted(ref)
        ]
    )
    n, hours = len(gains), reference.n_scored_patient_hours
    rng = np.random.default_rng(config.seed)
    if n >= 2:
        means = np.empty(config.n_bootstrap)
        for start in range(0, config.n_bootstrap, 256):
            stop = min(start + 256, config.n_bootstrap)
            means[start:stop] = rng.choice(
                gains, size=(stop - start, n), replace=True
            ).mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975], method="linear")
    else:
        low, high = None, None
    statistic = abs(float(gains.mean()))
    if n <= 12:
        signs = np.array(list(product((-1, 1), repeat=n)))
        permuted = (signs * gains).mean(axis=1)
        p_value = float(np.mean(np.abs(permuted) >= statistic - 1e-15))
        method, replicates = "exact_patient_sign_flip", len(signs)
    else:
        permutation_rng = np.random.default_rng(config.seed + 1)
        count = 0
        for start in range(0, config.n_permutations, 256):
            size = min(256, config.n_permutations - start)
            signs = permutation_rng.choice([-1, 1], size=(size, n))
            count += int(
                np.sum(np.abs((signs * gains).mean(axis=1)) >= statistic - 1e-15)
            )
        p_value = float((count + 1) / (config.n_permutations + 1))
        method, replicates = (
            "monte_carlo_patient_sign_flip_plus_one",
            config.n_permutations,
        )
    values = {
        "paired_patient_gain_mean": float(gains.mean()),
        "paired_patient_gain_median": float(np.median(gains)),
        "paired_patient_win_rate": float(np.mean(gains > 0)),
        "paired_patient_tie_rate": float(np.mean(gains == 0)),
        "paired_patient_loss_rate": float(np.mean(gains < 0)),
        "paired_patient_win_count": int(np.sum(gains > 0)),
        "paired_patient_worst_gain": float(gains.min()),
        "paired_patient_gain_p10": float(np.quantile(gains, 0.1)),
        "paired_patient_bottom_decile_mean_gain": -empirical_cvar(-gains),
        "paired_patient_gain_ci_low": low,
        "paired_patient_gain_ci_high": high,
        "paired_patient_sign_flip_p_value": p_value,
    }
    metrics = {}
    for name, value in values.items():
        is_gain = "gain" in name
        unit = (
            ("nats" if loss == "bernoulli_log_loss" else "squared_probability")
            if is_gain
            else "fraction"
        )
        if name.endswith("count"):
            unit = "count"
        metrics[name] = metric_value(
            name,
            value,
            n=hours,
            patients=n,
            direction="maximize" if is_gain or "win" in name else "none",
            aggregation="paired_patient_weighted",
            unit=unit,
            reason="fewer_than_two_patients" if value is None else None,
            epsilon=config.epsilon if loss == "bernoulli_log_loss" else None,
        )
    return PairedComparison(
        reference=reference.policy_id,
        candidate=candidate.policy_id,
        loss=loss,
        n_patients=n,
        bootstrap_replicates=config.n_bootstrap if n >= 2 else 0,
        seed=config.seed,
        permutation_method=method,
        permutation_replicates=replicates,
        metrics=metrics,
    )


def evaluate_policies(
    predictions,
    outcomes,
    *,
    evaluated_at,
    experiment_id,
    reference=None,
    config=None,
    expected_keys=None,
    expected_policy_ids=None,
):
    """Require exactly the same planned patient-hours and separate matured outcomes.

    expected_keys may exclude ledger-only historical feedback rows. By default the
    entire supplied ledger defines the expected cohort. Never silently drop rows.
    """
    config = EvaluationConfig.model_validate(config if config is not None else {})
    try:
        evaluated_at = float(evaluated_at)
    except (TypeError, ValueError):
        raise ValueError("Evaluation cutoff must be a finite ICU-hour value") from None
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id is required")
    groups, ledger, (dataset, split) = _validate_records(
        predictions,
        outcomes,
        evaluated_at,
        expected_keys,
        expected_policy_ids,
    )
    policies = tuple(
        _policy_report(name, group, ledger, config)
        for name, group in sorted(groups.items())
    )
    comparisons = []
    if reference is not None:
        by_id = {policy.policy_id: policy for policy in policies}
        if reference not in by_id:
            raise ValueError("Reference policy missing")
        for candidate in policies:
            if candidate.policy_id != reference:
                for loss in ("bernoulli_log_loss", "brier_score"):
                    comparisons.append(
                        _paired_comparison(by_id[reference], candidate, loss, config)
                    )
    return EvaluationReport(
        experiment_id=experiment_id,
        dataset=dataset,
        split=split,
        evaluated_at=evaluated_at,
        config=config,
        policies=policies,
        comparisons=tuple(comparisons),
        contains_oracle_delay_simulation=any(
            row.feedback_protocol == "oracle_delay_simulation"
            for group in groups.values()
            for row in group.values()
        ),
    )
