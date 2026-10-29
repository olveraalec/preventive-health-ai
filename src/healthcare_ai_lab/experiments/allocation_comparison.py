"""Frozen-expert, patient-disjoint allocation experiment. No base-model fitting.

The policy search and reliability rule are documented before evaluation in
docs/DISCRETE_CONTINUOUS_ALLOCATION_PROTOCOL_v1.md. Nothing here changes serving.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.controllers.adaptive import ControllerConfig, adaptive_allocation
from healthcare_ai_lab.evaluation.classification import binary_losses
from healthcare_ai_lab.evaluation.contracts import Contract, EvaluationConfig
from healthcare_ai_lab.evaluation.registry import evaluate_classification_policies
from healthcare_ai_lab.models.artifact import META_FEATURES, software_versions
from healthcare_ai_lab.models.sepsis_inference import static_entropy

CONTEXT_COLUMNS = (
    "episode_id",
    "icu_hour",
    "p_static",
    "p_temporal",
    "HR_observed",
    "Resp_observed",
    "Temp_observed",
    "HR_age",
    "Resp_age",
)
TEMPERATURES = (0.005, 0.02, 0.1)
EXPOSURES = (0.5, 1.0)
VARIATION_PENALTIES = (0.0, 0.01, 0.1)
POLICIES = (
    "static",
    "temporal",
    "constant_mixture",
    "hard_switch",
    "continuous",
    "bounded_anticipatory",
    "continuous_ungated",
    "continuous_step_0_1",
    "continuous_penalty_0_01",
    "continuous_penalty_0_1",
)


class ExperimentError(ValueError):
    """Static, privacy-safe diagnostics; never include source rows or identifiers."""


class ExperimentConfig(Contract):
    seed: int = Field(default=12345, ge=0, strict=True)
    n_bootstrap: int = Field(default=2000, ge=1, strict=True)
    min_reliability_patients: int = Field(default=20, ge=2, strict=True)


@dataclass(frozen=True)
class FrozenCohort:
    context: pd.DataFrame
    outcomes: pd.DataFrame
    split: str
    artifact_version: str
    feature_version: str


@dataclass(frozen=True)
class OOFProof:
    fold_by_patient: dict[str, int]
    fit_patients_by_fold: dict[int, frozenset[str]]
    status: str


@dataclass(frozen=True)
class ExperimentSource:
    training: FrozenCohort
    evaluation: FrozenCohort
    proof: OOFProof
    metadata: dict
    existing_gain_model: object | None = None


def normal_context(frame, *, training=False):
    if (
        set(frame.columns) != set(CONTEXT_COLUMNS)
        or not frame.columns.is_unique
        or frame.empty
    ):
        raise ExperimentError(
            "Inference context must contain only the declared decision-time fields"
        )
    data = frame.copy()
    if (
        data.episode_id.isna().any()
        or not data.episode_id.map(lambda x: isinstance(x, str) and bool(x)).all()
    ):
        raise ExperimentError("Invalid episode keys")
    for name in CONTEXT_COLUMNS[1:]:
        data[name] = pd.to_numeric(data[name], errors="raise").astype(float)
    if data.duplicated(["episode_id", "icu_hour"]).any():
        raise ExperimentError("Duplicate patient-hour forecasts")
    data = data.sort_values(["episode_id", "icu_hour"]).reset_index(drop=True)
    hours = data.icu_hour.to_numpy()
    if not np.isfinite(hours).all() or np.any(hours <= 0):
        raise ExperimentError("Invalid ICU hours")
    static = data.p_static.to_numpy()
    temporal = data.p_temporal.to_numpy()
    if not np.isfinite(static).all() or np.any((static < 0) | (static > 1)):
        raise ExperimentError("Static predictor unavailable; no probability fabricated")
    if np.isinf(temporal).any() or np.any((temporal < 0) | (temporal > 1)):
        raise ExperimentError("Invalid temporal probabilities")
    if training and not np.isfinite(temporal).all():
        raise ExperimentError("OOF training requires both frozen predictions")
    if np.isinf(data[list(CONTEXT_COLUMNS[4:])].to_numpy()).any():
        raise ExperimentError("Infinite controller context")
    for name in ("HR_observed", "Resp_observed", "Temp_observed"):
        valid = data[name].dropna()
        if not valid.isin([0, 1]).all():
            raise ExperimentError("Invalid availability indicator")
    for name in ("HR_age", "Resp_age"):
        if (data[name].dropna() < 0).any():
            raise ExperimentError("Negative measurement age")
    return data


def training_rows(cohort):
    context = normal_context(cohort.context, training=True)
    outcomes = cohort.outcomes
    if (
        set(outcomes) != {"episode_id", "icu_hour", "y_true"}
        or outcomes.duplicated(["episode_id", "icu_hour"]).any()
    ):
        raise ExperimentError("Invalid outcome ledger")
    if not outcomes.y_true.isin([0, 1]).all():
        raise ExperimentError("Invalid binary outcomes")
    merged = context.merge(
        outcomes,
        on=["episode_id", "icu_hour"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ExperimentError("Outcome/forecast key mismatch")
    return merged.drop(columns="_merge")


def validate_oof(training, evaluation_context, proof):
    train_ids = set(training.episode_id)
    evaluation_ids = set(evaluation_context.episode_id)
    if train_ids & evaluation_ids or set(proof.fold_by_patient) != train_ids:
        raise ExperimentError("Evaluation overlap or incomplete OOF lineage")
    if len(set(proof.fold_by_patient.values())) < 2:
        raise ExperimentError("OOF fitting requires multiple patient folds")
    for patient, fold in proof.fold_by_patient.items():
        if fold not in proof.fit_patients_by_fold:
            raise ExperimentError("Missing base-model fit-fold witness")
        fitted = proof.fit_patients_by_fold[fold]
        if patient in fitted or not set(fitted).issubset(train_ids):
            raise ExperimentError(
                "OOF expert fitted its predicted patient or an evaluation patient"
            )


def patient_weights(frame):
    counts = frame.groupby("episode_id").episode_id.transform("size").to_numpy()
    return 1.0 / (frame.episode_id.nunique() * counts)


def patient_loss(frame, probability):
    return float(
        patient_weights(frame) @ binary_losses(frame.y_true.to_numpy(), probability)
    )


def variation_penalty(frame, weights):
    q = pd.Series(weights, index=frame.index)
    delta = q.groupby(frame.episode_id).diff().fillna(0).to_numpy()
    counts = frame.groupby("episode_id").episode_id.transform("size").to_numpy()
    return float(
        np.sum(delta**2 / np.maximum(counts - 1, 1)) / frame.episode_id.nunique()
    )


def features(frame):
    matrix = frame[
        ["HR_observed", "Resp_observed", "Temp_observed", "HR_age", "Resp_age"]
    ].copy()
    matrix["ICULOS"] = frame.icu_hour.to_numpy()
    matrix["entropy_static"] = static_entropy(frame.p_static.to_numpy())
    return matrix[META_FEATURES]


def fit_gain(frame):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )
    gain = binary_losses(
        frame.y_true.to_numpy(), frame.p_static.to_numpy()
    ) - binary_losses(frame.y_true.to_numpy(), frame.p_temporal.to_numpy())
    weights = patient_weights(frame)
    weights /= weights.mean()
    model.fit(features(frame), gain, ridge__sample_weight=weights)
    return model


def mixture(frame, weights):
    # Missing temporal predictions never enter the arithmetic, even at q=0.
    q = np.asarray(weights)
    if q.ndim > 1 or (q.ndim == 1 and len(q) != len(frame)):
        raise ExperimentError(
            "Allocation shape must be scalar or one weight per forecast"
        )
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ExperimentError("Invalid proposed allocation")
    if np.any(frame.p_temporal.isna().to_numpy() & (q != 0)):
        raise ExperimentError(
            "Missing secondary predictor requires static-only allocation"
        )
    temporal = frame.p_temporal.fillna(frame.p_static).to_numpy()
    return (1 - q) * frame.p_static.to_numpy() + q * temporal


def soft_weights(gain, reference, parameters):
    exposure = parameters["exposure"]
    if exposure == 0:
        return np.full(len(gain), reference, dtype=float)
    proposal = expit((gain - parameters["threshold"]) / parameters["temperature"])
    return reference + exposure * (proposal - reference)


def hard_weights(gain, parameters):
    if parameters["mode"] in ("static", "temporal"):
        return np.full(len(gain), parameters["mode"] == "temporal", dtype=float)
    return np.where(np.isfinite(gain), gain > parameters["threshold"], np.nan).astype(
        float
    )


def _selection(frame, gain, reference, thresholds, penalty):
    candidates = [{"exposure": 0.0, "temperature": 0.02, "threshold": 0.0}]
    candidates += [
        {
            "exposure": exposure,
            "temperature": temperature,
            "threshold": float(threshold),
        }
        for exposure in EXPOSURES
        for temperature in TEMPERATURES
        for threshold in thresholds
    ]
    scored = []
    for index, candidate in enumerate(candidates):
        q = soft_weights(gain, reference, candidate)
        score = patient_loss(frame, mixture(frame, q)) + penalty * variation_penalty(
            frame, q
        )
        scored.append((score, index))
    best = min(score for score, _ in scored)
    winner = next(index for score, index in scored if score <= best + 1e-12)
    return candidates[winner], len(candidates)


@dataclass(frozen=True)
class FittedAllocation:
    gain_model: object
    baseline_gain_model: object
    reference_weight: float
    hard_parameters: dict
    continuous_parameters: dict[float, dict]
    reliability: dict
    roles: dict[str, frozenset[str]]
    search_budget: dict


def fit_allocations(training_cohort, proof, config=None, *, existing_gain_model=None):
    """Only historical training OOF labels enter fitting, selection and abstention."""
    config = config or ExperimentConfig()
    training = training_rows(training_cohort)
    validate_oof(training, training.iloc[:0], proof)
    ids = np.array(sorted(training.episode_id.unique()))
    if len(ids) < 10:
        raise ExperimentError("At least ten training patients are required")
    ids = np.random.default_rng(config.seed).permutation(ids)
    a, b = int(0.6 * len(ids)), int(0.8 * len(ids))
    roles = {
        "fit": frozenset(ids[:a]),
        "select": frozenset(ids[a:b]),
        "reliability": frozenset(ids[b:]),
    }
    partitions = {
        name: training[training.episode_id.isin(members)].copy()
        for name, members in roles.items()
    }
    fit, select, check = (partitions[name] for name in ("fit", "select", "reliability"))
    grid = np.linspace(0, 1, 101)
    constant_scores = [patient_loss(fit, mixture(fit, q)) for q in grid]
    best_constant = min(constant_scores)
    reference = float(
        next(
            q
            for q, score in zip(grid, constant_scores, strict=True)
            if score <= best_constant + 1e-12
        )
    )
    gain_model = fit_gain(fit)
    fit_signal = gain_model.predict(features(fit))
    select_signal = gain_model.predict(features(select))
    thresholds = np.unique(np.r_[0.0, np.quantile(fit_signal, [0, 0.25, 0.5, 0.75, 1])])
    hard_candidates = [
        {"mode": "static", "threshold": None},
        {"mode": "temporal", "threshold": None},
    ]
    hard_candidates += [
        {"mode": "threshold", "threshold": float(t)} for t in thresholds
    ]
    hard_scores = [
        patient_loss(select, mixture(select, hard_weights(select_signal, p)))
        for p in hard_candidates
    ]
    best_hard = min(hard_scores)
    hard_index = next(
        i for i, score in enumerate(hard_scores) if score <= best_hard + 1e-12
    )
    continuous = {}
    for penalty in VARIATION_PENALTIES:
        continuous[penalty], soft_budget = _selection(
            select, select_signal, reference, thresholds, penalty
        )
    check_q = soft_weights(
        gain_model.predict(features(check)), reference, continuous[0.0]
    )
    row_gains = binary_losses(
        check.y_true.to_numpy(), mixture(check, reference)
    ) - binary_losses(check.y_true.to_numpy(), mixture(check, check_q))
    gains = (
        pd.Series(row_gains, index=check.index)
        .groupby(check.episode_id)
        .mean()
        .sort_index()
        .to_numpy()
    )
    boot = (
        np.random.default_rng(config.seed)
        .choice(gains, size=(config.n_bootstrap, len(gains)))
        .mean(axis=1)
    )
    lower, upper = map(float, np.quantile(boot, [0.025, 0.975]))
    supported = len(gains) >= config.min_reliability_patients
    enabled = bool(supported and lower > 0)
    reliability = {
        "enabled": enabled,
        "check_patients": len(gains),
        "mean_gain_vs_constant": float(gains.mean()),
        "ci_low": lower,
        "ci_high": upper,
        "reason": "positive_training_check_bound"
        if enabled
        else (
            "insufficient_patient_support"
            if not supported
            else "no_positive_training_check_bound"
        ),
        "calibrated_probability": False,
    }
    baseline = (
        fit_gain(training) if existing_gain_model is None else existing_gain_model
    )
    return FittedAllocation(
        gain_model,
        baseline,
        reference,
        hard_candidates[hard_index],
        continuous,
        reliability,
        roles,
        {
            "constant": 101,
            "hard": len(hard_candidates),
            "continuous_per_penalty": soft_budget,
            "variation_penalties": list(VARIATION_PENALTIES),
        },
    )


def _safe_gain(model, context):
    try:
        values = np.asarray(model.predict(features(context)), dtype=float)
        if values.shape != (len(context),):
            return np.full(len(context), np.nan)
        return np.where(np.isfinite(values), values, np.nan)
    except Exception:  # noqa: BLE001 -- optional experimental controller failure
        return np.full(len(context), np.nan)


def allocate(fitted, frame):
    """Label-free, causal replay. Returns arrays only; writes no patient records."""
    context = normal_context(frame)
    gain = _safe_gain(fitted.gain_model, context)
    baseline_gain = _safe_gain(fitted.baseline_gain_model, context)
    raw = soft_weights(gain, fitted.reference_weight, fitted.continuous_parameters[0.0])
    proposals = {
        "static": np.zeros(len(context)),
        "temporal": np.ones(len(context)),
        "constant_mixture": np.full(len(context), fitted.reference_weight),
        "hard_switch": hard_weights(gain, fitted.hard_parameters),
        "continuous": raw
        if fitted.reliability["enabled"]
        else np.full(len(context), fitted.reference_weight),
        "bounded_anticipatory": expit(baseline_gain / 0.01),
        "continuous_ungated": raw,
        "continuous_step_0_1": raw,
        "continuous_penalty_0_01": soft_weights(
            gain, fitted.reference_weight, fitted.continuous_parameters[0.01]
        ),
        "continuous_penalty_0_1": soft_weights(
            gain, fitted.reference_weight, fitted.continuous_parameters[0.1]
        ),
    }
    weights, reasons = {}, {}
    for policy in POLICIES:
        output, failures, previous, previous_episode = [], [], None, None
        for index, row in enumerate(context.itertuples(index=False)):
            if row.episode_id != previous_episode:
                previous = None
            proposed = proposals[policy][index]
            failure = None
            if policy != "static" and not np.isfinite(row.p_temporal):
                q, failure = 0.0, "secondary_prediction_unavailable"
            elif not np.isfinite(proposed):
                q, failure = 0.0, "controller_signal_unavailable"
            elif policy in ("bounded_anticipatory", "continuous_step_0_1"):
                cfg = (
                    ControllerConfig()
                    if policy == "bounded_anticipatory"
                    else ControllerConfig(
                        fallback_weight=fitted.reference_weight,
                        max_deviation=1.0,
                        max_step=0.1,
                    )
                )
                q = adaptive_allocation(
                    float(proposed),
                    0.0,
                    0.25 if policy == "bounded_anticipatory" else 1.0,
                    1.0,
                    False,
                    config=cfg,
                    previous_q=previous,
                )
            else:
                q = float(proposed)
            output.append(q)
            failures.append(failure)
            previous, previous_episode = q, row.episode_id
        weights[policy], reasons[policy] = np.asarray(output), tuple(failures)
    return context, weights, reasons


def _evaluation_inputs(source, context, weights, reasons, configured_reference):
    cutoff = float(context.icu_hour.max() + 1)
    predictions = []
    kinds = {
        "static": "static",
        "temporal": "temporal",
        "constant_mixture": "constant_mixture",
        "hard_switch": "hard_switch",
    }
    rows = list(context.itertuples(index=False))
    for policy, q in weights.items():
        probabilities = mixture(context, q)
        for index, row in enumerate(rows):
            reason = reasons[policy][index]
            predictions.append(
                {
                    "split": source.evaluation.split,
                    "episode_id": row.episode_id,
                    "icu_hour": row.icu_hour,
                    "decision_time": row.icu_hour,
                    "observation_cutoff": row.icu_hour,
                    "policy_id": policy,
                    "policy_kind": kinds.get(policy, "continuous_allocation"),
                    "policy_version": "discrete-continuous-v1",
                    "artifact_version": source.evaluation.artifact_version,
                    "feature_schema_version": source.evaluation.feature_version,
                    "static_probability": row.p_static,
                    "temporal_probability": row.p_temporal
                    if np.isfinite(row.p_temporal)
                    else None,
                    "final_probability": float(probabilities[index]),
                    "q_temporal": float(q[index]),
                    "status": "static_only_fallback" if reason else "ok",
                    "fallback_reason": reason,
                    "constant_weight": float(q[index])
                    if policy == "constant_mixture" and not reason
                    else None,
                }
            )
    # Preserve the configured constant even if an emergency overrides it.
    for row in predictions:
        if row["policy_id"] == "constant_mixture":
            row["constant_weight"] = configured_reference
    outcomes = [
        {
            **row,
            "split": source.evaluation.split,
            "outcome_time": row["icu_hour"],
            "available_at": cutoff,
            "availability_basis": "retrospective_only",
        }
        for row in source.evaluation.outcomes.to_dict("records")
    ]
    return predictions, outcomes, cutoff


def run_experiment(source, config=None):
    config = config or ExperimentConfig()
    training = training_rows(source.training)
    evaluation_context = normal_context(source.evaluation.context)
    validate_oof(training, evaluation_context, source.proof)
    fitted = fit_allocations(
        source.training,
        source.proof,
        config,
        existing_gain_model=source.existing_gain_model,
    )
    context, weights, reasons = allocate(fitted, evaluation_context)
    allocation_digest = hashlib.sha256(
        b"".join(weights[name].tobytes() for name in POLICIES)
    ).hexdigest()
    # Allocations are frozen before the evaluation ledger is joined or scored.
    predictions, outcomes, cutoff = _evaluation_inputs(
        source, context, weights, reasons, fitted.reference_weight
    )
    metrics_config = EvaluationConfig(seed=config.seed, n_bootstrap=config.n_bootstrap)
    report = evaluate_classification_policies(
        predictions,
        outcomes,
        evaluated_at=cutoff,
        experiment_id="discrete-continuous-v1",
        reference="constant_mixture",
        config=metrics_config,
        expected_policy_ids=list(POLICIES),
    )
    comparisons = list(report.comparisons)
    metric_records = report.metric_records()
    for reference in ("hard_switch", "static"):
        subset = [
            row for row in predictions if row["policy_id"] in (reference, "continuous")
        ]
        paired = evaluate_classification_policies(
            subset,
            outcomes,
            evaluated_at=cutoff,
            experiment_id="discrete-continuous-v1",
            reference=reference,
            config=metrics_config,
        )
        comparisons.extend(paired.comparisons)
        metric_records.extend(
            row for row in paired.metric_records() if row["scope"] == "comparison"
        )
    summaries = {}
    for policy in report.policies:
        values = {name: result.value for name, result in policy.metrics.items()}
        values["calibration"] = [row.model_dump() for row in policy.calibration]
        values["prediction_coverage"] = policy.prediction_coverage
        values["scored_patient_hours"] = policy.n_scored_patient_hours
        values["eventual_positive_patient_log_loss"] = policy.retrospective_strata[
            "eventual_positive_patients"
        ]["patient_mean_bernoulli_log_loss"].value
        values["positive_label_hour_log_loss"] = policy.retrospective_strata[
            "positive_label_hours"
        ]["hourly_bernoulli_log_loss"].value
        values["fallback_reasons"] = {
            reason: reasons[policy.policy_id].count(reason)
            for reason in sorted(set(reasons[policy.policy_id]) - {None})
        }
        summaries[policy.policy_id] = values
    checks = []
    for reference in ("constant_mixture", "hard_switch"):
        comp = next(
            row
            for row in comparisons
            if row.reference == reference
            and row.candidate == "continuous"
            and row.loss == "bernoulli_log_loss"
        )
        low = comp.metrics["paired_patient_gain_ci_low"].value
        ref, candidate = summaries[reference], summaries["continuous"]
        positive = candidate["eventual_positive_patient_log_loss"]
        ref_positive = ref["eventual_positive_patient_log_loss"]
        checks.append(
            bool(
                low is not None
                and low > 0
                and candidate["patient_cvar90_bernoulli_log_loss"]
                <= ref["patient_cvar90_bernoulli_log_loss"]
                and positive is not None
                and ref_positive is not None
                and positive <= ref_positive
            )
        )
    support = all(checks) and source.metadata.get("fresh_holdout", False)
    comparison_rows = []
    for comp in comparisons:
        comparison_rows.append(
            {
                "reference": comp.reference,
                "candidate": comp.candidate,
                "loss": comp.loss,
                **{name: value.value for name, value in comp.metrics.items()},
                "n_patients": comp.n_patients,
                "bootstrap_replicates": comp.bootstrap_replicates,
                "seed": comp.seed,
                "permutation_method": comp.permutation_method,
                "permutation_replicates": comp.permutation_replicates,
                "uncertainty_scope": comp.uncertainty_scope,
            }
        )
    counts = {
        name: {"patients": len(ids), "hours": int(training.episode_id.isin(ids).sum())}
        for name, ids in fitted.roles.items()
    }
    return {
        "experiment_version": "discrete-continuous-v1",
        "config": config.model_dump(),
        "source": source.metadata,
        "oof_status": source.proof.status,
        "role_counts": counts,
        "evaluation_patients": context.episode_id.nunique(),
        "evaluation_hours": len(context),
        "parameters": {
            "constant_weight": fitted.reference_weight,
            "hard": fitted.hard_parameters,
            "continuous": {str(k): v for k, v in fitted.continuous_parameters.items()},
        },
        "search_budget": fitted.search_budget,
        "reliability": fitted.reliability,
        "allocation_digest": allocation_digest,
        "summaries": summaries,
        "comparisons": comparison_rows,
        "metric_records": metric_records,
        "software_versions": software_versions(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[3]
                / "docs/DISCRETE_CONTINUOUS_ALLOCATION_PROTOCOL_v1.md"
            ).read_bytes()
        ).hexdigest(),
        "synthetic_support_for_adaptation": bool(support),
        "clinical_promotion": False,
        "outcome_availability": "offline_replay_annotation_not_observed_online_time",
        "research_only": True,
        "clinical_use": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("synthetic", "local-validation"), default="synthetic"
    )
    parser.add_argument(
        "--scenario",
        choices=("informative", "uninformative", "shifted"),
        default="informative",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--train-patients", type=int, default=900)
    parser.add_argument("--evaluation-patients", type=int, default=300)
    parser.add_argument("--hours", type=int, default=12)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from healthcare_ai_lab.experiments.allocation_sources import (
        local_validation_source,
        synthetic_source,
    )

    try:
        config = ExperimentConfig(seed=args.seed, n_bootstrap=args.bootstrap)
        source = (
            local_validation_source(args.root)
            if args.source == "local-validation"
            else synthetic_source(
                args.seed,
                args.train_patients,
                args.evaluation_patients,
                args.hours,
                args.scenario,
            )
        )
        result = run_experiment(source, config)
        if args.output:
            if args.output.suffix != ".json":
                raise ExperimentError("Aggregate output must be a new JSON file")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
        concise = {
            key: value
            for key, value in result.items()
            if key not in ("metric_records", "summaries")
        }
        selected = (
            "patient_mean_bernoulli_log_loss",
            "patient_mean_brier_score",
            "ece_binned",
            "patient_cvar90_bernoulli_log_loss",
            "eventual_positive_patient_log_loss",
            "patient_mean_mean_absolute_change",
            "patient_mean_mean_squared_change",
        )
        concise["summaries"] = {
            name: {key: row[key] for key in selected}
            for name, row in result["summaries"].items()
        }
        concise["comparisons"] = [
            row
            for row in result["comparisons"]
            if row["candidate"] == "continuous" and row["loss"] == "bernoulli_log_loss"
        ]
        print(json.dumps(concise, indent=2, allow_nan=False))
    except ExperimentError as exc:
        raise SystemExit(f"Experiment refused: {exc}") from None
    except Exception as exc:  # noqa: BLE001 -- never print patient-bearing library tracebacks
        raise SystemExit(
            f"Experiment failed ({type(exc).__name__}); no source records displayed"
        ) from None


if __name__ == "__main__":
    main()
