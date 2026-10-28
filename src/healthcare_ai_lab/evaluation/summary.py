"""Typed, aggregate-only view of already recorded research evidence.

Numbers are transcribed from the versioned source, not recomputed from labels.
This synthetic study is never linked to the installed Sepsis model's performance.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from healthcare_ai_lab.evaluation.registry import SUMMARY_METRICS

Finite = Annotated[float, Field(allow_inf_nan=False)]
MetricId = Literal[
    "patient_mean_bernoulli_log_loss",
    "patient_cvar90_bernoulli_log_loss",
    "positive_patient_mean_bernoulli_log_loss",
    "paired_patient_log_loss_gain",
]


class SummaryContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecordedMetric(SummaryContract):
    metric_id: MetricId
    value: Finite
    direction: Literal["minimize", "maximize"]
    aggregation: str
    unit: Literal["nats"] = "nats"
    metric_version: Literal["1"] = "1"
    category: Literal["retrospective_label_dependent"] = "retrospective_label_dependent"


class RecordedPolicy(SummaryContract):
    policy_id: str
    metrics: list[RecordedMetric]


class RecordedComparison(SummaryContract):
    reference_policy_id: str
    candidate_policy_id: str
    metric: RecordedMetric
    ci95: tuple[Finite, Finite]
    interval_method: Literal["exploratory_paired_patient_percentile_bootstrap"] = (
        "exploratory_paired_patient_percentile_bootstrap"
    )
    bootstrap_replicates: Literal[1500] = 1500


class RecordedProvenance(SummaryContract):
    source_path: str
    source_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evidence_status: Literal["document_reported_not_reproduced_here"] = (
        "document_reported_not_reproduced_here"
    )
    precision: Literal["rounded_to_6_decimal_places_in_source"] = (
        "rounded_to_6_decimal_places_in_source"
    )
    model_artifact_sha256: None = None
    score_clipping_epsilon: None = None
    missing_provenance: tuple[str, ...] = (
        "numeric_seed_not_recorded",
        "score_clipping_not_recorded",
        "cvar_finite_sample_convention_not_recorded",
        "reference_script_not_available",
    )


class RecordedExperiment(SummaryContract):
    experiment_id: Literal["synthetic_conditional_allocation.v0.1"] = (
        "synthetic_conditional_allocation.v0.1"
    )
    dataset: Literal["synthetic_binary_sequences"] = "synthetic_binary_sequences"
    target: Literal["simulated_binary_event"] = "simulated_binary_event"
    split: Literal["held_out_test"] = "held_out_test"
    n_test_episodes: Literal[600] = 600
    provenance: RecordedProvenance
    policies: list[RecordedPolicy]
    comparison: RecordedComparison
    limitations: tuple[str, ...] = (
        "Not Sepsis model evaluation or clinical evidence",
        "Conditional policy worsened positive-patient mean loss versus constant",
        "Document-reported results; reference implementation not reproduced here",
    )


class EvaluationSummary(SummaryContract):
    schema_version: Literal["evaluation-summary.v0.6"] = "evaluation-summary.v0.6"
    category: Literal["retrospective_label_dependent"] = "retrospective_label_dependent"
    status: Literal["available_recorded_synthetic_evidence_only"] = (
        "available_recorded_synthetic_evidence_only"
    )
    installed_sepsis_model_evaluation: Literal["unavailable"] = "unavailable"
    live_labels_used: Literal[False] = False
    experiments: list[RecordedExperiment]
    unavailable_metrics: tuple[str, ...] = (
        "auroc",
        "auprc",
        "brier_score",
        "ece",
        "calibration",
        "win_rate",
        "retrospective_alert_burden",
    )
    unavailable_reason: Literal["not_recorded_in_available_versioned_result"] = (
        "not_recorded_in_available_versioned_result"
    )
    research_only: Literal[True] = True
    clinical_use: Literal[False] = False


def _metric(metric_id: MetricId, value: float) -> RecordedMetric:
    spec = SUMMARY_METRICS[metric_id]
    return RecordedMetric(
        metric_id=metric_id,
        value=value,
        direction=spec.direction,
        aggregation=spec.aggregation,
    )


def recorded_evaluation_summary() -> EvaluationSummary:
    """Return fresh typed aggregates; no file/data/model/DB/metric execution."""
    rows = (
        ("static", 0.099691, 0.513579, 0.409814),
        ("training_selected_constant", 0.096219, 0.483398, 0.392995),
        ("global_brier_moment", 0.096218, 0.483395, 0.392992),
        ("conditional_two_regime_moment", 0.089724, 0.470539, 0.394507),
        ("shuffled_context_control", 0.096188, 0.483625, 0.392895),
    )
    ids = (
        "patient_mean_bernoulli_log_loss",
        "patient_cvar90_bernoulli_log_loss",
        "positive_patient_mean_bernoulli_log_loss",
    )
    return EvaluationSummary(
        experiments=[
            RecordedExperiment(
                provenance=RecordedProvenance(
                    source_path="markdowns/SYNTHETIC_CONDITIONAL_ALLOCATION_SANITY_EXPERIMENT_v0.1.md",
                    source_sha256="da2cdc0156faadea45eec6ed59523827c8749c04a267f3fab7dbdafe8222abaf",
                ),
                policies=[
                    RecordedPolicy(
                        policy_id=row[0],
                        metrics=[
                            _metric(name, value)
                            for name, value in zip(ids, row[1:], strict=True)
                        ],
                    )
                    for row in rows
                ],
                comparison=RecordedComparison(
                    reference_policy_id="training_selected_constant",
                    candidate_policy_id="conditional_two_regime_moment",
                    metric=_metric("paired_patient_log_loss_gain", 0.006495),
                    ci95=(0.004191, 0.008847),
                ),
            )
        ]
    )
