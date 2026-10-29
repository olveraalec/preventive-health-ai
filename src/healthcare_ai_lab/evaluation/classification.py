"""Additive Bernoulli metrics; never register these in the Gaussian METRICS map."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import xlogy
from sklearn.metrics import average_precision_score, roc_auc_score

from healthcare_ai_lab.evaluation.contracts import (
    Contract,
    EvaluationConfig,
    MetricDefinition,
    MetricValue,
)


@dataclass(frozen=True)
class ClassificationMetricSpec:
    definition: MetricDefinition
    function: Callable
    requires_two_classes: bool = False


CLASSIFICATION_METRICS: dict[str, ClassificationMetricSpec] = {}


def register_classification(
    name,
    *,
    direction="none",
    role="monitoring",
    unit="unitless",
    two_classes=False,
    log_score=False,
):
    def decorator(function):
        if name in CLASSIFICATION_METRICS:
            raise ValueError("Duplicate classification metric")
        CLASSIFICATION_METRICS[name] = ClassificationMetricSpec(
            MetricDefinition(
                metric_id=name,
                direction=direction,
                role=role,
                unit=unit,
                aggregation="hour_weighted",
                probability_clip=1e-10 if log_score else None,
            ),
            function,
            two_classes,
        )
        return function

    return decorator


def validate_binary(y, probability):
    y, p = np.asarray(y, dtype=float), np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or not len(y) or y.shape != p.shape:
        raise ValueError("Expected aligned nonempty 1D binary arrays")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Outcomes must be observed binary labels")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite in [0,1]")
    return y, p


def binary_losses(y, probability, epsilon=1e-10):
    y, p = validate_binary(y, probability)
    if not np.isfinite(epsilon) or not 0 < epsilon < 0.5 or 1 - epsilon == 1:
        raise ValueError("Invalid score clipping epsilon")
    clipped = np.clip(p, epsilon, 1 - epsilon)
    return -xlogy(y, clipped) - xlogy(1 - y, 1 - clipped)


class CalibrationBin(Contract):
    lower: float
    upper: float
    upper_inclusive: bool
    count: int
    mean_probability: float | None
    observed_fraction: float | None


def calibration_bins(y, probability, n_bins=10):
    y, p = validate_binary(y, probability)
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    edges = np.arange(n_bins + 1, dtype=float) / n_bins
    indexes = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
    bins = []
    for index in range(n_bins):
        mask = indexes == index
        bins.append(
            CalibrationBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                upper_inclusive=index == n_bins - 1,
                count=int(mask.sum()),
                mean_probability=float(p[mask].mean()) if mask.any() else None,
                observed_fraction=float(y[mask].mean()) if mask.any() else None,
            )
        )
    return tuple(bins)


@register_classification(
    "bernoulli_log_loss",
    direction="minimize",
    role="objective",
    unit="nats",
    log_score=True,
)
def bernoulli_log_loss(y, p, config):
    return float(binary_losses(y, p, config.epsilon).mean())


@register_classification(
    "brier_score", direction="minimize", role="objective", unit="squared_probability"
)
def brier_score(y, p, config):
    return float(np.mean((y - p) ** 2))


@register_classification("auprc", direction="maximize", two_classes=True)
def auprc(y, p, config):
    """Noninterpolated average precision, not trapezoidal area under a PR curve."""
    return float(average_precision_score(y, p))


@register_classification("auroc", direction="maximize", two_classes=True)
def auroc(y, p, config):
    return float(roc_auc_score(y, p))


@register_classification("ece_binned", direction="minimize")
def ece_binned(y, p, config):
    bins = calibration_bins(y, p, config.calibration_bins)
    return float(
        sum(
            bin.count * abs(bin.mean_probability - bin.observed_fraction)
            for bin in bins
            if bin.count
        )
        / len(y)
    )


@register_classification("mean_predicted_probability")
def mean_probability(y, p, config):
    return float(np.mean(p))


@register_classification("observed_prevalence")
def prevalence(y, p, config):
    return float(np.mean(y))


@register_classification("prediction_count", unit="count")
def prediction_count(y, p, config):
    return len(y)


@register_classification("predictive_entropy_mean", unit="nats")
def predictive_entropy(y, p, config):
    return float(np.mean(-xlogy(p, p) - xlogy(1 - p, 1 - p)))


class ClassificationReport(Contract):
    config: EvaluationConfig
    metrics: dict[str, MetricValue]
    calibration: tuple[CalibrationBin, ...]


def evaluate_classification(y, probability, *, config=None):
    config = EvaluationConfig.model_validate(config if config is not None else {})
    y, p = validate_binary(y, probability)
    two_classes = len(np.unique(y)) == 2
    metrics = {}
    for name, spec in CLASSIFICATION_METRICS.items():
        undefined = spec.requires_two_classes and not two_classes
        definition = spec.definition
        if name == "bernoulli_log_loss":
            definition = definition.model_copy(
                update={"probability_clip": config.epsilon}
            )
        metrics[name] = MetricValue(
            metric=definition,
            value=None if undefined else float(spec.function(y, p, config)),
            status="undefined" if undefined else "defined",
            reason="single_class_cohort" if undefined else None,
            n_observations=len(y),
        )
    return ClassificationReport(
        config=config,
        metrics=metrics,
        calibration=calibration_bins(y, p, config.calibration_bins),
    )
