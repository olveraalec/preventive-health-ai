"""Central registry of research evaluation metrics."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class MetricSpec:
    name: str
    function: Callable
    direction: str
    category: str
    requires_sigma: bool = False


METRICS: dict[str, MetricSpec] = {}


def register(name, direction, category, requires_sigma=False):
    """Register a metric with explicit optimization direction."""
    if direction not in ("minimize", "maximize", "target"):
        raise ValueError("Invalid metric direction")

    def decorator(func):
        if name in METRICS:
            raise ValueError(f"Duplicate metric: {name}")

        METRICS[name] = MetricSpec(
            name=name,
            function=func,
            direction=direction,
            category=category,
            requires_sigma=requires_sigma,
        )
        return func

    return decorator


@register("rmse", "minimize", "prediction")
def rmse(y, prediction):
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


@register("mae", "minimize", "prediction")
def mae(y, prediction):
    return float(np.mean(np.abs(y - prediction)))


@register("bias", "target", "prediction")
def bias(y, prediction):
    return float(np.mean(y - prediction))


@register("p90_absolute_error", "minimize", "risk")
def p90_error(y, prediction):
    return float(np.quantile(np.abs(y - prediction), 0.90))


@register("nll", "minimize", "probabilistic", True)
def gaussian_nll(y, prediction, sigma):
    z = (y - prediction) / sigma
    return float(np.mean(np.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5 * z**2))


@register("crps", "minimize", "probabilistic", True)
def gaussian_crps(y, prediction, sigma):
    z = (y - prediction) / sigma
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    cdf = ndtr(z)

    scores = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return float(np.mean(scores))


@register("coverage_90", "target", "calibration", True)
def coverage_90(y, prediction, sigma):
    z90 = 1.6448536269514722
    return float(np.mean(np.abs(y - prediction) <= z90 * sigma))


@register("width_90", "minimize", "calibration", True)
def width_90(y, prediction, sigma):
    return float(np.mean(2 * 1.6448536269514722 * sigma))


def evaluate(y, prediction, sigma=None):
    """Calculate all supported metrics with input validation."""
    y = np.asarray(y, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)

    if len(y) == 0 or y.shape != prediction.shape:
        raise ValueError("Invalid prediction dimensions")

    if not np.all(np.isfinite(y)):
        raise ValueError("Targets contain nonfinite values")

    if not np.all(np.isfinite(prediction)):
        raise ValueError("Predictions contain nonfinite values")

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(y.shape, float(sigma))
        else:
            sigma = sigma.reshape(-1)

        if sigma.shape != y.shape:
            raise ValueError("Invalid uncertainty dimensions")
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("Sigma must be finite and positive")

    results = {}

    for name, spec in METRICS.items():
        if spec.requires_sigma:
            if sigma is None:
                continue
            value = spec.function(y, prediction, sigma)
        else:
            value = spec.function(y, prediction)

        results[name] = float(value)

    return results


def evaluate_classification(y, probability, *, config=None):
    """Separate Bernoulli entry point; never changes the Gaussian METRICS map."""
    from healthcare_ai_lab.evaluation.classification import (
        evaluate_classification as score,
    )

    return score(y, probability, config=config)


def evaluate_classification_policies(predictions, outcomes, **kwargs):
    """Aligned retrospective patient/controller evaluation, separate from forecasting."""
    from healthcare_ai_lab.evaluation.patient_classification import evaluate_policies

    return evaluate_policies(predictions, outcomes, **kwargs)


def classification_metric_definitions():
    """Discover separate Bernoulli definitions; derived report metrics carry their own metadata."""
    from healthcare_ai_lab.evaluation.classification import CLASSIFICATION_METRICS

    return {name: spec.definition for name, spec in CLASSIFICATION_METRICS.items()}
