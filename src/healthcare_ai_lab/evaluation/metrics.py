"""Pure statistical evaluation functions.

These are research metrics, not HTTP monitoring metrics.
"""

import numpy as np
from scipy.stats import norm


def rmse(y_true, prediction):
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def gaussian_nll(y_true, mean, sigma):
    """Average Gaussian negative log likelihood."""
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(mean, dtype=float)
    std = np.asarray(sigma, dtype=float)

    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("sigma must be positive and finite")

    z = (y - mu) / std
    return float(np.mean(
        np.log(std) + 0.5 * (np.log(2 * np.pi) + z**2)
    ))


def gaussian_crps(y_true, mean, sigma):
    """Average CRPS for Gaussian predictive distributions."""
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(mean, dtype=float)
    std = np.asarray(sigma, dtype=float)

    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("sigma must be positive and finite")

    z = (y - mu) / std

    scores = std * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )

    return float(np.mean(scores))


def interval_coverage(y_true, lower, upper):
    """Fraction of outcomes inside a predictive interval."""
    y = np.asarray(y_true)
    lo = np.asarray(lower)
    hi = np.asarray(upper)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_decision_regret(selected_loss, oracle_loss):
    """Mean excess loss over a shared action-level oracle."""
    selected = np.asarray(selected_loss, dtype=float)
    oracle = np.asarray(oracle_loss, dtype=float)

    if np.any(selected < oracle - 1e-9):
        raise ValueError("Oracle loss cannot exceed selected loss")

    return float(np.mean(selected - oracle))
