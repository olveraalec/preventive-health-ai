"""Causal Gaussian HMM belief filtering.

The HMM parameters are supplied externally and must be
estimated using training data only.

No future observations or hidden ground-truth labels are used.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class HMMConfig:
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    initial: np.ndarray


class GaussianHMMFilter:
    def __init__(self, config: HMMConfig):
        self.config = config

        self.transition = np.asarray(config.transition, dtype=float)
        self.means = np.asarray(config.means, dtype=float)
        self.variances = np.asarray(config.variances, dtype=float)
        self.initial = np.asarray(config.initial, dtype=float)

        n_states, n_features = self.means.shape

        if self.transition.shape != (n_states, n_states):
            raise ValueError("Invalid transition shape")

        if self.variances.shape != self.means.shape:
            raise ValueError("Invalid variance shape")

        if self.initial.shape != (n_states,):
            raise ValueError("Invalid initial distribution")

        for array in (
            self.transition,
            self.means,
            self.variances,
            self.initial,
        ):
            if not np.isfinite(array).all():
                raise ValueError("HMM parameters must be finite")

        if np.any(self.variances <= 0):
            raise ValueError("Variances must be positive")

        if np.any(self.transition < 0):
            raise ValueError("Negative transition probability")

        if np.any(self.initial < 0):
            raise ValueError("Negative initial probability")

        if not np.allclose(self.transition.sum(axis=1), 1.0):
            raise ValueError("Transition rows must sum to 1")

        if not np.isclose(self.initial.sum(), 1.0):
            raise ValueError("Initial probabilities must sum to 1")

        self.n_features = n_features
        self.reset()

    def reset(self):
        """Reset belief at the start of a new patient episode."""
        self.belief = self.initial.copy()

    def update(self, observation):
        """Filter one observation using past and current data."""
        x = np.asarray(observation, dtype=float)

        if x.shape != (self.n_features,):
            raise ValueError("Invalid observation shape")

        # Missing channels contribute no emission evidence.
        observed = np.isfinite(x)

        if np.any(np.isinf(x)):
            raise ValueError("Infinite observations are invalid")

        prior = self.belief @ self.transition

        if not observed.any():
            self.belief = prior
            return prior.copy()

        mu = self.means[:, observed]
        var = self.variances[:, observed]
        values = x[observed]

        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * var) + ((values - mu) ** 2) / var,
            axis=1,
        )

        log_posterior = np.log(np.maximum(prior, 1e-300)) + log_likelihood

        shifted = log_posterior - log_posterior.max()
        posterior = np.exp(shifted)
        posterior /= posterior.sum()

        self.belief = posterior
        return posterior.copy()

    def filter_sequence(self, observations):
        """Sequential filtering; never uses backward smoothing."""
        data = np.asarray(observations, dtype=float)

        if data.ndim != 2 or data.shape[1] != self.n_features:
            raise ValueError("Expected [time, features]")

        self.reset()
        return np.array([self.update(row) for row in data])

    @staticmethod
    def entropy(belief):
        p = np.asarray(belief, dtype=float)
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))
