"""Online Bayesian hidden-regime estimation.

At timestamp t:
1. Incorporate a newly available residual from t-1.
2. Predict the hidden regime forward to t.
3. Incorporate the current observable proxy.

The true hidden regime is never supplied to this estimator.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StateEstimate:
    timestamp: int
    probability_regime_1: float
    uncertainty_nats: float
    feedback_used: bool


class BayesianStateEstimator:
    def __init__(
        self,
        switch_probability=1 / 80,
        proxy_separation=0.4,
        proxy_noise=1.0,
        sigma_0=1.0,
        sigma_1=2.5,
        use_error_feedback=True,
    ):
        if not 0 < switch_probability < 1:
            raise ValueError("Invalid switch probability")
        if proxy_noise <= 0:
            raise ValueError("Proxy noise must be positive")
        if min(sigma_0, sigma_1) <= 0:
            raise ValueError("Noise scales must be positive")

        self.switch_probability = switch_probability
        self.proxy_separation = proxy_separation
        self.proxy_noise = proxy_noise
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.use_error_feedback = use_error_feedback

        self.probability = 0.5
        self.last_timestamp = None

    @staticmethod
    def _entropy(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return -p * math.log(p) - (1 - p) * math.log(1 - p)

    @staticmethod
    def _update_probability(prior, log_likelihood_ratio):
        p = min(max(prior, 1e-12), 1 - 1e-12)
        log_odds = math.log(p / (1 - p))
        combined = log_odds + log_likelihood_ratio

        if combined >= 0:
            return 1 / (1 + math.exp(-combined))

        value = math.exp(combined)
        return value / (1 + value)

    def _transition(self, p, steps):
        # Exact propagation for a symmetric two-state Markov chain.
        q = self.switch_probability
        return 0.5 + (p - 0.5) * (1 - 2 * q) ** steps

    def update(self, timestamp, proxy, previous_residual=None):
        if not isinstance(timestamp, int):
            raise TypeError("timestamp must be an integer")

        if self.last_timestamp is not None:
            if timestamp <= self.last_timestamp:
                raise ValueError("Timestamps must increase")

        if not math.isfinite(proxy):
            raise ValueError("Proxy must be finite")

        if previous_residual is not None:
            if not math.isfinite(previous_residual):
                raise ValueError("Residual must be finite")
            if self.last_timestamp is None:
                raise ValueError("No prior state for feedback")

        p = self.probability
        feedback_used = False

        # The residual belongs to the previous observation.
        # It is incorporated before the state transition.
        if previous_residual is not None and self.use_error_feedback:
            e = previous_residual

            log_ratio = (
                math.log(self.sigma_0 / self.sigma_1)
                + 0.5 * e**2 * (
                    1 / self.sigma_0**2
                    - 1 / self.sigma_1**2
                )
            )

            p = self._update_probability(p, log_ratio)
            feedback_used = True

        if self.last_timestamp is not None:
            steps = timestamp - self.last_timestamp
            p = self._transition(p, steps)

        # Gaussian proxy evidence:
        # regime 0 -> N(0, proxy_noise^2)
        # regime 1 -> N(proxy_separation, proxy_noise^2)
        log_proxy_ratio = (
            self.proxy_separation * proxy
            - 0.5 * self.proxy_separation**2
        ) / self.proxy_noise**2

        p = self._update_probability(p, log_proxy_ratio)

        self.probability = p
        self.last_timestamp = timestamp

        return StateEstimate(
            timestamp=timestamp,
            probability_regime_1=p,
            uncertainty_nats=self._entropy(p),
            feedback_used=feedback_used,
        )
