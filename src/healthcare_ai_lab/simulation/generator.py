
"""Reproducible synthetic telemetry generator.

Hidden columns are for evaluation only.
They must not be exposed to deployed memory policies.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    n_observations: int = 8000
    seed: int = 35791
    mean_regime_duration: float = 80.0
    proxy_separation: float = 0.4
    proxy_noise: float = 1.0
    sigma_regime_0: float = 1.0
    sigma_regime_1: float = 2.5
    lag_effect: float = 0.12


def generate_telemetry(
    config: SimulationConfig,
) -> pd.DataFrame:
    """Generate a reproducible hidden-regime time series."""

    if config.n_observations <= 30:
        raise ValueError("n_observations must exceed 30")

    if config.mean_regime_duration <= 1:
        raise ValueError("mean_regime_duration must exceed 1")

    if config.proxy_noise < 0:
        raise ValueError("proxy_noise cannot be negative")

    if min(
        config.sigma_regime_0,
        config.sigma_regime_1,
    ) <= 0:
        raise ValueError("Noise scales must be positive")

    rng = np.random.default_rng(config.seed)

    n = config.n_observations
    time = np.arange(n)

    # Persistent hidden binary regime.
    regime = np.zeros(n, dtype=int)
    regime[0] = rng.integers(0, 2)

    switch_probability = 1 / config.mean_regime_duration

    for t in range(1, n):
        switch = rng.random() < switch_probability
        regime[t] = (
            1 - regime[t - 1]
            if switch
            else regime[t - 1]
        )

    # Observable telemetry.
    x1 = (
        70
        + 8 * np.sin(2 * np.pi * time / 48)
        + rng.normal(0, 1.5, n)
    )

    x2 = (
        16
        + 3 * np.cos(2 * np.pi * time / 72)
        + rng.normal(0, 0.8, n)
    )

    proxy = (
        config.proxy_separation * regime
        + rng.normal(0, config.proxy_noise, n)
    )

    data = pd.DataFrame({
        "time": time,
        "x1": x1,
        "x2": x2,
        "regime_proxy": proxy,
        "true_regime": regime,
    })

    data["lag12"] = data["x1"].shift(12)
    data["lag30"] = data["x1"].shift(30)

    data = data.dropna().reset_index(drop=True)

    data["phi12"] = (
        (data["lag12"] - 70)
        * (data["x2"] - 16)
    )

    data["phi30"] = (
        (data["lag30"] - 70)
        * (data["x2"] - 16)
    )

    z = data["true_regime"].to_numpy()

    # Ground-truth conditional mean.
    true_mean = (
        0.4 * (data["x1"].to_numpy() - 70)
        + 0.7 * (data["x2"].to_numpy() - 16)
        + config.lag_effect * (
            z * data["phi12"].to_numpy()
            + (1 - z) * data["phi30"].to_numpy()
        )
    )

    sigma = np.where(
        z == 1,
        config.sigma_regime_1,
        config.sigma_regime_0,
    )

    epsilon = sigma * rng.normal(
        0, 1, len(data)
    )

    data["true_mean"] = true_mean
    data["sigma"] = sigma
    data["epsilon"] = epsilon
    data["y"] = true_mean + epsilon

    return data


OBSERVABLE_COLUMNS = [
    "time",
    "x1",
    "x2",
    "regime_proxy",
    "lag12",
    "lag30",
    "phi12",
    "phi30",
]

EVALUATION_ONLY_COLUMNS = [
    "true_regime",
    "true_mean",
    "sigma",
    "epsilon",
]


def get_observable_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Return only telemetry fields available to policies."""

    return data[OBSERVABLE_COLUMNS].copy()
