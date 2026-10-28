"""Phase A.11 — Sequential, constrained hybrid allocation.

Exploratory patient-grouped validation.
Uses OOF training errors for moment estimation.
Validation outcomes never update the controller.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from healthcare_ai_lab.evaluation.registry import evaluate

# Reuse the previous experiment through the scripts directory.
from bidmc_psd_covariance import (
    TARGET,
    DATA,
    BASE,
    MEMORY,
    STATE,
    PSDGate,
    crossfit,
    model,
)


OUTPUT = Path("results/bidmc_hybrid_controller.csv")
RUNS = 10
BANDWIDTH = 0.5
RHO = 0.25

# Predeclared candidate policies; no validation tuning.
POLICIES = {
    "equal_blend": (0.0, 0.0, 0.0),
    "covariance": (0.0, 0.0, 0.0),
    "conservative": (0.5, 0.0, 0.0),
    "stable": (0.5, 0.5, 0.0),
    "hybrid": (0.5, 0.5, 0.05),
}


def sequential_weights(moments, lam_complex, lam_stable, lam_memory):
    """Return causal weights for the four-feature expert."""
    v1 = moments[:, 0]
    v4 = moments[:, 1]
    cross = moments[:, 2]

    weights = np.empty(len(moments), dtype=float)
    previous = 0.5

    for t in range(len(moments)):
        variance_difference = max(v1[t] + v4[t] - 2 * cross[t], 0.0)

        numerator = (
            v1[t]
            - cross[t]
            + lam_complex * 0.5
            + lam_stable * previous
            - lam_memory / 2
        )
        denominator = variance_difference + lam_complex + lam_stable

        if denominator <= 1e-10:
            weight = previous
        else:
            weight = float(np.clip(numerator / denominator, 0.0, 1.0))

        weights[t] = weight
        previous = weight

    return weights


def main():
    df = pd.read_parquet(DATA)
    df = df.sort_values(["recording_id", "time_seconds"]).dropna(subset=[TARGET]).copy()

    flags = [f"{col}_missing" for col in BASE if f"{col}_missing" in df.columns]
    current = BASE + flags

    features = {
        "one": current + MEMORY[:1],
        "four": current + MEMORY,
    }

    required = set(current + MEMORY + STATE + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    ids = np.array(sorted(df.recording_id.unique()))
    if len(ids) != 10:
        raise ValueError("Expected 10 recordings")

    rng = np.random.default_rng(12345)
    rows = []

    for run in range(RUNS):
        permutation = rng.permutation(ids)

        train = df[df.recording_id.isin(permutation[:6])]
        validation = df[df.recording_id.isin(permutation[6:8])]

        # Last two recordings intentionally unused.
        oof = crossfit(train, features)
        gate = PSDGate().fit(oof)

        experts = {}
        for name, cols in features.items():
            expert = model()
            expert.fit(train[cols], train[TARGET])
            experts[name] = expert

        # Shared OOF predictive scale for comparable scoring.
        sigma = max(
            float(
                np.sqrt(np.mean((oof[TARGET].to_numpy() - oof["four"].to_numpy()) ** 2))
            ),
            1e-6,
        )

        for pid, patient in validation.groupby("recording_id"):
            patient = patient.sort_values("time_seconds")
            y = patient[TARGET].to_numpy()

            p1 = experts["one"].predict(patient[features["one"]])
            p4 = experts["four"].predict(patient[features["four"]])

            local = gate.local_moments(patient, BANDWIDTH)

            moments = (1 - RHO) * local + RHO * gate.global_moments[None, :]

            # Use a shared distributional scale.
            distribution_sigma = np.full(len(y), sigma)

            for name, penalties in POLICIES.items():
                if name == "equal_blend":
                    weights = np.full(len(y), 0.5)
                else:
                    weights = sequential_weights(moments, *penalties)

                prediction = p1 + weights * (p4 - p1)

                scores = evaluate(y, prediction, sigma=distribution_sigma)

                variation = (
                    float(np.mean(np.abs(np.diff(weights))))
                    if len(weights) > 1
                    else 0.0
                )

                rows.append(
                    {
                        "run": run,
                        "recording_id": str(pid),
                        "policy": name,
                        **scores,
                        "mean_weight": float(weights.mean()),
                        "weight_std": float(weights.std()),
                        "weight_variation": variation,
                        "mean_weight_squared": float(np.mean(weights**2)),
                        "mean_deviation_from_equal": float(
                            np.mean((weights - 0.5) ** 2)
                        ),
                    }
                )

    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    metrics = [
        "rmse",
        "mae",
        "bias",
        "p90_absolute_error",
        "nll",
        "crps",
        "coverage_90",
        "width_90",
        "mean_weight",
        "weight_std",
        "weight_variation",
        "mean_deviation_from_equal",
    ]

    print("\nFULL METRIC REGISTRY")
    print(result.groupby("policy")[metrics].agg(["mean", "std"]).round(5).to_string())

    for metric in [
        "rmse",
        "nll",
        "crps",
        "p90_absolute_error",
    ]:
        paired = result.pivot(
            index=["run", "recording_id"],
            columns="policy",
            values=metric,
        )

        print(f"\nPAIRED {metric.upper()} IMPROVEMENTS")
        for name in POLICIES:
            if name == "equal_blend":
                continue

            gain = paired["equal_blend"] - paired[name]
            print(
                name,
                "mean:",
                round(gain.mean(), 5),
                "median:",
                round(gain.median(), 5),
                "win:",
                round((gain > 0).mean(), 3),
                "std:",
                round(gain.std(), 5),
            )

    print("\nWORST RECORDING RMSE")
    print(result.groupby("policy")["rmse"].max().round(4).to_string())

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
