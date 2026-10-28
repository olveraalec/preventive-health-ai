"""Hard versus continuous memory allocation.

The models share the same predictions and state beliefs.
The last 20% of each trajectory remains reserved.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.simulation.generator import (
    SimulationConfig,
    generate_telemetry,
)


BASE = ["x1", "x2", "regime_proxy"]


def make_ridge(data, features):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(data[features], data["y"])
    return model


def temperature_weight(p, temperature):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    log_odds = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-np.clip(log_odds / temperature, -50, 50)))


def evaluate_seed(seed, observations=5000):
    df = generate_telemetry(
        SimulationConfig(
            seed=seed,
            n_observations=observations,
        )
    ).copy()

    n = len(df)
    ref_end = int(0.30 * n)
    policy_end = int(0.60 * n)
    val_end = int(0.80 * n)

    reference = make_ridge(df.iloc[:ref_end], BASE)

    errors = pd.Series(np.nan, index=df.index)
    errors.iloc[ref_end:] = df["y"].iloc[ref_end:].to_numpy() - reference.predict(
        df.iloc[ref_end:][BASE]
    )

    past_error = errors.shift(1)
    past_proxy = df["regime_proxy"].shift(1)

    features = pd.DataFrame(index=df.index)
    features["proxy"] = df["regime_proxy"]
    features["proxy_mean"] = past_proxy.rolling(12, min_periods=12).mean()
    features["proxy_std"] = past_proxy.rolling(12, min_periods=12).std()
    features["absolute_error"] = past_error.abs()
    features["error_energy"] = past_error.pow(2).rolling(12, min_periods=12).mean()

    cols = list(features.columns)

    fit_idx = df.index[ref_end + 12 : policy_end]
    val_idx = df.index[policy_end:val_end]

    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000),
    )
    estimator.fit(
        features.loc[fit_idx, cols],
        df.loc[fit_idx, "true_regime"],
    )

    p = estimator.predict_proba(features.loc[val_idx, cols])[:, 1]

    train = df.iloc[:policy_end]
    val = df.loc[val_idx]

    predictions = {}
    for lag in (12, 30):
        specialist_features = BASE + [f"phi{lag}"]
        model = make_ridge(train, specialist_features)
        predictions[lag] = model.predict(val[specialist_features])

    # Fixed predeclared strategies; no validation tuning.
    entropy = -(
        p * np.log(np.clip(p, 1e-8, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-8, 1))
    )

    adaptive_tau = 0.35 + 1.65 * entropy / np.log(2)

    weights = {
        "hard_top1": (p >= 0.5).astype(float),
        "soft_belief": p,
        "adaptive_temperature": temperature_weight(p, adaptive_tau),
        "equal_blend": np.full_like(p, 0.5),
    }

    y = val["y"].to_numpy()
    true_mean = val["true_mean"].to_numpy()

    records = []

    for name, w in weights.items():
        prediction = w * predictions[12] + (1 - w) * predictions[30]

        records.append(
            {
                "seed": seed,
                "policy": name,
                "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
                "structural_mse": float(np.mean((true_mean - prediction) ** 2)),
                "mean_weight_12": float(np.mean(w)),
                "weight_variation": float(np.mean(np.abs(np.diff(w)))),
                "mixture_components_used": (1 if name == "hard_top1" else 2),
            }
        )

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--output",
        default="results/continuous_memory.csv",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    rows = [row for i in range(args.runs) for row in evaluate_seed(80000 + i)]

    results = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print("\nCONTINUOUS MEMORY EXPERIMENT")
    print(
        results.groupby("policy")[["rmse", "structural_mse", "weight_variation"]]
        .agg(["mean", "std"])
        .round(5)
        .to_string()
    )

    paired = results.pivot(index="seed", columns="policy", values="rmse")

    # Compare each policy against hard selection by seed.
    improvements = paired.drop(columns="hard_top1").rsub(paired["hard_top1"], axis=0)

    print("\nRMSE improvement relative to hard selection")
    print(improvements.mean().round(5).to_string())

    print("\nFraction of seeds outperforming hard selection")
    print((improvements > 0).mean().round(3).to_string())

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
