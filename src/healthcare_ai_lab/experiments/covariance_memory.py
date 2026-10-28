"""Covariance-aware continuous weighting experiment.

No validation outcomes are used to train specialists, state
estimators, covariance weights, or telemetry bin boundaries.
Synthetic regime labels are supervised research targets only.
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


def fit_ridge(data, columns):
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    model.fit(data[columns], data["y"])
    return model


def optimal_weight(e12, e30, fallback=0.5):
    """Empirical optimal convex weight for specialist 12."""
    e12 = np.asarray(e12, dtype=float)
    e30 = np.asarray(e30, dtype=float)

    numerator = np.mean(e30**2 - e12 * e30)
    denominator = np.mean((e12 - e30)**2)

    if denominator < 1e-10:
        return fallback

    return float(np.clip(numerator / denominator, 0, 1))


def evaluate_seed(seed, observations=5000):
    df = generate_telemetry(
        SimulationConfig(
            seed=seed,
            n_observations=observations,
        )
    ).copy()

    n = len(df)
    a = int(n * 0.30)
    b = int(n * 0.60)
    c = int(n * 0.80)

    # Fit specialists on reference-training data only.
    reference_train = df.iloc[:a]
    policy_train = df.iloc[a:b]
    validation = df.iloc[b:c]

    specialists = {
        lag: fit_ridge(
            reference_train,
            BASE + [f"phi{lag}"],
        )
        for lag in (12, 30)
    }

    def predict_specialist(lag, data):
        columns = BASE + [f"phi{lag}"]
        return specialists[lag].predict(data[columns])

    # Historical specialist errors: observable during
    # policy training, unavailable during validation.
    e12 = (
        policy_train["y"].to_numpy()
        - predict_specialist(12, policy_train)
    )
    e30 = (
        policy_train["y"].to_numpy()
        - predict_specialist(30, policy_train)
    )

    global_w = optimal_weight(e12, e30)

    # State-dependent moments: four telemetry regions.
    # Bin edges are learned from policy training only.
    proxy_train = policy_train["regime_proxy"].to_numpy()
    edges = np.unique(
        np.quantile(proxy_train, [0.25, 0.5, 0.75])
    )

    train_bin = np.digitize(proxy_train, edges)
    val_bin = np.digitize(
        validation["regime_proxy"].to_numpy(),
        edges,
    )

    # Shrink local weights toward the global estimate.
    # This limits instability in sparsely populated bins.
    state_weights = {}

    for k in range(len(edges) + 1):
        mask = train_bin == k
        count = int(mask.sum())

        if count < 20:
            state_weights[k] = global_w
            continue

        local_w = optimal_weight(
            e12[mask], e30[mask], fallback=global_w
        )

        shrink = count / (count + 100.0)
        state_weights[k] = (
            shrink * local_w + (1 - shrink) * global_w
        )

    conditional_w = np.array(
        [state_weights[k] for k in val_bin]
    )

    # Reference residual feedback for the regime model.
    # The reference predictor is fitted on the first 30%.
    reference = fit_ridge(reference_train, BASE)

    residuals = pd.Series(np.nan, index=df.index)
    residuals.iloc[a:] = (
        df["y"].iloc[a:].to_numpy()
        - reference.predict(df.iloc[a:][BASE])
    )

    past_error = residuals.shift(1)
    past_proxy = df["regime_proxy"].shift(1)

    features = pd.DataFrame(index=df.index)
    features["proxy"] = df["regime_proxy"]
    features["proxy_mean"] = (
        past_proxy.rolling(12, min_periods=12).mean()
    )
    features["proxy_std"] = (
        past_proxy.rolling(12, min_periods=12).std()
    )
    features["absolute_error"] = past_error.abs()
    features["error_energy"] = (
        past_error.pow(2)
        .rolling(12, min_periods=12).mean()
    )

    columns = list(features.columns)
    fit_idx = df.index[a + 12:b]
    val_idx = df.index[b:c]

    regime_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000),
    )

    regime_model.fit(
        features.loc[fit_idx, columns],
        df.loc[fit_idx, "true_regime"],
    )

    p = regime_model.predict_proba(
        features.loc[val_idx, columns]
    )[:, 1]

    weights = {
        "hard_top1": (p >= 0.5).astype(float),
        "soft_belief": p,
        "global_covariance": np.full_like(p, global_w),
        "state_covariance": conditional_w,
    }

    pred12 = predict_specialist(12, validation)
    pred30 = predict_specialist(30, validation)

    y = validation["y"].to_numpy()
    true_mean = validation["true_mean"].to_numpy()

    records = []

    for name, w in weights.items():
        prediction = w * pred12 + (1 - w) * pred30

        records.append({
            "seed": seed,
            "policy": name,
            "rmse": float(np.sqrt(
                np.mean((y - prediction)**2)
            )),
            "structural_mse": float(np.mean(
                (true_mean - prediction)**2
            )),
            "weight_variation": float(np.mean(
                np.abs(np.diff(w))
            )),
            "mean_weight_12": float(np.mean(w)),
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--output",
        default="results/covariance_memory.csv",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    rows = [
        record
        for i in range(args.runs)
        for record in evaluate_seed(90000 + i)
    ]

    results = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print("\nCOVARIANCE-AWARE MEMORY EXPERIMENT")
    print(
        results.groupby("policy")[
            ["rmse", "structural_mse", "weight_variation"]
        ].agg(["mean", "std"]).round(5).to_string()
    )

    paired = results.pivot(
        index="seed",
        columns="policy",
        values="rmse",
    )

    improvements = paired.drop(
        columns="hard_top1"
    ).rsub(
        paired["hard_top1"],
        axis=0,
    )

    print("\nPaired improvement vs hard selection")
    print(improvements.mean().round(5).to_string())

    print("\nFraction of seeds beating hard selection")
    print((improvements > 0).mean().round(3).to_string())

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
