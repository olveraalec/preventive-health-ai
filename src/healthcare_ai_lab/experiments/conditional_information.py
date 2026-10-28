"""Conditional information and residual-interaction ablation.

Uses strictly past error feedback and chronological fitting.
Hidden regime labels are synthetic supervised targets only.
The final 20% of each trajectory remains untouched.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from healthcare_ai_lab.simulation.generator import (
    SimulationConfig,
    generate_telemetry,
)


BASE = ["x1", "x2", "regime_proxy"]


def evaluate_seed(seed, observations=5000):
    df = generate_telemetry(
        SimulationConfig(
            seed=seed,
            n_observations=observations,
        )
    ).copy()

    n = len(df)
    a = int(0.30 * n)
    b = int(0.60 * n)
    c = int(0.80 * n)

    # Reference predictor fitted only on earlier observations.
    reference = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    reference.fit(df.iloc[:a][BASE], df.iloc[:a]["y"])

    # Unavailable outcomes are not treated as known errors.
    residual = pd.Series(np.nan, index=df.index)

    residual.iloc[a:] = (
        df["y"].iloc[a:].to_numpy()
        - reference.predict(df.iloc[a:][BASE])
    )

    # Information at time t must originate before t.
    past_error = residual.shift(1)
    past_proxy = df["regime_proxy"].shift(1)

    features = pd.DataFrame(index=df.index)

    features["proxy"] = df["regime_proxy"]
    features["proxy_mean"] = (
        past_proxy.rolling(12, min_periods=12).mean()
    )
    features["proxy_std"] = (
        past_proxy.rolling(12, min_periods=12).std()
    )

    features["abs_error"] = past_error.abs()
    features["error_energy"] = (
        past_error.pow(2)
        .rolling(12, min_periods=12).mean()
    )

    # Center and scale using fixed, predeclared values.
    # Learned StandardScaler parameters use training only.
    x = features["proxy"] - 0.2
    e = features["abs_error"]
    v = features["error_energy"]

    features["proxy_x_error"] = x * e
    features["proxy_x_energy"] = x * v

    features["error_squared"] = e**2
    features["energy_squared"] = v**2
    features["proxy_squared"] = x**2
    features["error_x_energy"] = e * v

    groups = {
        "A_telemetry": [
            "proxy", "proxy_mean", "proxy_std"
        ],
        "B_plus_errors": [
            "proxy", "proxy_mean", "proxy_std",
            "abs_error", "error_energy"
        ],
        "C_interactions": [
            "proxy", "proxy_mean", "proxy_std",
            "abs_error", "error_energy",
            "proxy_x_error", "proxy_x_energy"
        ],
        "D_nonlinear": list(features.columns),
    }

    train_idx = df.index[a + 12:b]
    val_idx = df.index[b:c]

    truth_train = df.loc[train_idx, "true_regime"]
    truth_val = df.loc[val_idx, "true_regime"]

    records = []

    for name, columns in groups.items():
        X_train = features.loc[train_idx, columns]
        X_val = features.loc[val_idx, columns]

        if not np.isfinite(X_train.to_numpy()).all():
            raise ValueError("Nonfinite training features")
        if not np.isfinite(X_val.to_numpy()).all():
            raise ValueError("Nonfinite validation features")

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000),
        )

        model.fit(X_train, truth_train)

        p_train = model.predict_proba(X_train)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]
        z_val = truth_val.to_numpy()

        records.append({
            "seed": seed,
            "model": name,
            "train_nll": float(log_loss(
                truth_train, p_train, labels=[0, 1]
            )),
            "val_nll": float(log_loss(
                truth_val, p_val, labels=[0, 1]
            )),
            "val_brier": float(np.mean(
                (p_val - z_val)**2
            )),
            "features": len(columns),
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--output",
        default="results/conditional_information.csv",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    results = pd.DataFrame([
        item
        for i in range(args.runs)
        for item in evaluate_seed(70000 + i)
    ])

    table = results.pivot(
        index="seed",
        columns="model",
        values="val_nll"
    )

    baseline = table["A_telemetry"]
    deltas = pd.DataFrame({
        "B_minus_A_gain":
            baseline - table["B_plus_errors"],
        "C_minus_B_gain":
            table["B_plus_errors"] - table["C_interactions"],
        "D_minus_C_gain":
            table["C_interactions"] - table["D_nonlinear"],
    })

    print("\nMODEL PERFORMANCE (mean across seeds)")
    print(
        results.groupby("model")[
            ["train_nll", "val_nll", "val_brier"]
        ].mean().round(5).to_string()
    )

    print("\nPAIRED VALIDATION NLL IMPROVEMENT")
    print("(Positive means the more complex model is better)")
    print(deltas.agg(["mean", "std"]).round(5).to_string())

    print("\nFraction of seeds with positive improvement")
    print((deltas > 0).mean().round(3).to_string())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
