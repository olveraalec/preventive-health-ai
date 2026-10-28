"""Controlled overfitting and distribution-shift experiment.

Synthetic regime labels may be used for supervised training.
The deliberately spurious feature is an experimental intervention,
not a deployable or time-valid clinical input.

Data partitions:
0-30%: reference predictor
30-60%: policy training
60-80%: validation
80-100%: untouched reserve
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.simulation.generator import (
    SimulationConfig,
    generate_telemetry,
)


def evaluate_seed(seed, n_observations=5000):
    rng = np.random.default_rng(seed + 800000)

    df = generate_telemetry(
        SimulationConfig(
            seed=seed,
            n_observations=n_observations,
        )
    ).copy()

    n = len(df)
    ref_end = int(n * 0.30)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    base = ["x1", "x2", "regime_proxy"]

    # Train reference model using only the first partition.
    reference = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )

    reference.fit(
        df.iloc[:ref_end][base],
        df.iloc[:ref_end]["y"],
    )

    # Historical errors are out-of-sample relative
    # to the reference model after ref_end.
    residual = pd.Series(np.nan, index=df.index)

    residual.iloc[ref_end:] = (
        df["y"].iloc[ref_end:].to_numpy()
        - reference.predict(df.iloc[ref_end:][base])
    )

    past_error = residual.shift(1)

    df["error_energy"] = (
        past_error.pow(2)
        .rolling(12, min_periods=12)
        .mean()
    )

    df["absolute_error"] = past_error.abs()

    df["proxy_mean"] = (
        df["regime_proxy"]
        .shift(1)
        .rolling(12, min_periods=12)
        .mean()
    )

    df["proxy_std"] = (
        df["regime_proxy"]
        .shift(1)
        .rolling(12, min_periods=12)
        .std()
    )

    baseline_features = [
        "regime_proxy",
        "proxy_mean",
        "proxy_std",
        "error_energy",
        "absolute_error",
    ]

    # 100 features unrelated to the hidden regime.
    noise_features = []

    for j in range(100):
        col = f"random_feature_{j}"
        df[col] = rng.normal(size=n)
        noise_features.append(col)

    # Deliberate spurious correlation:
    # nearly reveals the label in policy training;
    # provides no useful information in validation.
    #
    # This is a controlled distribution-shift example.
    df["spurious"] = rng.normal(size=n)

    df.loc[
        df.index[ref_end:train_end],
        "spurious"
    ] = (
        df["true_regime"].iloc[ref_end:train_end].to_numpy()
        + rng.normal(
            0, 0.15, train_end - ref_end
        )
    )

    train = df.iloc[ref_end + 12:train_end]
    val = df.iloc[train_end:val_end]

    configurations = {
        "baseline": baseline_features,
        "plus_100_noise": baseline_features + noise_features,
        "spurious": baseline_features + ["spurious"],
        "spurious_plus_noise": (
            baseline_features + noise_features + ["spurious"]
        ),
    }

    records = []

    for name, features in configurations.items():

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=10.0,
                max_iter=3000,
            ),
        )

        model.fit(
            train[features],
            train["true_regime"],
        )

        p_train = model.predict_proba(
            train[features]
        )[:, 1]

        p_val = model.predict_proba(
            val[features]
        )[:, 1]

        z_train = train["true_regime"].to_numpy()
        z_val = val["true_regime"].to_numpy()

        train_brier = float(np.mean(
            (p_train - z_train) ** 2
        ))

        val_brier = float(np.mean(
            (p_val - z_val) ** 2
        ))

        train_nll = float(log_loss(
            z_train, p_train, labels=[0, 1]
        ))

        val_nll = float(log_loss(
            z_val, p_val, labels=[0, 1]
        ))

        records.append({
            "seed": seed,
            "model": name,
            "n_features": len(features),
            "train_brier": train_brier,
            "val_brier": val_brier,
            "brier_gap": val_brier - train_brier,
            "train_nll": train_nll,
            "val_nll": val_nll,
            "nll_gap": val_nll - train_nll,
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--output",
        default="results/overfitting.csv",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    rows = []

    for i in range(args.runs):
        rows.extend(evaluate_seed(60000 + i))

    results = pd.DataFrame(rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    summary = (
        results.groupby("model")[
            [
                "train_brier",
                "val_brier",
                "brier_gap",
                "train_nll",
                "val_nll",
                "nll_gap",
            ]
        ]
        .mean()
        .round(4)
    )

    print("\nOVERFITTING EXPERIMENT")
    print(summary.to_string())
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
