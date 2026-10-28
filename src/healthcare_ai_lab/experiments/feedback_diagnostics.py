"""Compare valid, misspecified, and negative-control feedback.

IMPORTANT:
The innovation policy reads privileged synthetic information.
It is diagnostic only, never a deployable policy.
"""

import argparse

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.simulation.generator import (
    SimulationConfig,
    generate_telemetry,
)
from healthcare_ai_lab.models.state_estimator import (
    BayesianStateEstimator,
)


def fit_model(train, features):
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    model.fit(train[features], train["y"])
    return model


def evaluate_seed(seed, n_observations=5000):
    df = generate_telemetry(
        SimulationConfig(
            seed=seed,
            n_observations=n_observations,
        )
    )

    train_end = int(len(df) * 0.6)
    val_end = int(len(df) * 0.8)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]

    base_features = ["x1", "x2", "regime_proxy"]
    reference = fit_model(train, base_features)

    specialists = {
        lag: fit_model(
            train, base_features + [f"phi{lag}"]
        )
        for lag in (12, 30)
    }

    predictions = {
        lag: model.predict(
            val[base_features + [f"phi{lag}"]]
        )
        for lag, model in specialists.items()
    }

    y = val["y"].to_numpy()
    true_mean = val["true_mean"].to_numpy()

    # Observable reference-model residuals.
    raw_residuals = y - reference.predict(
        val[base_features]
    )

    # Privileged data: simulation diagnostics ONLY.
    innovations = val["epsilon"].to_numpy()

    # Misaligned reference residuals, without using future
    # outcomes. Constructed within this evaluation sequence.
    rng = np.random.default_rng(seed + 900000)
    shuffled_residuals = rng.permutation(raw_residuals)

    feedback_streams = {
        "proxy_only": None,
        "raw_residual": raw_residuals,
        "true_innovation_DIAGNOSTIC": innovations,
        "shuffled_residual_CONTROL": shuffled_residuals,
    }

    oracle_loss = np.minimum(
        (true_mean - predictions[12]) ** 2,
        (true_mean - predictions[30]) ** 2,
    )

    times = val["time"].to_numpy()
    proxies = val["regime_proxy"].to_numpy()
    actual_regimes = val["true_regime"].to_numpy()

    rows = []

    for policy, feedback in feedback_streams.items():
        estimator = BayesianStateEstimator(
            use_error_feedback=(feedback is not None)
        )

        choices = []
        beliefs = []

        for i, (timestamp, proxy) in enumerate(
            zip(times, proxies)
        ):
            past_error = None

            if feedback is not None and i > 0:
                past_error = float(feedback[i - 1])

            state = estimator.update(
                int(timestamp),
                float(proxy),
                previous_residual=past_error,
            )

            p = state.probability_regime_1
            beliefs.append(p)
            choices.append(12 if p >= 0.5 else 30)

        choices = np.asarray(choices)
        beliefs = np.asarray(beliefs)

        selected_prediction = np.where(
            choices == 12,
            predictions[12],
            predictions[30],
        )

        structural_loss = (
            true_mean - selected_prediction
        ) ** 2

        rows.append({
            "seed": seed,
            "policy": policy,
            "rmse": float(np.sqrt(np.mean(
                (y - selected_prediction) ** 2
            ))),
            "regret": float(np.mean(
                structural_loss - oracle_loss
            )),
            "regime_accuracy": float(np.mean(
                (beliefs >= 0.5) == actual_regimes
            )),
            "belief_brier": float(np.mean(
                (beliefs - actual_regimes) ** 2
            )),
            "switch_rate": float(np.mean(
                choices[1:] != choices[:-1]
            )),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--output",
        default="results/feedback_diagnostics.csv",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    results = pd.DataFrame([
        row
        for i in range(args.runs)
        for row in evaluate_seed(50000 + i)
    ])

    from pathlib import Path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print("\nFEEDBACK DIAGNOSTICS")
    print(
        results.groupby("policy")[
            ["rmse", "regret", "regime_accuracy",
             "belief_brier", "switch_rate"]
        ].agg(["mean", "std"]).round(4).to_string()
    )
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
