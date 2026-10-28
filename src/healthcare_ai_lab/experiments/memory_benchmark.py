"""Chronological benchmark for bounded memory policies.

This benchmark is preliminary:
- Bayesian parameters match the simulator.
- Only prior prediction errors are used as feedback.
- Ground-truth conditional means are evaluation-only.
- Final test data is not used.
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
from healthcare_ai_lab.memory.policies import (
    MemoryCandidate,
    AdaptiveRegimePolicy,
)
from healthcare_ai_lab.memory.controller import MemoryController


def fit_ridge(train, features):
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
    validation = df.iloc[train_end:val_end]

    reference_features = ["x1", "x2", "regime_proxy"]

    reference = fit_ridge(train, reference_features)

    specialists = {
        12: fit_ridge(
            train,
            ["x1", "x2", "regime_proxy", "phi12"],
        ),
        30: fit_ridge(
            train,
            ["x1", "x2", "regime_proxy", "phi30"],
        ),
    }

    pred_reference = reference.predict(
        validation[reference_features]
    )

    predictions = {
        lag: model.predict(
            validation[
                ["x1", "x2", "regime_proxy", f"phi{lag}"]
            ]
        )
        for lag, model in specialists.items()
    }

    estimators = {
        "bayesian_proxy": BayesianStateEstimator(
            use_error_feedback=False
        ),
        "bayesian_feedback": BayesianStateEstimator(
            use_error_feedback=True
        ),
    }

    controller = MemoryController(
        AdaptiveRegimePolicy(),
        budget=1,
    )

    selected_actions = {
        "fixed_12": [],
        "fixed_30": [],
        "bayesian_proxy": [],
        "bayesian_feedback": [],
    }

    residuals = (
        validation["y"].to_numpy() - pred_reference
    )

    times = validation["time"].to_numpy()
    proxies = validation["regime_proxy"].to_numpy()

    for i, (timestamp, proxy) in enumerate(
        zip(times, proxies)
    ):
        selected_actions["fixed_12"].append(12)
        selected_actions["fixed_30"].append(30)

        candidates = [
            MemoryCandidate("lag12", int(timestamp - 12), 12),
            MemoryCandidate("lag30", int(timestamp - 30), 30),
        ]

        for name, estimator in estimators.items():
            previous_residual = (
                float(residuals[i - 1])
                if i > 0 and name == "bayesian_feedback"
                else None
            )

            state = estimator.update(
                int(timestamp),
                float(proxy),
                previous_residual=previous_residual,
            )

            decision = controller.select(state, candidates)

            chosen = (
                12 if decision.selected_memory_ids[0] == "lag12"
                else 30
            )

            selected_actions[name].append(chosen)

    y = validation["y"].to_numpy()
    true_mean = validation["true_mean"].to_numpy()

    squared_loss_12 = (true_mean - predictions[12]) ** 2
    squared_loss_30 = (true_mean - predictions[30]) ** 2

    oracle_loss = np.minimum(
        squared_loss_12, squared_loss_30
    )

    results = []

    for name, actions in selected_actions.items():
        actions = np.asarray(actions)

        prediction = np.where(
            actions == 12,
            predictions[12],
            predictions[30],
        )

        structural_loss = (true_mean - prediction) ** 2

        results.append({
            "seed": seed,
            "policy": name,
            "rmse": float(
                np.sqrt(np.mean((y - prediction) ** 2))
            ),
            "mean_regret": float(
                np.mean(structural_loss - oracle_loss)
            ),
            "switch_rate": float(
                np.mean(actions[1:] != actions[:-1])
            ),
            "memory_budget": 1,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--observations", type=int, default=5000)
    parser.add_argument(
        "--output",
        default="results/memory_benchmark.csv",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    records = []

    for i in range(args.runs):
        records.extend(
            evaluate_seed(
                seed=50000 + i,
                n_observations=args.observations,
            )
        )

    results = pd.DataFrame(records)

    from pathlib import Path

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    summary = (
        results.groupby("policy")[
            ["rmse", "mean_regret", "switch_rate"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )

    print("\nMemory-policy benchmark")
    print(summary.to_string())
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
