"""Reproducible, chronological synthetic-data experiments."""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.simulation.generator import (
    SimulationConfig,
    generate_telemetry,
)
from healthcare_ai_lab.evaluation.metrics import (
    rmse,
    gaussian_nll,
    gaussian_crps,
)


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (12345, 23456, 34567)
    n_observations: int = 3000
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    alpha: float = 10.0


FEATURE_SETS = {
    "current_only": ["x1", "x2", "regime_proxy"],
    "historical_lags": [
        "x1", "x2", "regime_proxy",
        "phi12", "phi30",
    ],
}


def run_experiment(config: ExperimentConfig):
    """Fit on train, evaluate on validation, never fit on test."""

    if not (
        0 < config.train_fraction
        < config.train_fraction + config.validation_fraction
        < 1
    ):
        raise ValueError("Invalid chronological split fractions")

    results = []

    for seed in config.seeds:
        df = generate_telemetry(
            SimulationConfig(
                seed=seed,
                n_observations=config.n_observations,
            )
        )

        n = len(df)
        split_train = int(n * config.train_fraction)
        split_val = int(
            n * (
                config.train_fraction
                + config.validation_fraction
            )
        )

        train = df.iloc[:split_train]
        validation = df.iloc[split_train:split_val]

        for name, features in FEATURE_SETS.items():
            estimator = make_pipeline(
                StandardScaler(),
                Ridge(alpha=config.alpha),
            )

            estimator.fit(train[features], train["y"])

            train_pred = estimator.predict(train[features])
            prediction = estimator.predict(
                validation[features]
            )

            # Constant-variance baseline estimated on training only.
            variance = np.mean(
                (train["y"].to_numpy() - train_pred) ** 2
            )
            sigma = max(float(np.sqrt(variance)), 1e-6)

            results.append({
                "seed": int(seed),
                "model": name,
                "train_size": len(train),
                "validation_size": len(validation),
                "rmse": rmse(validation["y"], prediction),
                "nll": gaussian_nll(
                    validation["y"], prediction, sigma
                ),
                "crps": gaussian_crps(
                    validation["y"], prediction, sigma
                ),
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="results/baseline.json")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    config = ExperimentConfig(
        seeds=tuple(12345 + i for i in range(args.runs))
    )

    results = run_experiment(config)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps({
            "config": asdict(config),
            "results": results,
        }, indent=2)
    )

    print(f"Saved {len(results)} results to {output}")

    for name in FEATURE_SETS:
        scores = [
            row["rmse"]
            for row in results
            if row["model"] == name
        ]
        print(
            f"{name}: mean RMSE={np.mean(scores):.4f}, "
            f"std={np.std(scores, ddof=1) if len(scores)>1 else 0:.4f}"
        )


if __name__ == "__main__":
    main()
