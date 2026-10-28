"""Paired seed-level comparison of experimental policies."""

import argparse
import numpy as np
import pandas as pd


def paired_bootstrap(
    data,
    baseline,
    candidate,
    metric,
    n_bootstrap=5000,
    seed=12345,
):
    table = data.pivot(
        index="seed",
        columns="model",
        values=metric,
    )

    difference = (
        table[baseline] - table[candidate]
    ).dropna().to_numpy()

    if len(difference) < 2:
        raise ValueError("At least two paired seeds required")

    rng = np.random.default_rng(seed)

    # Resample whole seed-level differences rather than
    # treating autocorrelated observations as independent.
    indices = rng.integers(
        0, len(difference),
        size=(n_bootstrap, len(difference)),
    )

    bootstrap_means = difference[indices].mean(axis=1)

    lower, upper = np.quantile(
        bootstrap_means, [0.025, 0.975]
    )

    return {
        "comparison": f"{candidate} vs {baseline}",
        "metric": metric,
        "mean_improvement": difference.mean(),
        "ci_lower": lower,
        "ci_upper": upper,
        "fraction_positive": np.mean(difference > 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/conditional_information_30seeds.csv",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)

    comparisons = [
        ("A_telemetry", "B_plus_errors"),
        ("B_plus_errors", "C_interactions"),
        ("C_interactions", "D_nonlinear"),
        ("B_plus_errors", "D_nonlinear"),
    ]

    records = []

    for metric in ("val_nll", "val_brier"):
        for baseline, candidate in comparisons:
            records.append(
                paired_bootstrap(
                    data, baseline, candidate, metric
                )
            )

    report = pd.DataFrame(records)
    print(report.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
