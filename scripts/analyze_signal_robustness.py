"""Robustness analysis of covariance-policy exposure.

Uses saved paired alignment experiment results.
A mixture represents expected regret under randomized policy choice,
not interpolation of controller weights.
"""

from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("results/information_alignment.csv")
OUTPUT = Path("results/signal_robustness.csv")

df = pd.read_csv(INPUT)

required = {"scenario", "recording_id", "policy", "regret"}
assert required.issubset(df.columns)

paired = df.pivot(
    index=["scenario", "recording_id"],
    columns="policy",
    values="regret",
)

assert not paired[["static_equal", "continuous"]].isna().any().any()

# Values are expected regrets of randomized policies.
# Do not interpret as actual continuous weights.
rows = []

for q in np.linspace(0, 1, 21):
    mixed_regret = (1 - q) * paired["static_equal"] + q * paired["continuous"]

    for scenario, values in mixed_regret.groupby(level="scenario"):
        rows.append(
            {
                "covariance_exposure": float(q),
                "scenario": scenario,
                "mean_regret": float(values.mean()),
                "std_regret": float(values.std()),
                "p90_regret": float(values.quantile(0.9)),
                "worst_regret": float(values.max()),
            }
        )

results = pd.DataFrame(rows)

summary = results.pivot(
    index="covariance_exposure",
    columns="scenario",
    values="mean_regret",
)

summary["worst_scenario"] = summary.max(axis=1)
summary["equal_scenario_average"] = summary[
    ["aligned", "uninformative", "misleading"]
].mean(axis=1)

print("\nROBUST POLICY EXPOSURE")
print(summary.round(4).to_string())

minimax_q = summary["worst_scenario"].idxmin()
average_q = summary["equal_scenario_average"].idxmin()

print("\nOPTIMAL EXPOSURE UNDER TWO CRITERIA")
print("Minimax q:", minimax_q)
print("Equal-scenario average q:", average_q)

print("\nBASELINE AND FULL COVARIANCE")
print(summary.loc[[0.0, 1.0]].round(4).to_string())

print("\nDOWNSIDE RISK AT q=0, 0.25, 0.50, 1")
print(
    results[results["covariance_exposure"].isin([0.0, 0.25, 0.5, 1.0])][
        [
            "covariance_exposure",
            "scenario",
            "mean_regret",
            "p90_regret",
            "worst_regret",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
