"""Multi-objective controller analysis on held-out recordings."""

import numpy as np
import pandas as pd

PATH = "results/bidmc_learned_controller.csv"

df = pd.read_csv(PATH)
df = df.loc[df["split"] == "test"].copy()

# These objectives are minimized.
objectives = ["rmse", "mae", "nll", "crps", "p90_absolute_error"]

# Coverage is assessed by distance from nominal, not minimized.
df["coverage_gap"] = (df["coverage_90"] - 0.90).abs()
objectives.append("coverage_gap")

# Aggregate each run so that runs, not individual rows,
# determine the reported performance.
per_run = df.groupby(["run", "policy"])[objectives].mean().reset_index()

summary = per_run.groupby("policy")[objectives].mean()

# Pareto comparison of policy-level average objectives.
values = summary.to_numpy()
dominated = np.zeros(len(summary), dtype=bool)

for i in range(len(summary)):
    for j in range(len(summary)):
        if i == j:
            continue

        if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
            dominated[i] = True
            break

summary["pareto_efficient"] = ~dominated

print("\nMULTI-OBJECTIVE PARETO ANALYSIS")
print(summary.round(4).to_string())

# Compare RMSE on identical runs, including downside risk.
wide = per_run.pivot(
    index="run",
    columns="policy",
    values="rmse",
)

reference = "fixed_050"

print("\nPAIRED RMSE DOWNSIDE RISK")

for candidate in wide.columns:
    if candidate == reference:
        continue

    gain = (wide[reference] - wide[candidate]).dropna()

    print(f"\n{candidate}")
    print("Mean gain:", round(gain.mean(), 4))
    print("Win fraction:", round((gain > 0).mean(), 3))
    print("10th percentile gain:", round(gain.quantile(0.10), 4))
    print("Worst observed gain:", round(gain.min(), 4))

summary.to_csv("results/controller_pareto.csv")

print("\nSaved: results/controller_pareto.csv")
