"""Paired, multi-metric controller analysis."""

import numpy as np
import pandas as pd

SEED = 12345
BOOTSTRAPS = 5000
PATH = "results/bidmc_learned_controller.csv"

df = pd.read_csv(PATH)
df = df[df["split"] == "test"].copy()

metrics = [
    "rmse",
    "mae",
    "nll",
    "crps",
    "p90_absolute_error",
    "width_90",
]

reference = "fixed_050"
rng = np.random.default_rng(SEED)

# One mean per run/policy: recordings equally weighted.
run_scores = df.groupby(["run", "policy"])[metrics].mean()

rows = []

for metric in metrics:
    paired = run_scores[metric].unstack("policy")

    for policy in paired.columns:
        if policy == reference:
            continue

        # Positive means candidate is better.
        gains = (paired[reference] - paired[policy]).dropna().to_numpy()

        n = len(gains)

        if n == 0:
            continue

        samples = rng.choice(
            gains,
            size=(BOOTSTRAPS, n),
            replace=True,
        )

        boot_means = samples.mean(axis=1)

        rows.append(
            {
                "metric": metric,
                "candidate": policy,
                "mean_gain": gains.mean(),
                "median_gain": np.median(gains),
                "win_fraction": np.mean(gains > 0),
                "ci_lower": np.quantile(boot_means, 0.025),
                "ci_upper": np.quantile(boot_means, 0.975),
            }
        )

summary = pd.DataFrame(rows)

print("\nPAIRED COMPARISON VS FIXED 0.50")
print(summary.round(4).to_string(index=False))

output = "results/controller_tradeoffs.csv"
summary.to_csv(output, index=False)

print("\nSaved:", output)
