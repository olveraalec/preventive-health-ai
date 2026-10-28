"""Reliability-aware continuous feedback with multi-objective evaluation.

Patient-grouped repeated splits.
Causal 10-second residual feedback.
Validation-only policy selection and variance calibration.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 12345
RUNS = 30
HORIZON = 10
TARGET = "target_hr_10s"

BASE = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]

FEEDBACK = [
    "past_error",
    "mean_error",
    "error_energy",
]

POLICIES = [
    "baseline",
    "fixed_025",
    "magnitude",
    "reliability",
    "reliability_smooth",
]


def model(alpha=10.0):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def prepare(data, base_model):
    data = data.sort_values(["recording_id", "time_seconds"]).copy()

    data["base_prediction"] = base_model.predict(data[BASE])

    # A prediction for time t+10 becomes available no earlier
    # than time t+10. Shift by 10 one-second samples.
    data["base_error"] = data[TARGET] - data["base_prediction"]

    group = data.groupby("recording_id", sort=False)

    def history(s):
        available = s.shift(HORIZON)
        return pd.DataFrame(
            {
                "past_error": available,
                "mean_error": available.rolling(12, min_periods=6).mean(),
                "error_energy": available.pow(2).rolling(12, min_periods=6).mean(),
            },
            index=s.index,
        )

    parts = [history(g["base_error"]) for _, g in group]

    history_df = pd.concat(parts).sort_index()

    for col in FEEDBACK:
        data[col] = history_df[col]

    data["missing_fraction"] = data[BASE].isna().mean(axis=1)

    data["hr_volatility"] = group["heart_rate"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=6).std()
    )

    return data.dropna(subset=[TARGET] + FEEDBACK)


def gaussian_crps(y, mu, sigma):
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def policy_predictions(data, feedback_model, scale):
    correction = feedback_model.predict(data[FEEDBACK])

    energy = np.sqrt(np.maximum(data["error_energy"].to_numpy(), 0))

    volatility = data["hr_volatility"].fillna(0).to_numpy()

    missing = data["missing_fraction"].to_numpy()

    # Predeclared heuristic reliability indicators.
    # These are NOT calibrated uncertainty probabilities.
    quality = 1.0 / (1.0 + energy / scale) / (1.0 + volatility) * (1.0 - missing)

    alpha = {
        "baseline": np.zeros(len(data)),
        "fixed_025": np.full(len(data), 0.25),
        "magnitude": 0.4 / (1.0 + np.abs(correction) / scale),
        "reliability": 0.5 * quality,
    }

    # Causal smoothing: current quality and previous quality.
    smooth_quality = (
        pd.Series(quality, index=data.index)
        .groupby(data["recording_id"])
        .transform(lambda s: s.ewm(alpha=0.25, adjust=False).mean())
    )

    alpha["reliability_smooth"] = 0.5 * smooth_quality.to_numpy()

    base = data["base_prediction"].to_numpy()

    return {
        name: (base + weight * correction, weight) for name, weight in alpha.items()
    }


def evaluate(y, prediction, weights, sigma):
    errors = y - prediction

    return {
        "rmse": np.sqrt(np.mean(errors**2)),
        "mae": np.mean(np.abs(errors)),
        "nll": np.mean(norm.logpdf(y, loc=prediction, scale=sigma) * -1),
        "crps": np.mean(gaussian_crps(y, prediction, sigma)),
        "coverage_90": np.mean(np.abs(errors) <= norm.ppf(0.95) * sigma),
        "mean_alpha": np.mean(weights),
        "alpha_std": np.std(weights),
    }


df = pd.read_parquet("data/processed/bidmc_batch.parquet")

ids = np.array(sorted(df["recording_id"].unique()))
rng = np.random.default_rng(SEED)

rows = []

for run in range(RUNS):
    permutation = rng.permutation(ids)

    base_ids = permutation[:4]
    feedback_ids = permutation[4:6]
    val_ids = permutation[6:8]
    test_ids = permutation[8:10]

    base_train = df[df.recording_id.isin(base_ids)].dropna(subset=[TARGET])

    base_model = model()
    base_model.fit(base_train[BASE], base_train[TARGET])

    feedback = prepare(
        df[df.recording_id.isin(feedback_ids)],
        base_model,
    )

    validation = prepare(
        df[df.recording_id.isin(val_ids)],
        base_model,
    )

    test = prepare(
        df[df.recording_id.isin(test_ids)],
        base_model,
    )

    feedback_model = model(alpha=100.0)
    feedback_model.fit(
        feedback[FEEDBACK],
        feedback["base_error"],
    )

    scale = max(float(feedback["base_error"].std()), 1e-6)

    val_predictions = policy_predictions(validation, feedback_model, scale)

    test_predictions = policy_predictions(test, feedback_model, scale)

    # Calibrate one Gaussian sigma per policy on validation.
    # Test labels are never used for this step.
    sigmas = {}

    for name in POLICIES:
        pred, _ = val_predictions[name]
        residuals = validation[TARGET].to_numpy() - pred

        sigmas[name] = max(
            float(np.sqrt(np.mean(residuals**2))),
            1e-4,
        )

    for name in POLICIES:
        for split_name, data, predictions in [
            ("validation", validation, val_predictions),
            ("test", test, test_predictions),
        ]:
            pred, weights = predictions[name]

            for recording_id, part in data.groupby("recording_id", sort=False):
                mask = data["recording_id"].to_numpy() == recording_id

                scores = evaluate(
                    part[TARGET].to_numpy(),
                    pred[mask],
                    weights[mask],
                    sigmas[name],
                )

                # Weight stability is only computed within recording.
                scores["weight_variation"] = (
                    float(np.mean(np.abs(np.diff(weights[mask]))))
                    if mask.sum() > 1
                    else 0.0
                )

                rows.append(
                    {
                        "run": run,
                        "split": split_name,
                        "recording_id": recording_id,
                        "policy": name,
                        **scores,
                    }
                )


results = pd.DataFrame(rows)

output = Path("results/bidmc_multiobjective.csv")
output.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output, index=False)

test = results[results["split"] == "test"]

print("\nMULTI-OBJECTIVE TEST COMPARISON")

print(
    test.groupby("policy")[
        [
            "rmse",
            "mae",
            "nll",
            "crps",
            "coverage_90",
            "weight_variation",
        ]
    ]
    .mean()
    .round(4)
    .to_string()
)

print("\nWORST RECORDING RMSE PER RUN")

worst = (
    test.groupby(["run", "policy"])["rmse"].max().groupby("policy").mean().sort_values()
)

print(worst.round(4).to_string())

print("\nPAIRED RMSE IMPROVEMENTS VS FIXED 0.25")

paired = test.groupby(["run", "policy"])["rmse"].mean().unstack("policy")

for name in POLICIES:
    if name == "fixed_025":
        continue

    gain = paired["fixed_025"] - paired[name]

    print(f"{name}: mean={gain.mean():.4f}, win_rate={(gain > 0).mean():.3f}")

print(f"\nSaved: {output}")
