"""Learned continuous residual-feedback controller.

Disjoint recording groups for baseline, correction, gate,
validation, and test. No test-set model selection.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

SEED = 12345
RUNS = 30
HORIZON = 10
TARGET = "target_hr_10s"

BASE = ["heart_rate", "pulse_rate", "respiratory_rate", "spo2"]
HISTORY = ["error_lag_10", "mean_error_12", "energy_12"]

GATE = [
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "mean_error_12",
    "energy_12",
]

ALPHAS = {
    "baseline": 0.0,
    "fixed_025": 0.25,
    "fixed_050": 0.50,
}


def make_model(alpha=10.0):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def prepare(data, base_model):
    blocks = []

    for _, part in data.groupby("recording_id"):
        part = part.sort_values("time_seconds").copy()
        part["baseline"] = base_model.predict(part[BASE])
        part["error"] = part[TARGET] - part["baseline"]

        # Only use residuals whose outcomes have matured.
        available = part["error"].shift(HORIZON)

        part["error_lag_10"] = available
        part["mean_error_12"] = available.rolling(12, min_periods=6).mean()
        part["energy_12"] = available.pow(2).rolling(12, min_periods=6).mean()

        blocks.append(part.dropna(subset=[TARGET] + HISTORY))

    return pd.concat(blocks, ignore_index=True)


def gate_features(data, center, scale):
    z = data[GATE].copy()
    z = z.fillna(center)
    return (z - center) / scale


def learned_weight(data, correction, coefficients, center, scale, cap):
    z = gate_features(data, center, scale).to_numpy()

    # Linear state-conditioned weight, regularized toward 0.25.
    raw = 0.25 + z @ coefficients

    return np.clip(raw, 0.0, cap)


def summarize(y, pred):
    error = y - pred
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def probabilistic_metrics(y, pred, sigma):
    """Gaussian predictive scores with fixed calibrated sigma."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    sigma = max(float(sigma), 1e-6)

    z = (y - pred) / sigma

    nll = 0.5 * (np.log(2 * np.pi * sigma**2) + z**2)

    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

    lower = pred - norm.ppf(0.95) * sigma
    upper = pred + norm.ppf(0.95) * sigma

    return {
        "nll": float(np.mean(nll)),
        "crps": float(np.mean(crps)),
        "coverage_90": float(np.mean((y >= lower) & (y <= upper))),
        "width_90": float(np.mean(upper - lower)),
        "p90_absolute_error": float(np.quantile(np.abs(y - pred), 0.90)),
    }


def alpha_variation(weights, time):
    """Measure weight movement in chronological order."""
    order = np.argsort(np.asarray(time))
    sorted_weights = np.asarray(weights)[order]

    if len(sorted_weights) < 2:
        return 0.0

    return float(np.mean(np.abs(np.diff(sorted_weights))))


df = pd.read_parquet("data/processed/bidmc_batch.parquet")
ids = np.array(sorted(df.recording_id.unique()))
rng = np.random.default_rng(SEED)

rows = []

for run in range(RUNS):
    ids_run = rng.permutation(ids)

    base_ids = ids_run[:3]
    correction_ids = ids_run[3:5]
    gate_ids = ids_run[5:6]
    val_ids = ids_run[6:8]
    test_ids = ids_run[8:10]

    def select(recording_ids):
        return df[df.recording_id.isin(recording_ids)].copy()

    base_train = select(base_ids).dropna(subset=[TARGET])
    base_model = make_model()
    base_model.fit(base_train[BASE], base_train[TARGET])

    correction_train = prepare(select(correction_ids), base_model)
    gate_train = prepare(select(gate_ids), base_model)
    validation = prepare(select(val_ids), base_model)
    test = prepare(select(test_ids), base_model)

    correction_model = make_model(alpha=100.0)
    correction_model.fit(
        correction_train[HISTORY],
        correction_train["error"],
    )

    gate_correction = correction_model.predict(gate_train[HISTORY])

    # Gate normalization learned only from gate-training data.
    center = gate_train[GATE].median()
    scale = gate_train[GATE].std().clip(lower=1e-6)

    z_gate = gate_features(gate_train, center, scale).to_numpy()

    # Learn how deviations from fixed 0.25 should affect
    # the residual correction. Penalize deviations from
    # that reference, equivalent to ridge regularization.
    design = z_gate * gate_correction[:, None]
    desired = gate_train["error"].to_numpy() - 0.25 * gate_correction

    # Intercept is fixed at 0.25; only state effects are fitted.
    penalty = 100.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty * np.eye(design.shape[1]),
        design.T @ desired,
    )

    # Calibration uses validation recordings only.
    # The final test outcomes never estimate uncertainty.
    calibration_sigma = {}

    for name, fixed_alpha in ALPHAS.items():
        val_corr = correction_model.predict(validation[HISTORY])
        val_pred = validation["baseline"].to_numpy() + fixed_alpha * val_corr

        calibration_sigma[name] = max(
            float(np.sqrt(np.mean((validation[TARGET].to_numpy() - val_pred) ** 2))),
            1e-6,
        )

    val_corr = correction_model.predict(validation[HISTORY])
    val_weights = learned_weight(
        validation,
        val_corr,
        coefficients,
        center,
        scale,
        cap=0.50,
    )

    val_pred = validation["baseline"].to_numpy() + val_weights * val_corr

    calibration_sigma["learned_bounded"] = max(
        float(np.sqrt(np.mean((validation[TARGET].to_numpy() - val_pred) ** 2))),
        1e-6,
    )

    for split, data in [
        ("validation", validation),
        ("test", test),
    ]:
        correction = correction_model.predict(data[HISTORY])
        base = data["baseline"].to_numpy()
        y = data[TARGET].to_numpy()

        policies = {name: np.full(len(data), alpha) for name, alpha in ALPHAS.items()}

        policies["learned_bounded"] = learned_weight(
            data,
            correction,
            coefficients,
            center,
            scale,
            cap=0.50,
        )

        for name, weights in policies.items():
            pred = base + weights * correction
            recording_array = data["recording_id"].to_numpy()

            for recording_id in data["recording_id"].unique():
                mask = recording_array == recording_id

                record = {
                    "run": run,
                    "split": split,
                    "recording_id": recording_id,
                    "policy": name,
                    **summarize(y[mask], pred[mask]),
                    "mean_alpha": float(weights[mask].mean()),
                    "alpha_std": float(weights[mask].std()),
                    "weight_variation": alpha_variation(
                        weights[mask],
                        data.loc[mask, "time_seconds"],
                    ),
                }

                # Validation outcomes calibrated sigma;
                # report probabilistic scores only on test.
                if split == "test":
                    record.update(
                        probabilistic_metrics(
                            y[mask],
                            pred[mask],
                            calibration_sigma[name],
                        )
                    )

                rows.append(record)


results = pd.DataFrame(rows)

output = Path("results/bidmc_learned_controller.csv")
output.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output, index=False)

print("\nLEARNED CONTROLLER: HELD-OUT TEST")
test = results[results.split == "test"]

print(
    test.groupby("policy")[["rmse", "mae", "mean_alpha", "alpha_std"]]
    .agg(["mean", "std"])
    .round(4)
    .to_string()
)

paired = test.groupby(["run", "policy"])["rmse"].mean().unstack()

gain = paired["fixed_025"] - paired["learned_bounded"]

print("\nPAIRED LEARNED VS FIXED 0.25")
print("Mean gain:", round(gain.mean(), 4))
print("Median gain:", round(gain.median(), 4))
print("Win fraction:", round((gain > 0).mean(), 3))
print("Gain std:", round(gain.std(), 4))

print("\nSaved:", output)


print("\nFULL HELD-OUT TEST METRIC SUITE")

test = results[results["split"] == "test"]

metrics = [
    "rmse",
    "mae",
    "bias",
    "nll",
    "crps",
    "coverage_90",
    "width_90",
    "p90_absolute_error",
    "mean_alpha",
    "alpha_std",
    "weight_variation",
]

print(test.groupby("policy")[metrics].mean().round(4).to_string())

print("\nWORST-RECORDING RMSE")

worst = (
    test.groupby(["run", "policy"])["rmse"].max().groupby("policy").agg(["mean", "std"])
)

print(worst.round(4).to_string())

print("\nPAIRED RMSE IMPROVEMENT VS FIXED 0.25")

paired = test.groupby(["run", "policy"])["rmse"].mean().unstack()

for policy in paired.columns:
    if policy == "fixed_025":
        continue

    gain = paired["fixed_025"] - paired[policy]

    print(
        f"{policy}: "
        f"mean={gain.mean():.4f}, "
        f"median={gain.median():.4f}, "
        f"win_rate={(gain > 0).mean():.3f}"
    )
