"""Phase A.8 — Covariance-aware continuous memory allocation.

Conditional error moments are estimated from out-of-fold
training-patient predictions, never validation outcomes.

Exploratory patient-grouped validation; two recordings remain
unused in each run. Reused patient groups are not independent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.evaluation.registry import evaluate


DATA = Path("data/processed/bidmc_batch.parquet")
OUTPUT = Path("results/bidmc_covariance_gate.csv")

TARGET = "target_hr_10s"
RUNS = 10

BASE = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]

MEMORY = [
    "pulse_rate_mean_30",
    "heart_rate_std_15",
    "pulse_rate_std_30",
    "pulse_rate_missing_fraction_15",
]

STATE = [
    "heart_rate_std_15",
    "pulse_rate_std_30",
    "heart_rate",
    "respiratory_rate",
]


def make_model(alpha=10.0):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def train_expert(data, columns):
    m = make_model(10.0)
    m.fit(data[columns], data[TARGET])
    return m


def crossfit(data, policies):
    result = data[[TARGET, "recording_id"] + STATE].copy()
    for name in policies:
        result[name] = np.nan

    for pid in data["recording_id"].unique():
        fit = data[data.recording_id != pid]
        held = data[data.recording_id == pid]

        for name, cols in policies.items():
            m = train_expert(fit, cols)
            result.loc[held.index, name] = m.predict(held[cols])

    return result


class CovarianceGate:
    def fit(self, oof):
        y = oof[TARGET].to_numpy()
        p1 = oof["one"].to_numpy()
        p4 = oof["four"].to_numpy()

        e1 = y - p1
        e4 = y - p4

        # Conditional second-moment targets.
        # Ridge regularization reduces variance in their estimates.
        targets = {
            "v1": e1**2,
            "v4": e4**2,
            "cross": e1 * e4,
        }

        self.models = {}
        self.scales = {}
        self.global_moments = {}

        for name, target in targets.items():
            self.global_moments[name] = float(np.mean(target))

            scale = max(
                float(np.sqrt(np.mean(target**2))),
                1e-6,
            )
            self.scales[name] = scale

            m = make_model(alpha=100.0)
            m.fit(
                oof[STATE],
                np.clip(target / scale, -10, 10),
            )
            self.models[name] = m

        return self

    def weights(self, frame):
        moments = {
            name: self.models[name].predict(frame[STATE]) * self.scales[name]
            for name in self.models
        }

        # Blend noisy conditional moments toward global estimates.
        shrinkage = 0.25

        for name in moments:
            moments[name] = (
                shrinkage * moments[name] + (1 - shrinkage) * self.global_moments[name]
            )

        v1 = np.maximum(moments["v1"], 1e-6)
        v4 = np.maximum(moments["v4"], 1e-6)

        # Enforce a feasible 2x2 covariance/moment matrix.
        cross = np.clip(
            moments["cross"],
            -np.sqrt(v1 * v4),
            np.sqrt(v1 * v4),
        )

        denominator = v1 + v4 - 2 * cross

        w = np.full(len(frame), 0.5, dtype=float)
        valid = denominator > 1e-6
        w[valid] = (v1[valid] - cross[valid]) / denominator[valid]

        return np.clip(w, 0.0, 1.0)


def main():
    df = pd.read_parquet(DATA)
    df = df.sort_values(["recording_id", "time_seconds"]).dropna(subset=[TARGET]).copy()

    flags = [f"{col}_missing" for col in BASE if f"{col}_missing" in df.columns]

    current = BASE + flags
    policies = {
        "one": current + MEMORY[:1],
        "four": current + MEMORY,
    }

    required = set(current + MEMORY + STATE + [TARGET])
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing features: {missing}")

    ids = np.array(sorted(df.recording_id.unique()))
    if len(ids) != 10:
        raise ValueError("Expected 10 patient recordings.")

    rng = np.random.default_rng(12345)
    rows = []
    gate_diagnostics = []

    for run in range(RUNS):
        shuffled = rng.permutation(ids)

        train = df[df.recording_id.isin(shuffled[:6])]
        validation = df[df.recording_id.isin(shuffled[6:8])]

        # Remaining two recordings are unused.
        oof = crossfit(train, policies)
        gate = CovarianceGate().fit(oof)

        experts = {
            name: train_expert(train, features) for name, features in policies.items()
        }

        # One shared predictive scale to isolate mixture behavior.
        sigma = max(
            float(
                np.sqrt(np.mean((oof[TARGET].to_numpy() - oof["four"].to_numpy()) ** 2))
            ),
            1e-6,
        )

        for pid, patient in validation.groupby("recording_id"):
            y = patient[TARGET].to_numpy()

            p1 = experts["one"].predict(patient[policies["one"]])
            p4 = experts["four"].predict(patient[policies["four"]])

            w = gate.weights(patient)

            # Additional SRM-style shrinkage toward equal blending.
            w_shrunk = 0.5 + 0.25 * (w - 0.5)

            predictions = {
                "fixed_one": p1,
                "fixed_four": p4,
                "equal_blend": 0.5 * (p1 + p4),
                "covariance_gate": (p1 + w * (p4 - p1)),
                "covariance_shrunk": (p1 + w_shrunk * (p4 - p1)),
            }

            gate_diagnostics.append(
                {
                    "run": run,
                    "recording_id": str(pid),
                    "mean_weight": float(w.mean()),
                    "weight_std": float(w.std()),
                    "weight_variation": float(np.mean(np.abs(np.diff(w))))
                    if len(w) > 1
                    else 0.0,
                }
            )

            for name, pred in predictions.items():
                scores = evaluate(
                    y,
                    pred,
                    sigma=np.full(len(y), sigma),
                )

                rows.append(
                    {
                        "run": run,
                        "recording_id": str(pid),
                        "policy": name,
                        **scores,
                    }
                )

    results = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    diagnostics = pd.DataFrame(gate_diagnostics)
    diagnostics.to_csv(
        OUTPUT.with_name("bidmc_covariance_gate_diagnostics.csv"),
        index=False,
    )

    metrics = [
        "rmse",
        "mae",
        "bias",
        "p90_absolute_error",
        "nll",
        "crps",
        "coverage_90",
        "width_90",
    ]

    print("\nFULL METRIC REGISTRY")
    print(results.groupby("policy")[metrics].agg(["mean", "std"]).round(4).to_string())

    print("\nPAIRED GAINS AGAINST EQUAL BLEND")
    for metric in ["rmse", "nll", "crps"]:
        paired = results.pivot(
            index=["run", "recording_id"],
            columns="policy",
            values=metric,
        )

        for name in [
            "covariance_gate",
            "covariance_shrunk",
        ]:
            gain = paired["equal_blend"] - paired[name]

            print(
                metric,
                name,
                "mean:",
                round(gain.mean(), 5),
                "median:",
                round(gain.median(), 5),
                "win:",
                round((gain > 0).mean(), 3),
            )

    print("\nCONTROLLER WEIGHT DIAGNOSTICS")
    print(diagnostics.describe().round(4).to_string())

    print("\nWORST RECORDING RMSE")
    print(results.groupby("policy")["rmse"].max().round(4).to_string())

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
