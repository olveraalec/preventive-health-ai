"""Phase A.9 — Two-dimensional covariance regularization.

Training-patient OOF moments; patient-grouped validation.
No evaluation-driven hyperparameter selection in this script.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_ai_lab.evaluation.registry import evaluate

TARGET = "target_hr_10s"
BASE = ["heart_rate", "pulse_rate", "respiratory_rate", "spo2"]
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

# Predeclared grid; not tuned on validation.
MOMENT_SHRINKAGE = [0.0, 0.25, 0.5, 0.75, 1.0]
DECISION_SHRINKAGE = [0.0, 0.25, 0.5, 0.75, 1.0]
RUNS = 10


def model(alpha=10.0):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def expert_fit(data, features):
    m = model()
    m.fit(data[features], data[TARGET])
    return m


def crossfit(data, policies):
    oof = data[["recording_id", TARGET] + STATE].copy()

    for name in policies:
        oof[name] = np.nan

    for pid in data.recording_id.unique():
        train = data[data.recording_id != pid]
        held = data[data.recording_id == pid]

        for name, cols in policies.items():
            m = expert_fit(train, cols)
            oof.loc[held.index, name] = m.predict(held[cols])

    return oof


class MomentGate:
    def fit(self, oof):
        y = oof[TARGET].to_numpy()
        e1 = y - oof["one"].to_numpy()
        e4 = y - oof["four"].to_numpy()

        raw = np.column_stack([e1**2, e4**2, e1 * e4])

        self.global_moments = raw.mean(axis=0)
        self.models = []
        self.scales = []

        for j in range(3):
            scale = max(
                float(np.sqrt(np.mean(raw[:, j] ** 2))),
                1e-6,
            )
            m = model(alpha=100.0)
            m.fit(
                oof[STATE],
                np.clip(raw[:, j] / scale, -10, 10),
            )
            self.models.append(m)
            self.scales.append(scale)

        return self

    def weight(self, frame, rho, gamma):
        conditional = np.column_stack(
            [
                m.predict(frame[STATE]) * scale
                for m, scale in zip(self.models, self.scales)
            ]
        )

        moments = (1 - rho) * conditional + rho * self.global_moments

        v1 = np.maximum(moments[:, 0], 1e-6)
        v4 = np.maximum(moments[:, 1], 1e-6)

        c = np.clip(
            moments[:, 2],
            -np.sqrt(v1 * v4),
            np.sqrt(v1 * v4),
        )

        denominator = v1 + v4 - 2 * c
        optimal = np.full(len(frame), 0.5)

        valid = denominator > 1e-6
        optimal[valid] = (v1[valid] - c[valid]) / denominator[valid]
        optimal = np.clip(optimal, 0, 1)

        return (1 - gamma) * optimal + gamma * 0.5


def main():
    df = pd.read_parquet("data/processed/bidmc_batch.parquet")
    df = df.sort_values(["recording_id", "time_seconds"]).dropna(subset=[TARGET]).copy()

    flags = [f"{c}_missing" for c in BASE if f"{c}_missing" in df.columns]
    current = BASE + flags

    policies = {
        "one": current + MEMORY[:1],
        "four": current + MEMORY,
    }

    required = set(current + MEMORY + STATE + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ids = np.array(sorted(df.recording_id.unique()))
    if len(ids) != 10:
        raise ValueError("Expected 10 recordings")

    rng = np.random.default_rng(12345)
    results = []

    for run in range(RUNS):
        shuffled = rng.permutation(ids)

        train = df[df.recording_id.isin(shuffled[:6])]
        val = df[df.recording_id.isin(shuffled[6:8])]
        # shuffled[8:] remains unused.

        oof = crossfit(train, policies)
        gate = MomentGate().fit(oof)

        experts = {
            name: expert_fit(train, features) for name, features in policies.items()
        }

        # Shared OOF sigma for comparable probabilistic scores.
        sigma = max(float(np.sqrt(np.mean((oof[TARGET] - oof["four"]) ** 2))), 1e-6)

        for pid, patient in val.groupby("recording_id"):
            y = patient[TARGET].to_numpy()
            p1 = experts["one"].predict(patient[policies["one"]])
            p4 = experts["four"].predict(patient[policies["four"]])

            for rho in MOMENT_SHRINKAGE:
                for gamma in DECISION_SHRINKAGE:
                    w = gate.weight(patient, rho, gamma)
                    pred = p1 + w * (p4 - p1)

                    scores = evaluate(
                        y,
                        pred,
                        sigma=np.full(len(y), sigma),
                    )

                    results.append(
                        {
                            "run": run,
                            "recording_id": str(pid),
                            "rho": rho,
                            "gamma": gamma,
                            "mean_weight": float(w.mean()),
                            "weight_std": float(w.std()),
                            "weight_variation": (
                                float(np.abs(np.diff(w)).mean()) if len(w) > 1 else 0.0
                            ),
                            **scores,
                        }
                    )

    result = pd.DataFrame(results)
    path = Path("results/bidmc_covariance_regularization.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)

    metrics = [
        "rmse",
        "mae",
        "bias",
        "p90_absolute_error",
        "nll",
        "crps",
        "coverage_90",
        "width_90",
        "mean_weight",
        "weight_std",
        "weight_variation",
    ]

    print("\nFULL METRIC REGISTRY")
    print(
        result.groupby(["rho", "gamma"])[metrics]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    print("\nRMSE BY MOMENT / DECISION SHRINKAGE")
    print(
        result.pivot_table(
            index="rho",
            columns="gamma",
            values="rmse",
            aggfunc="mean",
        )
        .round(5)
        .to_string()
    )

    print("\nNLL BY MOMENT / DECISION SHRINKAGE")
    print(
        result.pivot_table(
            index="rho",
            columns="gamma",
            values="nll",
            aggfunc="mean",
        )
        .round(5)
        .to_string()
    )

    print("\nCRPS BY MOMENT / DECISION SHRINKAGE")
    print(
        result.pivot_table(
            index="rho",
            columns="gamma",
            values="crps",
            aggfunc="mean",
        )
        .round(5)
        .to_string()
    )

    print("\nSaved:", path)


if __name__ == "__main__":
    main()
