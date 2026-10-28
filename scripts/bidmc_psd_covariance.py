"""Phase A.10: PSD conditional error-moment allocation.

Uses BIDMC patient-grouped validation and the full metric registry.
All error moments come from training-patient OOF predictions.
Held-out validation outcomes are used only for evaluation.

Exploratory experiment, not a final independent test.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

from healthcare_ai_lab.evaluation.registry import evaluate


TARGET = "target_hr_10s"
DATA = Path("data/processed/bidmc_batch.parquet")
OUTPUT = Path("results/bidmc_psd_covariance.csv")

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

# Predeclared exploratory parameters.
BANDWIDTHS = [0.5, 1.0, 2.0]
MOMENT_SHRINKAGE = [0.25, 0.75, 1.0]
DECISION_SHRINKAGE = [0.0, 0.5, 0.75]


def model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=10.0),
    )


def crossfit(data, policies):
    oof = data[["recording_id", TARGET] + STATE].copy()

    for name in policies:
        oof[name] = np.nan

    for pid in data.recording_id.unique():
        fit = data[data.recording_id != pid]
        held = data[data.recording_id == pid]

        for name, cols in policies.items():
            m = model()
            m.fit(fit[cols], fit[TARGET])
            oof.loc[held.index, name] = m.predict(held[cols])

    return oof


class PSDGate:
    def fit(self, oof):
        self.transformer = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )

        self.states = self.transformer.fit_transform(oof[STATE])

        y = oof[TARGET].to_numpy()

        self.errors = np.column_stack(
            [
                y - oof["one"].to_numpy(),
                y - oof["four"].to_numpy(),
            ]
        )

        # Outer-product moments: e1², e4², e1*e4.
        self.outer = np.column_stack(
            [
                self.errors[:, 0] ** 2,
                self.errors[:, 1] ** 2,
                self.errors[:, 0] * self.errors[:, 1],
            ]
        )

        self.global_moments = self.outer.mean(axis=0)
        return self

    def local_moments(self, frame, bandwidth):
        z = self.transformer.transform(frame[STATE])

        # Squared Euclidean distance in standardized state space.
        d2 = euclidean_distances(z, self.states, squared=True)
        d2 = np.maximum(d2, 0.0)

        log_weights = -d2 / (2 * bandwidth**2)

        # Numerically stable normalized Gaussian weights.
        log_weights -= log_weights.max(axis=1, keepdims=True)
        weights = np.exp(log_weights)
        weights /= weights.sum(axis=1, keepdims=True)

        # Convex combination of PSD outer products.
        return weights @ self.outer

    @staticmethod
    def allocation(moments, gamma):
        v1 = np.maximum(moments[:, 0], 0.0)
        v4 = np.maximum(moments[:, 1], 0.0)
        cross = moments[:, 2]

        denominator = np.maximum(v1 + v4 - 2 * cross, 0.0)

        w = np.full(len(v1), 0.5)
        valid = denominator > 1e-8

        # Weight allocated to the four-feature specialist.
        w[valid] = (v1[valid] - cross[valid]) / denominator[valid]

        w = np.clip(w, 0.0, 1.0)

        # Conservative decision shrinkage toward equal blend.
        return (1 - gamma) * w + gamma * 0.5

    def predict_weights(self, local, rho, gamma):
        moments = (1 - rho) * local + rho * self.global_moments[None, :]

        return self.allocation(moments, gamma)


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
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ids = np.array(sorted(df.recording_id.unique()))
    if len(ids) != 10:
        raise ValueError("Expected 10 recordings")

    rng = np.random.default_rng(12345)
    rows = []

    for run in range(RUNS):
        shuffled = rng.permutation(ids)

        train = df[df.recording_id.isin(shuffled[:6])]
        validation = df[df.recording_id.isin(shuffled[6:8])]

        # Two recordings unused per run.
        oof = crossfit(train, policies)
        gate = PSDGate().fit(oof)

        experts = {}
        for name, cols in policies.items():
            m = model()
            m.fit(train[cols], train[TARGET])
            experts[name] = m

        # Shared scale for fair probabilistic scoring.
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

            def record(name, pred, weight, bandwidth, rho, gamma):
                scores = evaluate(y, pred, sigma=np.full(len(y), sigma))
                rows.append(
                    {
                        "run": run,
                        "recording_id": str(pid),
                        "policy": name,
                        "bandwidth": bandwidth,
                        "rho": rho,
                        "gamma": gamma,
                        "mean_weight": float(np.mean(weight)),
                        "weight_std": float(np.std(weight)),
                        "weight_variation": (
                            float(np.abs(np.diff(weight)).mean())
                            if len(weight) > 1
                            else 0.0
                        ),
                        **scores,
                    }
                )

            record(
                "equal_blend",
                0.5 * (p1 + p4),
                np.full(len(y), 0.5),
                np.nan,
                np.nan,
                np.nan,
            )
            record("fixed_four", p4, np.ones(len(y)), np.nan, np.nan, np.nan)

            for bandwidth in BANDWIDTHS:
                local = gate.local_moments(patient, bandwidth)

                for rho in MOMENT_SHRINKAGE:
                    for gamma in DECISION_SHRINKAGE:
                        w = gate.predict_weights(local, rho, gamma)

                        pred = p1 + w * (p4 - p1)

                        record("psd_gate", pred, w, bandwidth, rho, gamma)

    results = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

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

    summary = results.groupby(["policy", "bandwidth", "rho", "gamma"], dropna=False)[
        metrics
    ].agg(["mean", "std"])

    print("\nFULL METRIC REGISTRY")
    print(summary.round(4).to_string())

    gate_results = results[results["policy"] == "psd_gate"]

    for metric in ["rmse", "nll", "crps"]:
        print(f"\n{metric.upper()} GRID")
        table = gate_results.pivot_table(
            index=["bandwidth", "rho"],
            columns="gamma",
            values=metric,
            aggfunc="mean",
        )
        print(table.round(5).to_string())

    # The next tables are descriptive, not test-set selection.
    print("\nEQUAL BLEND REFERENCE")
    print(results[results.policy == "equal_blend"][metrics].mean().round(5).to_string())

    print("\nPSD CONTROLLER WEIGHT DIAGNOSTICS")
    print(
        gate_results[["mean_weight", "weight_std", "weight_variation"]]
        .describe()
        .round(4)
        .to_string()
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
