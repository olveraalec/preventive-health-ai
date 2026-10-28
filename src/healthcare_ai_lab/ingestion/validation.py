"""Basic integrity checks for physiological telemetry."""

import numpy as np
import pandas as pd

MEASUREMENTS = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]


def validate_telemetry(df: pd.DataFrame) -> dict:
    required = ["recording_id", "time_seconds", *MEASUREMENTS]
    missing_columns = sorted(set(required) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df.empty:
        raise ValueError("Telemetry dataset is empty")

    # Work in chronological order.
    data = df.sort_values("time_seconds").copy()
    time = data["time_seconds"]

    intervals = time.diff().dropna()

    missing = data[MEASUREMENTS].isna()

    # Diagnostic checks, not clinical judgments.
    summary = {
        "observations": len(data),
        "recordings": int(data["recording_id"].nunique()),
        "duplicate_timestamps": int(time.duplicated().sum()),
        "nonpositive_intervals": int((intervals <= 0).sum()),
        "irregular_intervals": int((~np.isclose(intervals, 1.0)).sum()),
        "missing_counts": missing.sum().to_dict(),
        "any_missing_rows": int(missing.any(axis=1).sum()),
        "pulse_spo2_joint_missing": int(
            (missing["pulse_rate"] & missing["spo2"]).sum()
        ),
    }

    return summary