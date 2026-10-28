"""Causal temporal features for physiological telemetry.

Every feature uses only the current and previously observed
measurements. No future data enters the rolling windows.
"""

import numpy as np
import pandas as pd


SIGNALS = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]


def build_temporal_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 15, 30),
) -> pd.DataFrame:
    """Build features independently within each recording."""

    required = {"recording_id", "time_seconds", *SIGNALS}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not windows or any(w < 2 for w in windows):
        raise ValueError("Windows must contain integers >= 2")

    data = df.sort_values(["recording_id", "time_seconds"]).copy()

    if data.duplicated(["recording_id", "time_seconds"]).any():
        raise ValueError("Duplicate recording/timestamp pairs")

    groups = data.groupby("recording_id", sort=False)

    for signal in SIGNALS:
        values = pd.to_numeric(data[signal], errors="coerce")
        data[signal] = values

        # Preserve gaps rather than blindly imputing.
        data[f"{signal}_missing"] = values.isna().astype(int)

        # Change per second, only between adjacent valid samples.
        previous = groups[signal].shift(1)
        previous_time = groups["time_seconds"].shift(1)

        dt = data["time_seconds"] - previous_time
        valid = (dt > 0) & values.notna() & previous.notna()

        rate = (values - previous) / dt.where(valid)
        data[f"{signal}_slope"] = rate

        for window in windows:
            rolling = data.groupby("recording_id", sort=False)[signal].rolling(
                window=window, min_periods=2
            )

            data[f"{signal}_mean_{window}"] = rolling.mean().reset_index(
                level=0, drop=True
            )
            data[f"{signal}_std_{window}"] = rolling.std().reset_index(
                level=0, drop=True
            )

            missing_roll = (
                data.groupby("recording_id", sort=False)[f"{signal}_missing"]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            data[f"{signal}_missing_fraction_{window}"] = missing_roll

    # Cross-signal difference, not a clinical diagnosis.
    data["heart_pulse_difference"] = data["heart_rate"] - data["pulse_rate"]

    data["any_missing"] = data[SIGNALS].isna().any(axis=1).astype(int)

    return data