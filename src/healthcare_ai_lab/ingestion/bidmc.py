"""BIDMC numerical telemetry ingestion and validation."""

from pathlib import Path
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = {
    "Time [s]": "time_seconds",
    "HR": "heart_rate",
    "PULSE": "pulse_rate",
    "RESP": "respiratory_rate",
    "SpO2": "spo2",
}

MEASUREMENTS = [
    "heart_rate",
    "pulse_rate",
    "respiratory_rate",
    "spo2",
]


def load_bidmc_numerics(path, recording_id):
    """Read and validate one physiological recording."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.rename(columns=EXPECTED_COLUMNS)

    for column in ["time_seconds", *MEASUREMENTS]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df.empty:
        raise ValueError("Recording is empty.")

    if df["time_seconds"].isna().any():
        raise ValueError("Missing timestamps detected.")

    if not df["time_seconds"].is_monotonic_increasing:
        raise ValueError("Timestamps are not ordered.")

    if df["time_seconds"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if (df["time_seconds"].diff().dropna() <= 0).any():
        raise ValueError("Invalid time intervals.")

    # Retain the actual sampling intervals.
    df["sampling_interval_seconds"] = df["time_seconds"].diff()

    # Preserve missingness explicitly.
    for column in MEASUREMENTS:
        df[f"{column}_missing"] = df[column].isna().astype("int8")

    df["any_missing"] = (df[MEASUREMENTS].isna().any(axis=1)).astype("int8")

    df["pulse_spo2_joint_missing"] = (
        df["pulse_rate"].isna() & df["spo2"].isna()
    ).astype("int8")

    # Consecutive missing observations form episodes.
    missing = df["any_missing"].astype(bool)
    new_episode = missing & ~missing.shift(fill_value=False)

    episode_numbers = new_episode.cumsum()

    df["missing_episode_id"] = episode_numbers.where(missing).astype("Int64")

    # Track how long the current gap has persisted.
    df["missing_streak_seconds"] = 0.0

    for episode_id, group in df.loc[missing].groupby("missing_episode_id"):
        times = group["time_seconds"]
        df.loc[group.index, "missing_streak_seconds"] = (
            times - times.iloc[0] + 1.0
        ).to_numpy()

    df.insert(0, "recording_id", str(recording_id))

    return df


def summarize_recording(df):
    """Produce lightweight data-quality metadata."""
    missing = df["any_missing"].eq(1)

    if missing.any():
        episodes = (
            df.loc[missing]
            .groupby("missing_episode_id")
            .agg(
                start_seconds=("time_seconds", "min"),
                end_seconds=("time_seconds", "max"),
                observations=("time_seconds", "size"),
            )
        )
    else:
        episodes = pd.DataFrame(
            columns=[
                "start_seconds",
                "end_seconds",
                "observations",
            ]
        )

    return {
        "recording_id": df["recording_id"].iloc[0],
        "observations": len(df),
        "duration_seconds": float(
            df["time_seconds"].iloc[-1] - df["time_seconds"].iloc[0]
        ),
        "missing_by_column": (df[MEASUREMENTS].isna().sum().to_dict()),
        "missing_episodes": episodes,
    }


def export_parquet(df, path):
    """Write processed telemetry to Parquet."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return output