"""Prepare BIDMC numerics across multiple recordings."""

from pathlib import Path

import pandas as pd

from healthcare_ai_lab.ingestion.bidmc import (
    load_bidmc_numerics,
)
from healthcare_ai_lab.preprocessing.temporal import (
    build_temporal_features,
)


RAW = Path("data/raw/bidmc")
OUTPUT = Path("data/processed/bidmc_batch.parquet")


def main():
    recordings = []

    for path in sorted(RAW.glob("bidmc_*_Numerics.csv")):
        recording_id = path.name.replace("_Numerics.csv", "")

        df = load_bidmc_numerics(
            path,
            recording_id=recording_id,
        )
        recordings.append(df)

    if not recordings:
        raise FileNotFoundError("No BIDMC recordings found")

    combined = pd.concat(recordings, ignore_index=True)

    features = build_temporal_features(combined)

    # Predict heart rate ten seconds in the future.
    # The target must never appear in input features.
    features["target_hr_10s"] = features.groupby("recording_id")["heart_rate"].shift(
        -10
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT, index=False)

    print("Recordings:", features["recording_id"].nunique())
    print("Shape:", features.shape)
    print("Missing future targets:", features["target_hr_10s"].isna().sum())
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
