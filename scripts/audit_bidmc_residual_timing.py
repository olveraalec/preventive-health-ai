"""Audit target alignment and causal feedback availability."""

import numpy as np
import pandas as pd

PATH = "data/processed/bidmc_batch.parquet"
HORIZON = 10
TARGET = "target_hr_10s"

df = pd.read_parquet(PATH)
df = df.sort_values(["recording_id", "time_seconds"]).reset_index(drop=True)

required = {"recording_id", "time_seconds", "heart_rate", TARGET}
assert required.issubset(df.columns)

groups = df.groupby("recording_id", sort=False)
dt = groups["time_seconds"].diff()

print("\n1. SAMPLING INTERVALS")
print(dt.dropna().value_counts().head(10).to_string())

regular = dt.dropna().eq(1).all()
print("All intervals exactly 1 second:", regular)

assert not df.duplicated(["recording_id", "time_seconds"]).any(), (
    "Duplicate recording timestamps"
)

# Verify that the future target really is HR at t+10.
expected_target = groups["heart_rate"].shift(-HORIZON)

mask = df[TARGET].notna() & expected_target.notna()

alignment_difference = (df.loc[mask, TARGET] - expected_target.loc[mask]).abs()

print("\n2. TARGET ALIGNMENT")
print("Compared observations:", int(mask.sum()))
print(
    "Maximum difference:",
    float(alignment_difference.max()) if mask.any() else np.nan,
)

target_missing_mismatch = (df[TARGET].isna() != expected_target.isna()).sum()
print("Target missingness mismatches:", target_missing_mismatch)

# Reconstruct the latest completed forecast outcome.
# At time t, the forecast generated at t-10 has matured.
previous_target = groups[TARGET].shift(HORIZON)

available_observation = df["heart_rate"]

valid = previous_target.notna() & available_observation.notna()

availability_difference = (
    previous_target.loc[valid] - available_observation.loc[valid]
).abs()

print("\n3. RESIDUAL OUTCOME AVAILABILITY")
print("Compared observations:", int(valid.sum()))
print(
    "Maximum reconstruction difference:",
    float(availability_difference.max()) if valid.any() else np.nan,
)

# Check actual timestamp separation, not merely row separation.
previous_time = groups["time_seconds"].shift(HORIZON)

elapsed = df["time_seconds"] - previous_time

timing_valid = elapsed.loc[valid].ge(HORIZON).all()

print(
    "All compared feedback at least 10 seconds old:",
    bool(timing_valid),
)

# A non-null shifted target at t should represent an outcome
# that occurred no later than the current timestamp.
target_available = previous_time + HORIZON <= df["time_seconds"]

future_leak_count = int((previous_target.notna() & ~target_available).sum())

print("\n4. LEAKAGE CHECKS")
print("Future-outcome timing violations:", future_leak_count)
print(
    "Recording count:",
    df["recording_id"].nunique(),
)

print("\n5. FINAL AUDIT")
passed = (
    regular
    and mask.any()
    and np.isclose(alignment_difference.max(), 0)
    and target_missing_mismatch == 0
    and valid.any()
    and np.isclose(availability_difference.max(), 0)
    and timing_valid
    and future_leak_count == 0
)

print("PASS" if passed else "REQUIRES INVESTIGATION")

if not passed:
    raise AssertionError("Causal alignment audit failed; inspect the output.")
