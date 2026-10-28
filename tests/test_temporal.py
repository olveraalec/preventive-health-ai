import pandas as pd
import pytest

from healthcare_ai_lab.preprocessing.temporal import (
    build_temporal_features,
)


def test_future_measurements_do_not_change_past_features():
    df = pd.DataFrame(
        {
            "recording_id": ["test"] * 6,
            "time_seconds": [0, 1, 2, 3, 4, 5],
            "heart_rate": [80, 81, 82, 83, 84, 85],
            "pulse_rate": [79, 80, 81, 82, 83, 84],
            "respiratory_rate": [18] * 6,
            "spo2": [97] * 6,
        }
    )

    original = build_temporal_features(df, windows=(3,))

    changed = df.copy()
    changed.loc[5, "heart_rate"] = 200
    updated = build_temporal_features(changed, windows=(3,))

    feature = "heart_rate_mean_3"

    pd.testing.assert_series_equal(
        original.loc[:4, feature],
        updated.loc[:4, feature],
    )


def test_missing_measurements_remain_missing():
    df = pd.DataFrame(
        {
            "recording_id": ["test"] * 4,
            "time_seconds": [0, 1, 2, 3],
            "heart_rate": [80, 81, 82, 83],
            "pulse_rate": [80, 81, 82, 83],
            "respiratory_rate": [18, 18, 19, 19],
            "spo2": [97, None, None, 96],
        }
    )

    result = build_temporal_features(df, windows=(3,))

    assert pd.isna(result.loc[1, "spo2"])
    assert result.loc[2, "spo2_missing"] == 1
    assert result.loc[2, "spo2_missing_fraction_3"] == pytest.approx(2 / 3)
