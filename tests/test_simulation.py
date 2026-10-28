
import numpy as np
import pandas as pd
import pytest

from healthcare_ai_lab.simulation.generator import (
    SimulationConfig,
    generate_telemetry,
    get_observable_data,
    EVALUATION_ONLY_COLUMNS,
)


def test_simulation_reproducible():
    config = SimulationConfig(
        n_observations=500,
        seed=12345,
    )

    data_1 = generate_telemetry(config)
    data_2 = generate_telemetry(config)

    pd.testing.assert_frame_equal(data_1, data_2)


def test_simulation_changes_with_seed():
    data_1 = generate_telemetry(
        SimulationConfig(n_observations=500, seed=1)
    )

    data_2 = generate_telemetry(
        SimulationConfig(n_observations=500, seed=2)
    )

    assert not data_1["y"].equals(data_2["y"])


def test_simulation_has_expected_shape():
    data = generate_telemetry(
        SimulationConfig(n_observations=500)
    )

    assert len(data) == 470
    assert not data.isna().any().any()


def test_ground_truth_equation():
    config = SimulationConfig(
        n_observations=500,
        seed=12345,
    )

    data = generate_telemetry(config)

    np.testing.assert_allclose(
        data["y"].to_numpy(),
        data["true_mean"].to_numpy()
        + data["epsilon"].to_numpy(),
    )


def test_hidden_columns_not_exposed():
    data = generate_telemetry(
        SimulationConfig(n_observations=500)
    )

    observable = get_observable_data(data)

    for column in EVALUATION_ONLY_COLUMNS:
        assert column not in observable.columns


def test_invalid_noise_scale_rejected():
    with pytest.raises(ValueError):
        generate_telemetry(
            SimulationConfig(sigma_regime_1=-1)
        )
