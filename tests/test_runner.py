import pytest

from healthcare_ai_lab.experiments.runner import (
    ExperimentConfig,
    run_experiment,
)


def test_experiment_reproducible():
    config = ExperimentConfig(
        seeds=(12345,),
        n_observations=500,
    )

    first = run_experiment(config)
    second = run_experiment(config)

    assert first == second


def test_experiment_returns_both_baselines():
    results = run_experiment(
        ExperimentConfig(
            seeds=(12345,),
            n_observations=500,
        )
    )

    assert {r["model"] for r in results} == {
        "current_only",
        "historical_lags",
    }


def test_invalid_split_rejected():
    config = ExperimentConfig(
        train_fraction=0.9,
        validation_fraction=0.2,
        n_observations=500,
    )

    with pytest.raises(ValueError):
        run_experiment(config)
