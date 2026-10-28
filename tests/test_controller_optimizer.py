import numpy as np
import pytest

from healthcare_ai_lab.controller.optimizer import (
    ControllerConfig,
    project_covariance,
    optimize_weight,
    objective_terms,
    realized_objective,
)


def test_psd_projection():
    v1, v4, cross = project_covariance(1.0, 1.0, 2.0)
    matrix = np.array(
        [
            [v1, cross],
            [cross, v4],
        ]
    )

    assert np.linalg.eigvalsh(matrix).min() >= -1e-10


def test_equal_error_variances():
    config = ControllerConfig(
        complexity_penalty=0,
        stability_penalty=0,
    )
    result = optimize_weight(
        (1.0, 1.0, 0.0),
        previous_weight=0.5,
        config=config,
    )
    assert result.weight == pytest.approx(0.5)


def test_lower_variance_expert_preferred():
    config = ControllerConfig(
        complexity_penalty=0,
        stability_penalty=0,
    )
    result = optimize_weight(
        (4.0, 1.0, 0.0),
        previous_weight=0.5,
        config=config,
    )
    assert result.weight > 0.5


def test_stability_penalty_reduces_change():
    moments = (4.0, 1.0, 0.0)

    free = optimize_weight(
        moments,
        0.1,
        ControllerConfig(0, 0, 0),
    )
    stable = optimize_weight(
        moments,
        0.1,
        ControllerConfig(0, 10, 0),
    )

    assert abs(stable.weight - 0.1) < abs(free.weight - 0.1)


def test_memory_penalty_reduces_allocation():
    moments = (4.0, 1.0, 0.0)

    free = optimize_weight(
        moments,
        0.5,
        ControllerConfig(0, 0, 0),
    )
    costly = optimize_weight(
        moments,
        0.5,
        ControllerConfig(0, 0, 2),
    )

    assert costly.weight < free.weight


def test_matches_numerical_minimum():
    config = ControllerConfig(
        complexity_penalty=0.5,
        stability_penalty=0.5,
        memory_penalty=0.1,
    )
    moments = project_covariance(3.0, 2.0, 0.4)

    selected = optimize_weight(moments, 0.3, config)

    grid = np.linspace(0, 1, 1001)
    values = [objective_terms(w, 0.3, moments, config)["total"] for w in grid]

    assert selected.estimated_objective <= min(values) + 1e-6


def test_realized_objective_decomposition():
    config = ControllerConfig(
        complexity_penalty=0.5,
        stability_penalty=0.25,
        memory_penalty=0.1,
    )
    parts = realized_objective(
        y_true=[1, 2, 3],
        y_pred=[1.1, 2.2, 2.8],
        weights=[0.4, 0.5, 0.6],
        config=config,
    )

    assert parts["total"] == pytest.approx(
        sum(value for key, value in parts.items() if key != "total")
    )
    assert parts["total"] >= parts["realized_mse"]
