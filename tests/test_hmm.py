import numpy as np
import pytest

from healthcare_ai_lab.state_estimation.hmm import (
    HMMConfig,
    GaussianHMMFilter,
)


@pytest.fixture
def hmm():
    return GaussianHMMFilter(
        HMMConfig(
            transition=np.array(
                [
                    [0.95, 0.05],
                    [0.05, 0.95],
                ]
            ),
            means=np.array(
                [
                    [0.0],
                    [3.0],
                ]
            ),
            variances=np.ones((2, 1)),
            initial=np.array([0.5, 0.5]),
        )
    )


def test_beliefs_are_probabilities(hmm):
    for value in [0.0, 0.5, 3.0, 2.5]:
        p = hmm.update([value])
        assert np.isclose(p.sum(), 1)
        assert np.all(p >= 0)


def test_observation_updates_state(hmm):
    p = hmm.update([3.0])
    assert p[1] > p[0]


def test_missing_observation_uses_transition(hmm):
    p = hmm.update([np.nan])
    assert np.allclose(p, [0.5, 0.5])


def test_future_observation_cannot_change_past(hmm):
    first = np.array([[0.0], [0.5], [1.0]])
    extended = np.vstack([first, [[3.0], [3.0]]])

    original = hmm.filter_sequence(first)
    extended_result = hmm.filter_sequence(extended)

    np.testing.assert_allclose(original, extended_result[: len(first)])


def test_reset_restores_prior(hmm):
    hmm.update([3.0])
    hmm.reset()
    np.testing.assert_allclose(hmm.belief, [0.5, 0.5])


def test_invalid_transition_rejected():
    with pytest.raises(ValueError):
        GaussianHMMFilter(
            HMMConfig(
                transition=np.array(
                    [
                        [0.8, 0.8],
                        [0.1, 0.9],
                    ]
                ),
                means=np.array([[0.0], [3.0]]),
                variances=np.ones((2, 1)),
                initial=np.array([0.5, 0.5]),
            )
        )
