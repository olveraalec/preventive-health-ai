import pytest

from healthcare_ai_lab.agent.workflow import agent_graph


def test_agent_graph_executes():
    result = agent_graph.invoke(
        {
            "episode_id": "demo_001",
            "step": 0,
            "variance_1": 4.0,
            "variance_2": 1.0,
            "covariance_12": 0.2,
            "previous_weight": 0.5,
            "resource_cost": 1.0,
        }
    )

    assert result["verified"] is True

    assert result["trace"] == [
        "observe",
        "estimate",
        "allocate",
        "execute",
        "verify",
    ]

    assert result["decision"]["weight_1"] + result["decision"][
        "weight_2"
    ] == pytest.approx(1.0)

    assert result["execution"]["status"] == "simulated"


def test_agent_graph_rejects_invalid_moments():
    with pytest.raises(ValueError):
        agent_graph.invoke(
            {
                "episode_id": "demo_invalid",
                "step": 0,
                "variance_1": -1.0,
                "variance_2": 1.0,
                "covariance_12": 0.0,
            }
        )
