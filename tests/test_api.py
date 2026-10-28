"""Integration tests for the REST API and LangGraph workflow."""

import pytest
from fastapi.testclient import TestClient

from healthcare_ai_lab.api.main import app


client = TestClient(app)


@pytest.fixture
def valid_request():
    return {
        "episode_id": "api_test",
        "step": 0,
        "variance_1": 4.0,
        "variance_2": 1.0,
        "covariance_12": 0.2,
        "previous_weight": 0.5,
        "resource_cost": 1.0,
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness():
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["inference_enabled"] is False


def test_controller_endpoint(valid_request):
    response = client.post(
        "/api/v1/controller/decide",
        json=valid_request,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["weight_1"] >= 0
    assert data["weight_2"] >= 0
    assert (data["weight_1"] + data["weight_2"]) == pytest.approx(1.0)


def test_agent_memory_selection(valid_request):
    payload = {
        **valid_request,
        "memory_budget": 2,
        "memory_candidates": [
            {
                "memory_id": "recent_hr",
                "pathway": 1,
                "information_score": 0.8,
            },
            {
                "memory_id": "historical_trend",
                "pathway": 2,
                "information_score": 0.9,
            },
            {
                "memory_id": "historical_context",
                "pathway": 2,
                "information_score": 0.7,
            },
        ],
    }

    response = client.post(
        "/api/v1/agent/run",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["verified"] is True
    assert data["execution"]["used"] == 2
    assert data["execution"]["status"] == "selected"

    assert data["execution"]["selected_memory_ids"] == [
        "historical_trend",
        "historical_context",
    ]

    assert data["trace"] == [
        "observe",
        "estimate",
        "allocate",
        "execute",
        "verify",
    ]


def test_invalid_variance_rejected(valid_request):
    payload = {
        **valid_request,
        "variance_1": -5.0,
    }

    response = client.post(
        "/api/v1/controller/decide",
        json=payload,
    )

    assert response.status_code == 422


def test_negative_budget_rejected(valid_request):
    payload = {
        **valid_request,
        "memory_budget": -1,
    }

    response = client.post(
        "/api/v1/agent/run",
        json=payload,
    )

    assert response.status_code == 422


def test_duplicate_memory_rejected(valid_request):
    payload = {
        **valid_request,
        "memory_candidates": [
            {
                "memory_id": "duplicate",
                "pathway": 1,
                "information_score": 0.5,
            },
            {
                "memory_id": "duplicate",
                "pathway": 2,
                "information_score": 0.8,
            },
        ],
    }

    response = client.post(
        "/api/v1/agent/run",
        json=payload,
    )

    assert response.status_code == 422


from unittest.mock import patch


def test_agent_database_memory_retrieval(valid_request):
    from unittest.mock import patch

    candidates = [
        {
            "memory_id": "recent_hr",
            "pathway": 1,
            "information_score": 0.8,
        },
        {
            "memory_id": "historical_trend",
            "pathway": 2,
            "information_score": 0.9,
        },
        {
            "memory_id": "historical_context",
            "pathway": 2,
            "information_score": 0.7,
        },
    ]

    with (
        patch(
            "healthcare_ai_lab.api.main.get_episode_memories",
            return_value=candidates,
        ),
        patch("healthcare_ai_lab.api.main.save_agent_run_with_selections") as save_mock,
    ):
        response = client.post(
            "/api/v1/agent/run-from-memory",
            json={
                **valid_request,
                "episode_id": "memory_database_demo",
                "memory_budget": 2,
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["verified"] is True
    assert data["persisted"] is True
    assert data["execution"]["used"] == 2
    assert data["execution"]["selected_memory_ids"] == [
        "historical_trend",
        "historical_context",
    ]

    save_mock.assert_called_once()

    saved_run, saved_selections = save_mock.call_args.args

    assert saved_run["run_id"] == data["run_id"]
    assert [item["memory_id"] for item in saved_selections] == [
        "historical_trend",
        "historical_context",
    ]
