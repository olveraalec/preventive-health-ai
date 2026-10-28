"""Database integration test; opt-in with RUN_DB_TESTS=1."""

import os
import uuid

import pytest

from healthcare_ai_lab.database.repository import (
    get_agent_run,
    save_agent_run,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_TESTS") != "1",
    reason="Requires local PostgreSQL",
)


def test_persist_and_retrieve_run():
    run_id = str(uuid.uuid4())

    record = {
        "run_id": run_id,
        "episode_id": "database_test",
        "status": "completed",
        "verified": True,
        "execution_mode": "research_prototype",
        "decision": {
            "weight_1": 0.25,
            "weight_2": 0.75,
        },
        "execution": {
            "selected_memory_ids": ["memory_A"],
            "status": "selected",
        },
        "trace": [
            "observe",
            "estimate",
            "allocate",
            "execute",
            "verify",
        ],
    }

    save_agent_run(record)
    loaded = get_agent_run(run_id)

    assert loaded is not None
    assert loaded["run_id"] == run_id
    assert loaded["verified"] is True
    assert loaded["decision"]["weight_2"] == 0.75
    assert loaded["execution"]["selected_memory_ids"] == ["memory_A"]
    assert loaded["trace"] == record["trace"]
