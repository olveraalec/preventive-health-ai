"""Database integration test; opt-in with RUN_DB_TESTS=1."""

import os
import uuid

import psycopg
import pytest

from healthcare_ai_lab.database.repository import (
    database_connection,
    get_agent_run,
    get_episode_memories,
    get_run_memory_selections,
    save_agent_run,
    save_agent_run_with_selections,
    save_memory,
    save_memory_selections,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_TESTS") != "1",
    reason="Requires local PostgreSQL",
)


@pytest.fixture(autouse=True)
def require_disposable_database():
    """Refuse repository defaults/remote hosts before opening any connection."""
    database = os.getenv("POSTGRES_DB", "")
    if (
        os.getenv("POSTGRES_HOST") != "127.0.0.1"
        or not (
            database == "healthcare_ai_test"
            or database.startswith("healthcare_ai_test_")
        )
        or os.getenv("POSTGRES_USER") != "healthcare_test"
    ):
        pytest.fail(
            "Live tests require an explicitly configured loopback disposable healthcare_ai_test database and healthcare_test role",
            pytrace=False,
        )


def synthetic_record(episode_id):
    return {
        "run_id": str(uuid.uuid4()),
        "episode_id": episode_id,
        "status": "completed",
        "verified": True,
        "execution_mode": "research_prototype",
        "decision": {"weight_1": 0.5, "weight_2": 0.5},
        "execution": {"status": "selected"},
        "trace": ["synthetic"],
    }


def synthetic_memory(episode_id, suffix, score=0.5, pathway=1):
    item = {
        "memory_id": f"synthetic-{uuid.uuid4()}-{suffix}",
        "pathway": pathway,
        "information_score": score,
    }
    save_memory(
        **item,
        episode_id=episode_id,
        content="synthetic fixture only",
        metadata={"synthetic": True},
    )
    return item


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


def test_existing_migrations_inventory():
    with database_connection() as conn:
        tables = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'memory_items'"
        ).fetchall()
    assert {row["tablename"] for row in tables} == {
        "agent_runs",
        "memory_items",
        "evaluation_results",
        "memory_selections",
    }
    assert "episode_id" in {row["column_name"] for row in columns}


def test_episode_isolation_ordering_and_limit():
    episode = f"synthetic-{uuid.uuid4()}"
    foreign_episode = f"synthetic-{uuid.uuid4()}"
    low = synthetic_memory(episode, "low", score=0.1)
    high = [synthetic_memory(episode, suffix, score=0.9) for suffix in ("a", "b")]
    foreign = synthetic_memory(foreign_episode, "foreign", score=1.0)
    expected = sorted(high, key=lambda item: item["memory_id"]) + [low]
    assert get_episode_memories(episode) == expected
    assert get_episode_memories(episode, limit=1) == expected[:1]
    assert get_episode_memories(foreign_episode) == [foreign]
    assert get_episode_memories(f"synthetic-empty-{uuid.uuid4()}") == []


def test_episode_identifier_is_a_parameter_not_sql():
    episode = f"synthetic-{uuid.uuid4()}-' OR 1=1 --"
    item = synthetic_memory(episode, "quoted")
    synthetic_memory(f"synthetic-other-{uuid.uuid4()}", "foreign")
    assert get_episode_memories(episode) == [item]


def test_atomic_run_and_selection_round_trip():
    episode = f"synthetic-{uuid.uuid4()}"
    selected = [
        synthetic_memory(episode, "second", pathway=2),
        synthetic_memory(episode, "first"),
    ]
    record = synthetic_record(episode)
    save_agent_run_with_selections(record, selected)
    assert get_agent_run(record["run_id"])["episode_id"] == episode
    assert get_run_memory_selections(record["run_id"]) == [
        {**item, "selected_rank": rank} for rank, item in enumerate(selected, start=1)
    ]


@pytest.mark.parametrize("failure", ["duplicate_memory", "invalid_pathway"])
def test_combined_write_rolls_back_run_and_earlier_selection(failure):
    episode = f"synthetic-{uuid.uuid4()}"
    first = synthetic_memory(episode, "valid")
    second = dict(first)
    expected_error = psycopg.errors.UniqueViolation
    if failure == "invalid_pathway":
        second.update(memory_id=f"synthetic-{uuid.uuid4()}", pathway=3)
        expected_error = psycopg.errors.CheckViolation
    record = synthetic_record(episode)
    with pytest.raises(expected_error):
        save_agent_run_with_selections(record, [first, second])
    assert get_agent_run(record["run_id"]) is None
    assert get_run_memory_selections(record["run_id"]) == []
    # Memory creation preceded the failed transaction and must survive.
    assert get_episode_memories(episode) == [first]


def test_selection_only_failure_preserves_preexisting_run():
    episode = f"synthetic-{uuid.uuid4()}"
    item = synthetic_memory(episode, "valid")
    record = synthetic_record(episode)
    save_agent_run(record)
    with pytest.raises(psycopg.errors.UniqueViolation):
        save_memory_selections(record["run_id"], [item, item])
    assert get_agent_run(record["run_id"]) is not None
    assert get_run_memory_selections(record["run_id"]) == []


def test_selection_requires_existing_run():
    item = synthetic_memory(f"synthetic-{uuid.uuid4()}", "valid")
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        save_memory_selections(str(uuid.uuid4()), [item])


@pytest.mark.parametrize(
    "ranks,error",
    [([0], psycopg.errors.CheckViolation), ([1, 1], psycopg.errors.UniqueViolation)],
)
def test_database_rank_constraints(ranks, error):
    record = synthetic_record(f"synthetic-{uuid.uuid4()}")
    save_agent_run(record)
    with pytest.raises(error), database_connection() as conn:
        for rank in ranks:
            conn.execute(
                "INSERT INTO memory_selections "
                "(run_id, memory_id, pathway, information_score, selected_rank) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    uuid.UUID(record["run_id"]),
                    f"synthetic-{uuid.uuid4()}",
                    1,
                    0.5,
                    rank,
                ),
            )
    assert get_run_memory_selections(record["run_id"]) == []


@pytest.mark.parametrize("score", [-1.0, float("nan"), float("inf")])
def test_database_memory_score_constraint(score):
    with pytest.raises(psycopg.errors.CheckViolation), database_connection() as conn:
        conn.execute(
            "INSERT INTO memory_items "
            "(memory_id, episode_id, pathway, information_score, content) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                f"synthetic-{uuid.uuid4()}",
                f"synthetic-{uuid.uuid4()}",
                1,
                score,
                "synthetic",
            ),
        )


def test_unknown_run_returns_none():
    assert get_agent_run(str(uuid.uuid4())) is None
