"""PostgreSQL repository for research agent executions."""

import os
from contextlib import contextmanager
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import load_dotenv


load_dotenv()


def save_memory(
    *,
    memory_id: str,
    episode_id: str,
    pathway: int,
    information_score: float,
    content: str,
    metadata: dict | None = None,
) -> None:
    """Store a research-safe memory candidate."""
    import math

    if not memory_id or not episode_id:
        raise ValueError("Memory and episode IDs are required")

    if pathway not in (1, 2):
        raise ValueError("Invalid pathway")

    if not math.isfinite(information_score) or information_score < 0:
        raise ValueError("Invalid information score")

    query = """
        INSERT INTO memory_items (
            memory_id,
            episode_id,
            pathway,
            information_score,
            content,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    with database_connection() as conn:
        conn.execute(
            query,
            (
                memory_id,
                episode_id,
                pathway,
                information_score,
                content,
                Jsonb(metadata or {}),
            ),
        )


def get_episode_memories(
    episode_id: str,
    limit: int = 100,
) -> list[dict]:
    """Retrieve memory candidates for one episode only."""
    if not episode_id:
        raise ValueError("episode_id is required")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    query = """
        SELECT
            memory_id,
            pathway,
            information_score
        FROM memory_items
        WHERE episode_id = %s
        ORDER BY information_score DESC, memory_id
        LIMIT %s
    """

    with database_connection() as conn:
        return conn.execute(
            query,
            (episode_id, limit),
        ).fetchall()


def save_memory_selections(
    run_id: str,
    selected: list[dict],
) -> None:
    """Store the ranked selected memories for an agent run."""
    query = """
        INSERT INTO memory_selections (
            run_id,
            memory_id,
            pathway,
            information_score,
            selected_rank
        )
        VALUES (%s, %s, %s, %s, %s)
    """

    with database_connection() as conn:
        for rank, item in enumerate(selected, start=1):
            conn.execute(
                query,
                (
                    UUID(run_id),
                    item["memory_id"],
                    item["pathway"],
                    item["information_score"],
                    rank,
                ),
            )


@contextmanager
def database_connection():
    """Manage a PostgreSQL connection and transaction."""
    password = os.environ.get("POSTGRES_PASSWORD")

    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is not configured")

    with psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "healthcare_ai"),
        user=os.getenv("POSTGRES_USER", "healthcare"),
        password=password,
        connect_timeout=5,
        row_factory=dict_row,
    ) as connection:
        yield connection


def save_agent_run(record: dict) -> None:
    """Store one completed agent execution atomically."""
    query = """
        INSERT INTO agent_runs (
            run_id,
            episode_id,
            status,
            verified,
            execution_mode,
            decision,
            execution,
            trace
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    with database_connection() as conn:
        conn.execute(
            query,
            (
                UUID(str(record["run_id"])),
                record["episode_id"],
                record["status"],
                record["verified"],
                record["execution_mode"],
                Jsonb(record["decision"]),
                Jsonb(record["execution"]),
                Jsonb(record["trace"]),
            ),
        )


def get_agent_run(run_id: str) -> dict | None:
    """Retrieve a stored execution by its UUID."""
    query = """
        SELECT
            run_id,
            episode_id,
            status,
            verified,
            execution_mode,
            decision,
            execution,
            trace,
            created_at
        FROM agent_runs
        WHERE run_id = %s
    """

    with database_connection() as conn:
        row = conn.execute(query, (UUID(run_id),)).fetchone()

    if row is None:
        return None

    return {
        **row,
        "run_id": str(row["run_id"]),
        "created_at": row["created_at"].isoformat(),
    }


def list_memory_candidates(limit: int = 20) -> list[dict]:
    """Get stored candidates for later agent retrieval."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    query = """
        SELECT memory_id, pathway, information_score
        FROM memory_items
        ORDER BY information_score DESC, memory_id
        LIMIT %s
    """

    with database_connection() as conn:
        rows = conn.execute(query, (limit,)).fetchall()

    return rows


def save_agent_run_with_selections(
    record: dict,
    selected: list[dict],
) -> None:
    """Persist an agent run and its selections atomically.

    Either every INSERT succeeds, or the transaction rolls back.
    """
    run_query = """
        INSERT INTO agent_runs (
            run_id, episode_id, status, verified,
            execution_mode, decision, execution, trace
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    selection_query = """
        INSERT INTO memory_selections (
            run_id, memory_id, pathway,
            information_score, selected_rank
        )
        VALUES (%s, %s, %s, %s, %s)
    """

    run_id = UUID(str(record["run_id"]))

    # Both writes share one database transaction.
    with database_connection() as conn:
        conn.execute(
            run_query,
            (
                run_id,
                record["episode_id"],
                record["status"],
                record["verified"],
                record["execution_mode"],
                Jsonb(record["decision"]),
                Jsonb(record["execution"]),
                Jsonb(record["trace"]),
            ),
        )

        for rank, item in enumerate(selected, start=1):
            conn.execute(
                selection_query,
                (
                    run_id,
                    item["memory_id"],
                    item["pathway"],
                    item["information_score"],
                    rank,
                ),
            )


def get_run_memory_selections(run_id: str) -> list[dict]:
    """Retrieve selected memories in their saved order."""
    query = """
        SELECT
            memory_id, pathway,
            information_score, selected_rank
        FROM memory_selections
        WHERE run_id = %s
        ORDER BY selected_rank
    """

    with database_connection() as conn:
        return conn.execute(query, (UUID(run_id),)).fetchall()
