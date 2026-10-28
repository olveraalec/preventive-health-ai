"""Healthcare AI Lab API.

Initial health and readiness interface.
No clinical inference is exposed yet.
"""

from fastapi import FastAPI
from healthcare_ai_lab.database.repository import (
    get_episode_memories,
)
from healthcare_ai_lab.database.repository import (
    save_agent_run_with_selections,
)

app = FastAPI(
    title="Healthcare AI Lab",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Process liveness."""
    return {
        "status": "ok",
        "service": "healthcare-ai-lab",
    }


@app.get("/ready")
def ready():
    """Initial application readiness.

    Add database, checkpoint, and inference dependency
    checks before using this as a production readiness gate.
    """
    return {
        "status": "ready",
        "inference_enabled": False,
    }


from typing import Literal
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, Field

from healthcare_ai_lab.agent.workflow import agent_graph
from healthcare_ai_lab.controller.contracts import ControllerInput
from healthcare_ai_lab.controller.service import ControllerService


class DecisionRequest(BaseModel):
    episode_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    variance_1: float = Field(ge=0, allow_inf_nan=False)
    variance_2: float = Field(ge=0, allow_inf_nan=False)
    covariance_12: float = Field(allow_inf_nan=False)
    previous_weight: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    resource_cost: float = Field(default=1.0, ge=0, allow_inf_nan=False)


class MemoryCandidateRequest(BaseModel):
    memory_id: str = Field(min_length=1)
    pathway: Literal[1, 2]
    information_score: float = Field(ge=0, allow_inf_nan=False)


class AgentRunRequest(DecisionRequest):
    memory_candidates: list[MemoryCandidateRequest] | None = None
    memory_budget: int = Field(default=4, ge=0, le=100)


def make_controller_input(request: DecisionRequest):
    return ControllerInput(**request.model_dump())


@app.post("/api/v1/controller/decide")
def decide(request: DecisionRequest):
    try:
        result = ControllerService().decide(make_controller_input(request))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "episode_id": result.episode_id,
        "step": result.step,
        "weight_1": result.weight_1,
        "weight_2": result.weight_2,
        "estimated_risk": result.estimated_risk,
        "estimated_objective": result.estimated_objective,
        "policy": result.policy_name,
        "used_fallback": result.used_fallback,
    }


@app.post("/api/v1/agent/run")
def run_agent(request: AgentRunRequest):
    run_id = str(uuid4())

    try:
        payload = request.model_dump(exclude_none=True)
        result = agent_graph.invoke(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "run_id": run_id,
        "episode_id": request.episode_id,
        "decision": result["decision"],
        "execution": result["execution"],
        "verified": result["verified"],
        "trace": result["trace"],
        "execution_mode": "simulation",
    }


from uuid import UUID

import psycopg

from healthcare_ai_lab.database.repository import (
    save_agent_run,
    get_agent_run,
)


@app.post("/api/v1/agent/runs", status_code=201)
def create_persistent_agent_run(request: AgentRunRequest):
    """Execute and persist one research agent run."""
    run_id = uuid4()

    try:
        result = agent_graph.invoke(request.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    record = {
        "run_id": str(run_id),
        "episode_id": request.episode_id,
        "status": ("completed" if result["verified"] else "failed"),
        "verified": result["verified"],
        "execution_mode": "research_prototype",
        "decision": result["decision"],
        "execution": result["execution"],
        "trace": result["trace"],
    }

    try:
        save_agent_run(record)
    except (psycopg.Error, OSError, RuntimeError):
        raise HTTPException(
            status_code=503,
            detail="Execution storage unavailable",
        ) from None

    return record


@app.get("/api/v1/agent/runs/{run_id}")
def read_persistent_agent_run(run_id: UUID):
    """Retrieve a persisted agent execution."""
    try:
        record = get_agent_run(str(run_id))
    except (psycopg.Error, OSError, RuntimeError):
        raise HTTPException(
            status_code=503,
            detail="Execution storage unavailable",
        ) from None

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Agent run not found",
        )

    return record


class DatabaseAgentRunRequest(DecisionRequest):
    memory_budget: int = Field(default=4, ge=0, le=100)


@app.post("/api/v1/agent/run-from-memory")
def run_agent_from_memory(request: DatabaseAgentRunRequest):
    """Retrieve, execute, and atomically persist memory selection."""
    from uuid import uuid4

    try:
        candidates = get_episode_memories(
            request.episode_id,
            limit=100,
        )
    except (psycopg.Error, OSError, RuntimeError):
        raise HTTPException(
            status_code=503,
            detail="Memory storage unavailable",
        ) from None

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No memories found for episode",
        )

    payload = request.model_dump(exclude_none=True)
    payload["memory_candidates"] = candidates

    try:
        result = agent_graph.invoke(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    run_id = str(uuid4())

    selected_ids = result["execution"].get("selected_memory_ids", [])

    # Reconstruct selections from the retrieved candidate pool.
    # Never trust arbitrary selected IDs without checking provenance.
    by_id = {item["memory_id"]: item for item in candidates}

    if (
        len(selected_ids) != len(set(selected_ids))
        or any(mid not in by_id for mid in selected_ids)
        or len(selected_ids) > request.memory_budget
        or not result["verified"]
    ):
        raise HTTPException(
            status_code=500,
            detail="Agent selection verification failed",
        )

    selected = [by_id[mid] for mid in selected_ids]

    record = {
        "run_id": run_id,
        "episode_id": request.episode_id,
        "status": "completed",
        "verified": True,
        "execution_mode": "database_selection",
        "decision": result["decision"],
        "execution": result["execution"],
        "trace": result["trace"],
    }

    try:
        save_agent_run_with_selections(record, selected)
    except (psycopg.Error, OSError, RuntimeError):
        raise HTTPException(
            status_code=503,
            detail="Execution persistence unavailable",
        ) from None

    return {
        **record,
        "persisted": True,
    }


# ============================================================
# SEPSIS RESEARCH INFERENCE ENDPOINT
# Engineered features only — not raw telemetry
# ============================================================

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from healthcare_ai_lab.models.sepsis_inference import (
    load_bundle,
    predict_engineered_row,
)


class EngineeredPredictionRequest(BaseModel):
    features: dict[str, Any]
    previous_weight: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    reliability: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)


@lru_cache(maxsize=1)
def get_research_bundle():
    path = Path(
        os.environ.get(
            "SEPSIS_MODEL_BUNDLE", "artifacts/models/prototype_bundle.joblib"
        )
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    return load_bundle(path)


@app.post("/api/v1/research/sepsis/predict-engineered")
def predict_sepsis_engineered(request: EngineeredPredictionRequest):
    try:
        bundle = get_research_bundle()
    except (FileNotFoundError, OSError):
        raise HTTPException(
            status_code=503, detail="Research model bundle not installed"
        ) from None

    try:
        return predict_engineered_row(
            bundle,
            request.features,
            previous_weight=request.previous_weight,
            reliability=request.reliability,
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
