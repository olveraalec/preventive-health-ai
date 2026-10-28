"""First executable agentic middleware workflow.

Separates orchestration from controller mathematics.
The execution node currently performs a deterministic action,
not clinical inference or external tool execution.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from healthcare_ai_lab.controller.contracts import ControllerInput
from healthcare_ai_lab.controller.service import ControllerService
from healthcare_ai_lab.memory.selector import (
    MemoryCandidate,
    select_memories,
)

class AgentState(TypedDict, total=False):
    episode_id: str
    step: int
    variance_1: float
    variance_2: float
    covariance_12: float
    previous_weight: float
    resource_cost: float
    state_entropy: float
    information_value: float
    memory_candidates: list[dict]
    memory_budget: int

    observations_valid: bool
    decision: dict
    execution: dict
    verified: bool
    trace: list[str]


def observe_node(state: AgentState) -> dict:
    return {
        "observations_valid": True,
        "trace": ["observe"],
    }


def estimate_node(state: AgentState) -> dict:
    # Validate supplied moments, but do not claim to estimate
    # covariance or HMM beliefs in this first integration.
    ControllerInput(
        episode_id=state["episode_id"],
        step=state["step"],
        variance_1=state["variance_1"],
        variance_2=state["variance_2"],
        covariance_12=state["covariance_12"],
        previous_weight=state.get("previous_weight", 0.5),
        resource_cost=state.get("resource_cost", 1.0),
        state_entropy=state.get("state_entropy"),
        information_value=state.get("information_value"),
    )

    return {"trace": state["trace"] + ["estimate"]}


def allocate_node(state: AgentState) -> dict:
    service = ControllerService()

    request = ControllerInput(
        episode_id=state["episode_id"],
        step=state["step"],
        variance_1=state["variance_1"],
        variance_2=state["variance_2"],
        covariance_12=state["covariance_12"],
        previous_weight=state.get("previous_weight", 0.5),
        resource_cost=state.get("resource_cost", 1.0),
        state_entropy=state.get("state_entropy"),
        information_value=state.get("information_value"),
    )

    decision = service.decide(request)

    return {
        "decision": {
            "weight_1": decision.weight_1,
            "weight_2": decision.weight_2,
            "estimated_risk": decision.estimated_risk,
            "estimated_objective": decision.estimated_objective,
            "policy_name": decision.policy_name,
            "used_fallback": decision.used_fallback,
        },
        "trace": state["trace"] + ["allocate"],
    }



def execute_node(state: AgentState) -> dict:
    """Execute budgeted memory selection when candidates exist."""
    decision = state["decision"]

    if "memory_candidates" not in state:
        execution = {
            "action": "allocate_context",
            "pathway_1_weight": decision["weight_1"],
            "pathway_2_weight": decision["weight_2"],
            "status": "simulated",
        }
    else:
        candidates = [
            MemoryCandidate(**item)
            for item in state["memory_candidates"]
        ]

        result = select_memories(
            candidates=candidates,
            weight_1=decision["weight_1"],
            weight_2=decision["weight_2"],
            budget=state.get("memory_budget", 4),
        )

        execution = {
            "action": "select_memory",
            "selected_memory_ids": list(
                result.selected_ids
            ),
            "budget": result.budget,
            "used": result.used,
            "estimated_utility": result.total_weighted_utility,
            "status": "selected",
        }

    return {
        "execution": execution,
        "trace": state["trace"] + ["execute"],
    }



def verify_node(state: AgentState) -> dict:
    decision = state["decision"]


    execution = state["execution"]

    allocation_valid = (
        0 <= decision["weight_1"] <= 1
        and 0 <= decision["weight_2"] <= 1
        and abs(
            decision["weight_1"]
            + decision["weight_2"] - 1.0
        ) < 1e-8
    )

    if execution["status"] == "selected":
        selected_ids = execution["selected_memory_ids"]
        available_ids = {
            item["memory_id"]
            for item in state["memory_candidates"]
        }
        action_valid = (
            execution["used"] <= execution["budget"]
            and execution["used"] == len(selected_ids)
            and len(selected_ids) == len(set(selected_ids))
            and set(selected_ids).issubset(available_ids)
        )
    else:
        action_valid = execution["status"] == "simulated"

    valid = allocation_valid and action_valid


    return {
        "verified": valid,
        "trace": state["trace"] + ["verify"],
    }


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("observe", observe_node)
    builder.add_node("estimate", estimate_node)
    builder.add_node("allocate", allocate_node)
    builder.add_node("execute", execute_node)
    builder.add_node("verify", verify_node)

    builder.add_edge(START, "observe")
    builder.add_edge("observe", "estimate")
    builder.add_edge("estimate", "allocate")
    builder.add_edge("allocate", "execute")
    builder.add_edge("execute", "verify")
    builder.add_edge("verify", END)

    return builder.compile()


agent_graph = build_graph()
