import pytest

from healthcare_ai_lab.memory.selector import (
    MemoryCandidate,
    select_memories,
)


def candidates():
    return [
        MemoryCandidate("recent_hr", 1, 0.8),
        MemoryCandidate("recent_spo2", 1, 0.6),
        MemoryCandidate("historical_trend", 2, 0.9),
        MemoryCandidate("historical_context", 2, 0.7),
    ]


def test_budget_is_respected():
    result = select_memories(candidates(), 0.5, 0.5, budget=2)
    assert result.used == 2
    assert len(result.selected_ids) == 2


def test_controller_weights_change_selection():
    first = select_memories(candidates(), 0.9, 0.1, budget=1)
    second = select_memories(candidates(), 0.1, 0.9, budget=1)

    assert first.selected_ids == ("recent_hr",)
    assert second.selected_ids == ("historical_trend",)


def test_zero_budget_selects_nothing():
    result = select_memories(candidates(), 0.5, 0.5, budget=0)
    assert result.selected_ids == ()
    assert result.used == 0


def test_duplicate_memory_rejected():
    items = [
        MemoryCandidate("same_id", 1, 0.5),
        MemoryCandidate("same_id", 2, 0.8),
    ]
    with pytest.raises(ValueError):
        select_memories(items, 0.5, 0.5, 2)


def test_deterministic_selection():
    result_1 = select_memories(candidates(), 0.4, 0.6, 3)
    result_2 = select_memories(list(reversed(candidates())), 0.4, 0.6, 3)

    assert result_1 == result_2
