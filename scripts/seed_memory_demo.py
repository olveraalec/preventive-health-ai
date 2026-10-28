"""Seed deterministic, nonclinical demo memories."""

from healthcare_ai_lab.database.repository import (
    save_memory,
    get_episode_memories,
)


EPISODE = "memory_database_demo"

items = [
    ("recent_hr", 1, 0.8, "Synthetic recent heart-rate context"),
    ("recent_spo2", 1, 0.6, "Synthetic recent SpO2 context"),
    ("historical_trend", 2, 0.9, "Synthetic historical trend"),
    ("historical_context", 2, 0.7, "Synthetic historical context"),
]

for memory_id, pathway, score, content in items:
    save_memory(
        memory_id=memory_id,
        episode_id=EPISODE,
        pathway=pathway,
        information_score=score,
        content=content,
        metadata={"source": "deterministic_demo"},
    )

print("\nSTORED MEMORY CANDIDATES")
for item in get_episode_memories(EPISODE):
    print(item)
