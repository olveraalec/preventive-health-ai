
-- Scope memories to an episode/session.
-- Existing tables and agent runs remain intact.

ALTER TABLE memory_items
ADD COLUMN IF NOT EXISTS episode_id TEXT;

CREATE INDEX IF NOT EXISTS idx_memory_episode_score
ON memory_items (
    episode_id,
    information_score DESC,
    memory_id
);

CREATE TABLE IF NOT EXISTS memory_selections (
    run_id UUID NOT NULL
        REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL,
    pathway SMALLINT NOT NULL
        CHECK (pathway IN (1, 2)),
    information_score DOUBLE PRECISION NOT NULL,
    selected_rank INTEGER NOT NULL CHECK (selected_rank > 0),
    PRIMARY KEY (run_id, memory_id),
    UNIQUE (run_id, selected_rank)
);
