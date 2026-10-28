
-- Research prototype: agent execution and evaluation storage.
-- No identifiable patient information should be stored here.

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id UUID PRIMARY KEY,
    episode_id TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('completed', 'failed')),
    verified BOOLEAN NOT NULL,
    execution_mode TEXT NOT NULL,
    decision JSONB NOT NULL,
    execution JSONB NOT NULL,
    trace JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_episode
    ON agent_runs (episode_id, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_items (
    memory_id TEXT PRIMARY KEY,
    pathway SMALLINT NOT NULL
        CHECK (pathway IN (1, 2)),
    information_score DOUBLE PRECISION NOT NULL
        CHECK (
            information_score >= 0
            AND information_score < 'Infinity'::float8
        ),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    evaluation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id UUID REFERENCES agent_runs(run_id),
    policy TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    dataset TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluation_policy_metric
    ON evaluation_results (policy, metric_name);
