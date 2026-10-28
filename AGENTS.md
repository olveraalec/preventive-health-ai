# AGENTS.md — Healthcare AI Research Repository

These instructions apply to all Codex work in this repository. Read this file, `pyproject.toml`, relevant source/tests, and current research documentation before editing. Follow any more-specific `AGENTS.md` in a subdirectory. If a task conflicts with these rules, explain the conflict instead of silently overriding it.

## Purpose and boundaries

- Python package: `healthcare_ai_lab`; repository: `hackathon-starter` / preventive-health-ai.
- This is a retrospective **research prototype**, not a clinical decision-making system. Never describe outputs as clinically validated, actionable medical advice, or certified risk estimates.
- Separate **Sepsis Challenge hourly classification** from **BIDMC second-scale forecasting**. Never reuse temporal features, outcome assumptions, or measurement units across datasets without verified adapters.
- Treat `markdowns/` as the versioned research/decision record and `docs/` as implementation-facing documentation. Read pertinent contracts before changing numerical behavior. Do not overwrite older research iterations.

## Change discipline

1. At task start: read `git status --short`, inspect affected modules/tests/contracts, state the expected minimal change, and establish an appropriate test baseline.
2. Use a minimal, additive change set. Preserve existing public imports, API routes, model interfaces, database schema, and mathematical definitions unless the task explicitly authorizes change.
3. Do not delete, reset, or overwrite user work; do not run `git reset --hard`, destructive cleans, broad autoformatting, or change unrelated files. Never commit or push automatically unless explicitly requested.
4. Ask for approval before schema migrations, retraining/replacing models, changing feature definitions, loss functions, patient split assignments, fallback precedence, deployment/network changes, or any security-sensitive modification.
5. Never fabricate provenance or claim a model was trained/validated in a way that cannot be verified. Version and checksum artifacts; preserve exact feature names/order and preprocessing assumptions.
6. If mathematical or data-time semantics are ambiguous, preserve existing behavior and flag the question instead of silently making a research decision.
7. Keep existing covariance/PSD optimization, bounded adaptive allocation, HMM, information-theoretic experiments, and standardized metrics independently testable. Hybridization is an experiment, not an automatic refactor.

## Mandatory testing and verification — every implementation task

- **Every new production Python module or behavior change must include new or updated relevant `pytest` tests.** For a bug fix, first add a regression test demonstrating the bug when practical.
- Include normal-path and relevant failure/edge cases: malformed input, missing data, NaN/Inf, shape/schema errors, patient isolation, state reset, and fallback. For time-dependent models, test **prefix causality** and feature parity where applicable.
- Before editing, run applicable baseline tests. After implementation, run targeted tests **and the full non-database suite** from the repository root:

  `uv run --no-sync pytest -q`

- Also run `uv run --no-sync ruff check <changed Python files>` and `git diff --check`. Do not expand the task into fixing unrelated repository-wide lint findings.
- The existing database integration test may remain skipped when `RUN_DB_TESTS` is unset. Changes to persistence, SQL, transactions, or database interfaces **must also be tested** against a disposable local PostgreSQL instance using the opt-in database tests, e.g. `RUN_DB_TESTS=1 uv run --no-sync pytest tests/test_database_integration.py -q`, after verifying connection/configuration.
- If a required check cannot execute, disclose **exactly** which check was not run and why; do not claim the task passed. Never mark a task complete if relevant tests fail, except to report an explicitly blocked/incomplete task.
- Do not modify existing tests merely to make failures disappear. If assertions change, justify the changed contract and add equivalent coverage.
- Preserve baseline behavior: run existing regression suites, check numerical parity against trusted references, and use tight, justified tolerances when math is expected to be identical.

## Mathematical and temporal correctness

- Sepsis target is the **shifted challenge label**, not automatically real-world onset probability. Avoid clinical claims from benchmark labels.
- Only use information available at decision time: no future observations, leaked outcomes, post-event labels, or time-ineligible memories. Reactive feedback requires an explicit availability timestamp.
- Keep centered covariance and uncentered error second moments distinct. Do not silently substitute one for the other or substitute Gaussian NLL for Bernoulli log loss.
- Static prediction must be an executable emergency fallback even if temporal/gain/HMM/agent components fail. An emergency fallback takes priority over ordinary allocation step-smoothing constraints.
- Information-theoretic diagnostics (entropy, mutual information) are not inherently calibrated reliability estimates or proof of causal usefulness.
- Assess adaptive policies against static **and** training-selected simple constant-mixture baselines with patient-disjoint evaluation, paired patient statistics, calibration, and patient-level tail outcomes.
- Preserve per-patient state isolation. For HMM/streaming systems, reset between episodes and never apply retrospective smoothing in place of causal filtering.

## Data, privacy, and artifacts

- Do not print, upload, commit, log, or expose raw patient records, patient identifiers, credentials, `.env` secrets, database dumps, or model binaries. Use synthetic fixtures for committed tests.
- Do not load untrusted pickle/joblib binaries. Only load trusted, local model artifacts with explicit provenance/checksum validation.
- Keep patient data, caches, model artifacts, and local manifests Git-ignored. New persistent storage should minimize data fields and use pseudonymous IDs.
- No external network operations, deployment, cloud writes, destructive SQL, or schema migrations without explicit authorization.

## Completion report — required every time

Report concisely:
1. Summary and paths of files changed.
2. Exact tests/checks executed and their results (passed, failed, skipped); note database integration status separately.
3. API/schema/math behavior changes, if any, and confirmation existing behavior is preserved.
4. Remaining limitations, unverified claims, provenance caveats, and the next integration step.

For large tasks, propose a brief plan before writing files. Prefer small reviewable tasks and a checkpoint at important mathematical or database decisions.


## Research documentation

For substantive mathematical or architectural changes:
- Read `markdowns/CODEX_RESEARCH_READING_MAP.md`.
- Read the relevant versioned research contracts.
- Read `markdowns/RESEARCH_MIDDLEWARE_ARCHITECTURE_AND_EXPERIMENTS_v0.4.md` for system-level changes.
- Preserve previous research versions.
- Never equate a proposed hypothesis with a validated result.
- Report the research documents consulted and any unresolved conflicts.
