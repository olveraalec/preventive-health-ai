# Codex Research Reading Map — Healthcare AI Lab

**Date:** 2026-10-28  
**Purpose:** Index of authoritative context for task-specific Codex reading. This is **not** an additional top-level `AGENTS.md` and does not supersede the actual root repository instructions.

## Universal behavior

1. Read root `AGENTS.md` first; inspect `git status --short` and affected source/tests.
2. For a substantive design, model, metrics, or data contract change, read `markdowns/00_RESEARCH_INDEX.md`, `markdowns/RESEARCH_MIDDLEWARE_ARCHITECTURE_AND_EXPERIMENTS_v0.4.md`, and the task-relevant source references below. For a small mechanical task, read only the relevant reference(s), avoiding redundant entire-paper loading.
3. Preserve code compatibility and historical research iterations. If mathematical papers and code disagree, **report the difference** and request approval before changing numerical semantics.
4. For each implementation: baseline relevant tests, add targeted tests, run full non-database pytest, scoped Ruff and `git diff --check`; for database work run opt-in disposable PostgreSQL integration tests. Report exact skipped tests and any unverified evidence.
5. Do not equate documentation with validated results, and do not assume any new file written by ChatGPT is already present on disk; confirm paths before relying on them.

## Read by task

| Work area | Required context before a substantive change | Existing implementation to inspect |
|---|---|---|
| Sepsis probability inference, target, artifact | `docs/SEPSIS_ARTIFACT_VERIFICATION.md`; `docs/SEPSIS_FEATURES.md`; v0.4 Sections 1–2 | `models/sepsis_inference.py`, `models/artifact.py`, `preprocessing/sepsis.py`, `trajectories/*`, tests |
| Covariance/conditional allocation | `markdowns/COVARIANCE_AWARE_CONDITIONAL_ALLOCATION_v0.2.md`; `markdowns/RELIABILITY_GATED_HYBRID_AND_INFORMATION_THEORY_v0.3.md`; v0.4 Sections 3–4 | `controller/optimizer.py`, `controller/service.py`, `controllers/adaptive.py`, nested notebooks |
| HMM, information, memory | v0.4 Sections 5–6; `docs/MATHEMATICAL_FRAMEWORK.md`; existing information experiments | `state_estimation/hmm.py`, `models/state_estimator.py`, `memory/*`, `experiments/*` |
| Evaluation registry | `SEPSIS_EVALUATION_AND_HYBRID_EXPERIMENT_CONTRACT_v0.1.md` in its actual location; v0.4 Sections 7–10 | `evaluation/registry.py`, `evaluation/suite.py`, `evaluation/paired_analysis.py`, tests |
| LangGraph research workflow | v0.4 Sections 2 and 6; `docs/ARCHITECTURE.md` | `agent/workflow.py`, `controller/contracts.py`, API and test modules |
| PostgreSQL / agent persistence | `docs/CI_AND_POSTGRESQL.md`; v0.4 Sections 2 and 9 | `database/repository.py`, `sql/001_initial_schema.sql`, `sql/002_memory_scoping.sql`, `compose.yaml`, DB tests |
| AWS/CI/CD/HTTPS | `markdowns/AWS_API_SECURITY_AND_DEPLOYMENT_ADR_v0.2.md`, `markdowns/DEPLOYMENT_SAFETY_BLUE_GREEN_ADR_v0.1.md`; root AGENTS security section | `.github/workflows/ci.yml`, `Dockerfile`, `compose.yaml`, API |
| Research claims / demo | `markdowns/00_RESEARCH_INDEX.md`, v0.4 Section 0, 9 and 11; prior nested benchmark and failure-analysis notebooks | tests, artifact manifest, verified smoke-test logs |

## Scientific authority and status

- **Source of empirical truth:** reproducible notebooks/scripts, manifests, evaluated splits and recorded numerical outputs. A Markdown proposal is *not* experimental evidence.
- **Source of mathematical decisions:** versioned research contracts. If definitions conflict, defer to the explicitly chosen latest approved contract, but do not silently rewrite existing code.
- **Source of implementation truth:** inspected current repository, executed tests, CI checks, local database state.
- **Scope of LLM:** research-summary orchestration only; no authority to make numerical risk estimates, select unapproved deployment configurations, or infer clinical validity.

## Codex completion receipt

Report: (a) research files actually read, (b) affected production files, (c) mathematical assumptions preserved or newly proposed, (d) exact pytest/lint/CI/DB evidence, (e) data/privacy and time-leakage checks, (f) any open blockers. Distinguish **verified**, **reported**, and **proposed**.
