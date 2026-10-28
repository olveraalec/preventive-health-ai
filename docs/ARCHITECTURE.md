# Architecture and portability

Data/task adapter -> observation/state estimator -> value & reliability estimators -> covariance-aware controller -> budgeted memory selection -> LangGraph execution -> verification -> metrics + optional persistence.

## Current reusable interfaces
- `healthcare_ai_lab.memory.selector`: `MemoryCandidate`, `select_memories` and deterministic top-k budget enforcement.
- `healthcare_ai_lab.agent.workflow`: LangGraph `observe -> estimate -> allocate -> execute -> verify`.
- `healthcare_ai_lab.evaluation.registry` and `evaluation.suite`: metric registry, per-recording scoring, paired bootstrap.
- `controller/`, `database/`, `api/`, `sql/`, Docker and tests: **copy only if present**, then validate dependencies and external services.

## Important distinction
The existing workflow's `execute` stage selects memories or simulates allocation; it does **not** perform actual clinical inference or an external task by itself. Verification checks structural validity, not factual answer quality.

## Preserve package name
The snapshot keeps the original `healthcare_ai_lab` Python package name to avoid import breakage. Rename only after a deliberate import-update/test pass. This is a copy, not a rewrite or deployment.

## Tomorrow's adaptation points
1. New input adapter / schemas / data provenance.
2. Task outcome and utility definition.
3. Actual agent tools / execution implementation.
4. Candidate score meaning and comparability across pathways.
5. Covariance estimator and explicit assumption linking dependence to utility.
6. Reliability features available strictly before action.
7. Held-out evaluation split, calibration, and deployment guardrails.
