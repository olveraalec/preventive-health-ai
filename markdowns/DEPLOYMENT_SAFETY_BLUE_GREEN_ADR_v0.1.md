# Architecture Decision Record — Blue/Green CI/CD with a Resource-Constrained Fallback (v0.1)

**Project:** `hackathon-starter` / `healthcare_ai_lab`  
**Date:** 2026-10-28  
**Status:** Preferred future architecture; not yet authorized or implemented.  
**Goal:** Keep the verified inference engine, API, and PostgreSQL operationally stable while enabling reversible releases. Research-only deployment; no real patient data in CI logs, containers, or demo fixtures.

## 1. Nonnegotiable release contract

The deployable unit is **an immutable application image plus an explicit artifact manifest and schema-compatibility declaration**. Image digest, source commit SHA, Python/dependency lock, feature-builder version, model bundle SHA-256, metrics version, and database migration compatibility are recorded together.

`/health` = process liveness only; `/ready` = all requirements to serve *the advertised capability* are met. Do not require optional HMM, optional learned controller, or an unavailable external LLM to be healthy if static research inference can serve safely. Conversely, no endpoint should advertise real model readiness when its required model bundle is missing or incompatible. Add a specific research-model readiness check if existing `/ready` must stay backward compatible.

Test gates: (1) source unit + targeted mathematical invariants, (2) non-database pytest, (3) disposable PostgreSQL integration tests for schema/repository changes, (4) version/feature manifest checks, (5) container smoke test, (6) deployment health and rollback checks. GitHub Actions failing tests must block a deployment. No secrets in commits, artifacts, or logs.

## 2. Preferred true blue/green

Separate **blue** and **green** running environments behind a traffic switch (reverse proxy, load balancer, or orchestrator), with both serving the same versioned API contract. Stage the immutable green candidate; run live `/health`, advertised-capability readiness, a synthetic research inference smoke test, and migration-compatibility checks. Only then move traffic. Retain blue for an observation window and roll back traffic on failure; retire blue after acceptance.

AWS-supported patterns include separate EC2 instance groups with CodeDeploy/ELB, or ECS task sets with an ALB and CodeDeploy. Official EC2 CodeDeploy blue/green uses replacement instances, not two containers accidentally squeezed into one tiny instance. GitHub Actions -> AWS should use an **OIDC-federated, least-privilege** AWS role rather than hardcoded long-lived AWS keys. See official AWS CodeDeploy and GitHub Actions OIDC documentation for current prerequisites.

**Capacity gate:** before committing to a pattern, measure image size, inference/model memory, peak two-version RSS, startup time, PostgreSQL concurrency, and affordable instance/task capacity. Record actual measurements, not assumptions. A single low-memory EC2 instance running two Python/ML containers may be unable to maintain both versions; do not assert zero downtime if both cannot be ready concurrently.

## 3. Safe fallback when dual environments are infeasible

Use single-version **health-checked replacement with deterministic rollback**. This is *not true blue/green and not zero-downtime*. Keep the previous immutable image digest and artifact manifest; stop/replace/restart (or use one-at-a-time rolling when more than one instance exists); validate readiness and synthetic inference; automatically restore the last verified image/config on failure. Document any interruption window instead of mislabeling it blue/green.

## 4. Database compatibility is the central rollback constraint

A database is a shared stateful service; application rollbacks cannot necessarily roll back destructive SQL. Use **expand -> deploy compatible application -> migrate use -> contract later**:

- Expand: add new nullable tables/columns or compatibility views; preserve old read/write contract.
- Test blue and green against the shared forward-compatible schema and migration version.
- Switch traffic only if green and old blue are both compatible for the planned observation interval.
- Contract/remove obsolete columns only in a **later** separately approved migration after rollback is no longer needed.
- Back up/restore-test migrations; use least-privilege DB roles; use synthetic records in CI.

The current PostgreSQL repository and existing SQL migrations should be inspected and exercised in a disposable database **before** adding trajectory and metrics persistence. Prefer storing model/run IDs, pseudonymous episode keys, allocation, metric summary, and timing/provenance rather than raw physiology.

## 5. Suggested milestone gate and responsibility

| Milestone | Codex | Human review / authorization |
|---|---|---|
| CI baseline | Pytest, changed-file Ruff, manifest and smoke checks | Approve test scope, verify skips are explicit |
| Local PostgreSQL | Compose, disposable database tests, migration inventory | Approve migrations; no production writes |
| Application image | Docker build, model manifest inclusion/loading contract | Inspect security, non-root user, failure modes |
| Blue/green dry run | Scripted blue/green traffic switch with synthetic probes | Verify measured capacity and rollback |
| AWS deployment | ECR/EC2 or ECS workflow with OIDC and alarms | Explicit approval, budgets/IAM/security before any cloud write |
| Resource-constrained fallback | Health-checked single-version replacement, rollback | Clearly disclose downtime and non-blue/green status |

**Decision:** Target true blue/green if independent deployment capacity is available and costs are acceptable; otherwise ship tested rolling/single replacement. CI/CD reliability and reproducibility matter more than the deployment label.
