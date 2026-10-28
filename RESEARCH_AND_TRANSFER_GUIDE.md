# Hackathon Starter — Research, Mathematics, Reproducibility & Transfer Guide

**Status:** Research-stage middleware prototype; not clinically validated. **Reference date:** 2026-10-28. This document summarizes results supplied from local experiments; it has not independently reproduced them against the full local repository.

## Executive purpose

Build a portable, budget-constrained agent orchestration and memory-selection middleware. Separate (i) state estimation, (ii) conditional information value, (iii) covariance/statistical dependence, (iv) reliability estimation, (v) constrained allocation, (vi) execution and verification, and (vii) outcome evaluation. Do not assume that low variance equals high information value, that a well-formed agent trace means task success, or that simulated results imply clinical effectiveness.

## Architectural contracts

- **Python 3.12 + uv lockfile:** reproducible dependency environment (`uv sync --locked`).
- **FastAPI** (`src/healthcare_ai_lab/api/main.py`): application interface, health/readiness and agent routes; `/ready` returning `inference_enabled:false` is **not** proof that model inference is enabled.
- **LangGraph** (`src/healthcare_ai_lab/agent/workflow.py`): `observe → estimate → allocate → execute → verify` and trace. Current deterministic execution validates memory selection, not clinical inference or external tool use.
- **Controller** (`src/healthcare_ai_lab/controller/`): PSD covariance-aware regularized two-pathway allocation; keep static fallback.
- **Memory** (`src/healthcare_ai_lab/memory/`): `MemoryCandidate`, budgeted ranking, stable tie-breaks. Scores must be calibrated across pathways; discrete top-k makes downstream outcome discontinuous in continuous weights.
- **State models** (`src/healthcare_ai_lab/state_estimation/`, `models/`): HMM/Bayesian state estimators; optional and subject to transfer-time task adaptation.
- **Evaluation** (`src/healthcare_ai_lab/evaluation/`): registry and patient/recording grouped comparison. Use RMSE, MAE, bias, p90 absolute error for predictions and Gaussian NLL, CRPS, 90% coverage and width **only when valid predictive distributions are available**. Also measure realized utility, regret, task success, failure rate, latency, budget use, redundancy, controller variation and fallback frequency. Report paired group-level differences, win fractions and uncertainty, not pooled correlated timesteps as independent samples.
- **Persistence** (`src/healthcare_ai_lab/database/repository.py`): PostgreSQL agent runs, memory items, evaluations and selection links. Verify transactional behavior, including rollback, separately.
- **Infrastructure** (`Dockerfile`, `compose.yaml`, `sql/`, `.env.example`): multistage image, non-root service, local isolated database. Never publish `.env`, credentials or raw identifiable health data.

## Mathematical framework

Let `w_t ∈ Δ²` be nonnegative pathway weights summing to one, `Σ_t` an estimated positive semidefinite covariance matrix, `b_t` an optional latent-state belief and `S_t` the selected memory set (`|S_t| ≤ B`). A candidate objective is

\[
\min_{w_t \in \Delta^2}\;
-\widehat V_t(w_t\mid x_t,b_t)
+\lambda_\Sigma w_t^\top\widehat\Sigma_t^{\rm PSD}w_t
+\lambda_s\|w_t-w_{t-1}\|_2^2
+\lambda_m C_t(w_t)
+\lambda_u U_t(w_t).
\]

This is a **research proposal**, not a validated production optimization. The terms are only jointly meaningful with compatible scales, estimable quantities, and ablation evidence.

**Realized selection utility** may be defined in a controlled simulation as `U(S)=Σᵢ∈S vᵢ − λᵣ R(S)`; regret is `U(S*)−U(S)` for an oracle-optimal feasible set. Such `vᵢ` and oracle regret generally cannot be observed at deployment without experimental design, counterfactual estimation, or feedback.

**Covariance actionability:** define paired incremental benefit `Δ_t=L_static,t−L_cov,t`. Covariance is useful *for this decision problem* when `E[Δ_t | x_t] > 0` after accounting for its uncertainty, rather than merely when estimated covariance is nonzero. Pre-decision features must be available at the actual decision time; lagged residuals, missingness, calibration, signal disagreement and shift statistics may help but must be empirically tested.

**Anticipatory gate** (proposed):

\[
\widehat\Delta_t=f_\theta(x_t,\mathcal H_{t-1}),\qquad
q_t=\sigma\left(\frac{\widehat\Delta_t-\kappa\widehat u_t-\tau}{T}\right),\qquad
w_t=(1-q_t)w_{\rm static}+q_tw_{\rm cov}.
\]

`q_t` is a bounded exposure score, **not** a calibrated probability unless validated. Parameter bootstrap spread alone is not total predictive uncertainty. A blended weight can produce different discrete choices than a randomized mixture of policies. If regimes cannot be distinguished from observable features, anticipatory classification is impossible without additional information.

**Reactive benchmark:** an exponentially smoothed historical gain estimate `d̂_t=(1−β)d̂_(t−1)+βΔ_(t−1)`. This is causally ordered but responds after regime changes, and training on both policies' outcomes assumes full-information counterfactual feedback.

**HMM option:** `b_t(z)=P(Z_t=z | observations through t)`, with benefit estimated as `Σ_z b_t(z) Ê[Δ_t | Z_t=z]`. Hidden regimes in synthetic simulations are evaluation labels, **not** observable inputs in production. Fit/validate HMM state identification on training episodes only.

## Experimental record (reported terminal outputs)

### Original agent-level simulation (60 episodes)

| Policy | Mean realized utility | Mean regret | Redundant same-path pairs |
|---|---:|---:|---:|
| Static equal / top-score | 1.1903 | 0.2158 | 0.4000 |
| Continuous | 1.1009 | 0.3053 | 0.7333 |
| Recent-only | 0.9819 | 0.4242 | 0.0000 |

Static equal and top-score are identical because equal multiplicative pathway weights preserve ranking. Continuous vs static paired mean regret improvement = **−0.08946**, descriptive 95% bootstrap CI **[−0.19402, +0.00259]**. Covariance was generated independently of true utility, so this is an integration stress test, not evidence of inherent covariance inferiority.

### Covariance signal alignment (180 episodes per scenario)

| Scenario | Static mean regret | Continuous mean regret | Gain for continuous | Descriptive paired bootstrap CI |
|---|---:|---:|---:|---|
| Aligned | 0.2586 | 0.0640 | +0.19461 | [+0.15687, +0.23319] |
| Uninformative | 0.2586 | 0.2586 | 0 | [0, 0] |
| Misleading | 0.2586 | 0.9774 | −0.71881 | [−0.79144, −0.64800] |

The covariance–utility relationship was explicitly constructed. These results establish sensitivity to signal quality, **not** general real-data performance. Continuous allocation frequently chose same-path items (0.9889–0.9944 pairs in aligned/misleading regimes); pairwise redundancy was not directly optimized by the selector.

### Policy exposure under uncertainty

Randomized mixture *expected* regret: `R(q)=(1−q)R_static+q R_continuous`. Across three equally weighted scenarios, mean regret goes from 0.2586 (`q=0`) to 0.4333 (`q=1`); worst-scenario mean from 0.2586 to 0.9774. Minimax and equal-scenario-average exposure choose `q=0` in this deliberately chosen scenario family. Positive average benefit from unconditional exposure requires `P(aligned)/P(misleading) > 0.7188/0.1946 ≈ 3.69`, assuming no gain in uninformative regimes. Scenario frequencies are **unknown** in real tasks.

### Reactive gate (540 constructed decisions)

Mean regret static **0.2586**, always continuous **0.4333**, adaptive reactive **0.2358**. Reactive exposure: aligned 0.6850, misleading 0.0954, uninformative 0.1670. At transitions into misleading regimes, exposure was sometimes 0.7404–0.9576 before unfavorable feedback arrived. Structured repeated regimes and full-information feedback limit generality.

### Anticipatory gate (100 reused-pool synthetic sequences)

Mean regret static **0.2815**, continuous **0.4471**, reactive **0.2658**, anticipatory **0.2604**. Anticipatory vs static mean sequence-level improvement **+0.0210**, 83% of sequences positive, descriptive CI **[0.0171, 0.0252]**. Anticipatory vs reactive **+0.00534**, 60% positive, descriptive CI **[0.00071, 0.01015]**, worst sequence gain **−0.05939**. Predicted gain/realized gain correlation **0.4889**, MAE **0.3364**. An intentionally regime-correlated synthetic cue generated the predictive signal. The same original outcome pool was reused across sequences, so bootstrap intervals understate some uncertainty. These results support *feasibility conditional on informative pre-decision context*, not real-world effectiveness.

### Earlier BIDMC research context

BIDMC-derived numerical telemetry: HR, PULSE, RESP, SpO2, time; original single recording contained a 13-s missing PULSE/SpO2 interval. A 10-recording batch, time-aware features, 10-s HR forecasting, residual autocorrelation, regularization, nested recording holdouts, dynamic selection and uncertainty/calibration experiments were explored. One reported nested selected RMSE **1.22845** vs equal-blend **1.23582**; a small difference whose generality is uncertain. Out-of-sample information-value ranking was poor (reported mean Spearman ≈ **−0.0412**, sign accuracy ≈ **0.4889**). Preserve the real-data and synthetic evidence separately; confirm provenance, preprocessing, split definitions and all numerical figures from saved scripts before public citation.

## New-dataset protocol for the hackathon

1. **Specify actual objective:** prediction loss, successful tool action, information retrieval, task completion, dollars/latency, safety constraints. Decide what would count as *realized* value, and what is merely an estimated proxy.
2. **Map source quality:** schema, timestamps, missingness, identifiers, cardinalities, duplication, leakage risks, useful privacy boundaries, action budget.
3. **Establish simple baselines:** recent-only, top-score, static equal, and no-memory/no-controller if meaningful.
4. **Define pre-decision features:** lagged missingness, residual/uncertainty estimates, disagreement, lagged covariance, domain state, distribution-shift cues. Audit which features could inadvertently use future outcomes.
5. **Test covariance usefulness:** compute temporally valid, grouped held-out paired `Δ=L_static−L_cov` and study conditional gain `E[Δ|features]`, not just sample correlation or covariance magnitude.
6. **Fit a cautious anticipatory model:** choose small regularized models first; use train/validation/test separated by group or time. Tune thresholds/temperature on validation only. Use calibrated uncertainty if claiming bounds.
7. **Measure all applicable outcomes:** predictive RMSE/MAE/bias/p90 and appropriate probabilistic scores; task utility, regret if an oracle exists, costs, latency, failures, memory budget, redundant selections, weight variation, fallback rate. Include uncertainty at the independent-unit level.
8. **Ablate modules:** value alone, covariance alone, reactive gate, anticipatory gate, HMM, stability and regularization. Do not claim improvements without sound held-out comparisons.
9. **Prioritize reliability:** define deterministic fallback when data are missing, covariance PSD estimation fails, or the expected advantage is small/unreliable.
10. **Record reproducibility:** dataset version/license, seeds, exact commands, environment fingerprint, train/test partitions, metrics definitions, failure modes, and a limitations paragraph.

## Local start and validation

From `~/code/projects/hackathon-starter`:

```bash
uv sync --locked
PYTHONPATH=src uv run pytest tests/ -q
# Previously observed: 63 passed, 1 skipped (DB integration opt-in).
docker compose -p hackathon-starter config --quiet
docker compose -p hackathon-starter up -d --build
docker compose -p hackathon-starter ps
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8002/ready
```

Environment: `.env` is local and private; maintain `.env.example` with placeholders. Local Python connects to `POSTGRES_HOST=127.0.0.1`, `POSTGRES_PORT=5433`. API inside Compose connects to `postgres:5432`. `POSTGRES_HOST_PORT=5433` exposes only local loopback. `uv run --env-file .env` explicitly loads local environment configuration.

**Migrations:** apply `sql/001_initial_schema.sql`, then `sql/002_memory_scoping.sql` to a fresh isolated database before memory-backed integration. Do not assume automatic migration on Compose startup. Verify `agent_runs`, `evaluation_results`, `memory_items`, `memory_selections`.

**Container validation so far (reported):** PostgreSQL healthy on 5433; Compose API started and returned HTTP 200 for `/health` and `/ready` on 8002. Readiness included `inference_enabled:false`: this is *not* a clinical/model inference success claim. DB-backed memory execution and full transactional rollback validation remain separate acceptance tests until exercised in the copied stack.

**Security:** never commit `.env`, credentials, tokens, or raw sensitive patient data. Manifest scanner flagged `POSTGRES_PASSWORD` references in Compose and repository; these are environment-variable reads, not necessarily embedded secrets. Audit actual files and regenerate inventory after restoring omitted `models/` source and modifying Compose.

## Transfer map

**Reusable unchanged or with minor wiring:** dependency management, Dockerfile, Compose pattern, FastAPI skeleton, LangGraph graph skeleton, SQL migration pattern, repository transaction pattern, test harness, evaluation registry, paired group/bootstrap methodology, metrics definitions.

**Requires task adapters:** memory schema, scoring calibration, temporal features, API request schema, domain observations, controller input covariances, utility definition, group split definition, external agent tools and data source integration.

**Experimental/optional:** HMM, Bayesian estimator, covariance trust, anticipatory model, continuous hybrid objective. Keep disabled by default unless validated against the newly available task data.

**Not transportable as conclusions:** synthetic improvements, assumed covariance alignment, simulated oracle regret, clinical validity, calibrated confidence claims, and benchmark-specific model weights.

## Acceptance checklist

- [x] Local starter dependencies and unit tests reported: 63 passed, 1 skipped.
- [x] Multi-stage Docker image built and health/readiness endpoints passed.
- [x] Isolated starter PostgreSQL healthy on localhost port 5433.
- [x] Local Python database connection tested.
- [x] Full Compose API and PostgreSQL running together; health/readiness passed.
- [ ] Verify `002_memory_scoping.sql` applied; confirm all four tables.
- [ ] Seed a **unique** demo episode, exercise `run-from-memory`, and query persisted selections.
- [ ] Run opted-in DB integration test with `uv run --env-file .env`.
- [ ] Test transaction rollback, not just the success path.
- [ ] Check no secret values or raw confidential data in package and update manifest checksums.
- [ ] Confirm a clean, fresh clone/setup without reliance on original repository state.
- [ ] Commit/source-tag the finished starter (if appropriate), separate from original project.

**Honesty boundary:** No statement in this guide should be read as evidence of clinical performance or deployable autonomous decision safety. The project is an experiment-driven agent information allocation prototype.
