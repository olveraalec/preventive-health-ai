# Healthcare AI Lab — Research Index

**Last updated:** 2026-10-28  
**Project:** `hackathon-starter`  
**Purpose:** Preserve the distinction between mathematical proposals, implementation status, observed experimental results, and clinical validity.

## Research sequence

| Document | Function | Status |
|---|---|---|
| `RESEARCH_AND_TRANSFER_GUIDE.md` (if copied to this folder) | Earlier research history, assumptions, and transfer guidance | Historical source |
| `HYBRID_CONTROLLER_MATHEMATICAL_CONTRACT_v0.1.md` | Static/anticipatory/reactive/covariance proposal architecture, fallback, bounded SRM | Design contract, not validated hybrid |
| `SEPSIS_EVALUATION_AND_HYBRID_EXPERIMENT_CONTRACT_v0.1.md` | Bernoulli evaluation, patient-level paired metrics, CVaR, prediction keys, evaluation splits | Evaluation specification |
| `COVARIANCE_AWARE_CONDITIONAL_ALLOCATION_v0.2.md` | Exact moment/Brier derivation, global-versus-conditional allocation, train-only shrinkage, falsifiable ablation | New mathematical hypothesis + experimental plan |
| `docs/MATHEMATICAL_FRAMEWORK.md` | Existing repository mathematics | Review against current implementation before declaring authoritative |
| `docs/SEPSIS_FEATURES.md` | Causal feature-builder semantics | Codex-reported synthetic parity; real artifact parity pending |
| `docs/RESEARCH_TRAJECTORIES.md` | Trajectory provenance and runtime contracts | Codex-reported implementation and tests |

These links assume the historical documents are saved under the indicated names. Adjust paths if your local filenames differ.

## Recorded observations, not fresh experimental claims

- Earlier outer patient-disjoint folds did **not** establish a significant advantage of the anticipatory gate over the static model for patient-mean Bernoulli log loss. Do not erase this result when reporting the demo.
- A global unregularized uncentered-second-moment optimizer is mathematically equivalent to the same weighted training Brier-optimal constant blend; conditional allocation is the testable adaptive hypothesis.
- The controller is allowed to remain on static fallback for long intervals. Continuous evaluation does not imply compulsory weight movement.
- Codex reported 120 non-database tests passing after causal raw-hourly preprocessing and synthetic notebook parity; that report does **not** by itself establish real patient inference, model artifact validity or clinical safety.

## Engineering checkpoints

| Checkpoint | Current understanding | Evidence needed before promotion |
|---|---|---|
| Engineered inference and trajectory endpoint | Implemented and synthetic-tested, according to Codex | Local API tests and independent code review |
| Raw hourly feature builder | Codex reports synthetic notebook parity at `1e-13` | Real data / frozen artifact feature and prediction parity |
| Real retrospective model | In verification | Trusted local bundle, exact feature order/versions, patient-disjoint train provenance, single-patient smoke test |
| Bernoulli metrics registry | Specified | Additive implementation, analytic unit fixtures, grouped paired tests |
| Conditional moments | Mathematical spec v0.2 | Grouped cross-fitted experiment vs constant blends |
| Hybrid supervisor | Proposed | Only after simpler policies have credible incremental evidence |
| HMM usefulness | Filtering method exists | Training-only parameter fitting, prefix causality, independent patient-level ablation |
| Deployment | Stretch engineering task | Local end-to-end validated research demo and safe operational settings |

## New-result entry template

Create a new date/versioned Markdown entry for each completed experiment:

```
# EXPERIMENT-ID — Title
Date:
Status: proposed / implemented / evaluated / rejected / inconclusive
Question and predeclared hypothesis:
Frozen dataset and patient split:
Prediction label and temporal availability contract:
Base-model and feature artifact versions:
Optimized objective and diagnostics:
Patient-mean and hourly results:
Patient-cluster paired gains and uncertainty:
Adverse-tail results:
Assumptions, leakage checks, failure analysis:
Decision: keep / ablate further / revert
```

## Conventions

1. **Never replace a negative result with an optimistic exploratory finding.** Preserve notebooks, seeds and split identifiers.
2. Use `proposed`, `implemented`, `evaluated`, `inconclusive`, and `rejected` literally; only measured outcomes go under `evaluated`.
3. Only promote a research equation to a runtime default after tests, provenance verification and predeclared patient-disjoint evaluation.
4. Never portray the shifted PhysioNet Challenge label as a clinically validated sepsis-onset probability.
5. Keep `markdowns/` as the research log; keep `docs/` for implementation-facing documentation. Cross-link both rather than keeping two conflicting authoritative copies.
