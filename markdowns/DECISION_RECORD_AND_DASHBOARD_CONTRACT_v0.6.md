# Decision Record & Dashboard Contract — v0.6

Status: Proposed research architecture. Not all functionality is implemented.
Governing specification supplied by the repository owner on 2026-10-28; this
version records those requirements alongside the v0.4 mathematical architecture.
Earlier research versions remain unchanged.

## Product and target

An interpretable, time-aware preventive health research system should estimate
evolving probabilities and adapt inference only when evidence supports adaptation.
The initial dataset is PhysioNet 2019 Sepsis Challenge. The existing classifier
predicts the shifted Challenge label, not a clinically validated probability of
sepsis within the next six hours.

## Structured hourly decision record

Use additive, typed, versioned responses containing:

- Observation: pseudonymous episode, observed ICU hour, available measurements,
  missingness/freshness, dataset, feature and model provenance.
- Current prediction: static, temporal and final probabilities, disagreement,
  predictive entropy and genuinely available uncertainty diagnostics.
- Controller: allocation weights, policy/mode, implemented anticipatory/reactive/
  covariance information, reasons/constraints, emergency fallback, and whether the
  allocation is heuristic, learned or empirically validated.
- Previous decision: previous observed hour for the same episode, probabilities,
  allocation/reasons and elapsed hours.
- Next computational step: await a valid observation, update causal features and
  reevaluate. Unimplemented forecasts are explicitly unavailable.
- Interpretation: exact arithmetic decomposition and contributions, freshness and
  evidence limitations. Arithmetic attribution is not physiological causality.
- Provenance: actual model artifact SHA-256 when available, feature-builder version,
  dataset/target and research-only status. Do not invent absent provenance.

## Exact mathematical interpretation

Preserve `p(t) = S(t) + q(t) * [T(t) - S(t)]`, with temporal weight `q` and
`d(t) = T(t) - S(t)`. For successive observed predictions:

```text
p(t) - p(previous)
  = [S(t) - S(previous)]
  + q(previous) * [d(t) - d(previous)]
  + [q(t) - q(previous)] * d(t)
```

The terms describe static change, disagreement change under the previous allocation,
and controller reallocation at current disagreement. Test the identity strictly.
Across observation gaps, report elapsed time and "since previous observed hour."
Never invent missing expert predictions to complete this identity.

## Metric separation

Retrospective label-dependent evaluation includes Bernoulli log loss, Brier, AUROC,
AUPRC, prevalence, calibration/ECE, patient-mean loss, paired gains, patient-cluster
bootstrap intervals, win rates, CVaR90, adverse tails and retrospective alert burden.

Label-free inference diagnostics include controller statistics, switching variation,
model disagreement, predictive entropy, completeness, measurement ages, fallback,
invalid requests and latency. Experimental diagnostics include covariance/error
second moments, filtered HMM beliefs, conditional information and gain calibration.
Preserve Gaussian regression metrics. Future outcomes and retrospective gains must
never become online decision inputs.

## Agent and roadmap boundaries

The proposed LangGraph explainer consumes validated, sanitized records, explains
changes/freshness/evidence gaps, and records explanation provenance. It must not
override probabilities or weights, access unrestricted records, prescribe treatment
or retrain autonomously. Numerical inference remains separate.

A separately trained/evaluated next-hour forecaster, a regularized learned hybrid
comparing static/constant/hard-switch/continuous/anticipatory/reactive/covariance
policies, and a deterministic clinician-review layer with abstention are additional
experimental capabilities. Unimplemented capabilities remain unavailable; none
establishes clinical reliability or authorizes autonomous treatment.

## Integration and verification

Preserve trajectory endpoints, trained models, numerical definitions and database
schemas. Do not edit `frontend/`. The dashboard may show trajectories, previous and
current decisions, freshness, model contributions, fallback, real provenance and
limitations. Forecasts and workflow suggestions appear only when supported.

Follow AGENTS.md and the existing contracts. Establish a baseline; test deterministic
identities, causality, patient isolation, fallback and missing data; preserve Gaussian
metrics/API compatibility; run full non-database pytest, scoped Ruff and whitespace
checks. Report implemented behavior separately from proposed/unverified research.
