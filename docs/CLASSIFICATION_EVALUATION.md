# Bernoulli and patient-level evaluation — metric version v1

This additive evaluation layer implements the metric portions of the v0.4 research
architecture and v0.1 Sepsis Evaluation & Hybrid Experiment Contract. It scores frozen
research predictions, not clinical onset risk. It does not fit policies, call models,
run controllers, invoke LangGraph/LLMs, load model artifacts, or persist records.

Documents consulted: root `AGENTS.md`, `markdowns/CODEX_RESEARCH_READING_MAP.md`,
`markdowns/00_RESEARCH_INDEX.md`,
`markdowns/RESEARCH_MIDDLEWARE_ARCHITECTURE_AND_EXPERIMENTS_v0.4.md`,
`markdowns/SEPSIS_EVALUATION_AND_HYBRID_EXPERIMENT_CONTRACT_v0.1.md`, and
`docs/MATHEMATICAL_FRAMEWORK.md`. Historical research files are unchanged.

## Entry points and compatibility

```python
from healthcare_ai_lab.evaluation.registry import (
    classification_metric_definitions,
    evaluate_classification,
    evaluate_classification_policies,
)
from healthcare_ai_lab.evaluation.contracts import EvaluationConfig

scores = evaluate_classification([0, 1], [0.25, 0.75])
assert scores.metrics["brier_score"].value == 0.0625
```

The original `METRICS`, `MetricSpec`, `evaluate(y, prediction, sigma=None)`, Gaussian
RMSE/MAE/bias/NLL/CRPS/90% coverage/width, and regression paired comparison remain
unchanged. In particular `nll` still means **Gaussian NLL**. New definitions are
separate and discoverable; derived patient/controller definitions accompany every
result. Every result includes direction, objective/monitoring role, aggregation,
unit, clipping where relevant, metric version, observation/patient counts, and an
explicit undefined reason. Reports serialize through `model_dump()` or
`model_dump_json()`; no NaN/Infinity or callable functions appear in their JSON.

`report.metric_records()` emits long-form policy, retrospective-subgroup and paired
comparison records with experiment, partition, policy/model/feature provenance and
metric metadata. It omits patient IDs. Detailed per-patient summaries remain in
`report.policies[*].patients` for authorized local analysis; never log or publish
those summaries with real identifiers.

## Definitions actually implemented

For binary y and finite probabilities p in [0,1]:

| Metric | Definition / convention | Direction and use |
| --- | --- | --- |
| `bernoulli_log_loss` | Mean `-y log(p_clip) -(1-y) log(1-p_clip)`; default epsilon `1e-10` | Minimize, scoring objective |
| `brier_score` | Mean `(y-p)^2`, without clipping | Minimize, scoring objective |
| `auprc` | Noninterpolated average precision from sklearn; not trapezoidal PR area | Maximize, monitoring |
| `auroc` | ROC area, including rank ties | Maximize, monitoring |
| `ece_binned` | Sum over bins `n_bin/N * abs(mean_p - fraction_positive)` | Minimize, monitoring |
| Mean predicted probability, observed prevalence, count | On the identical accepted key set; prevalence is the AP baseline | No optimization direction, monitoring |
| `predictive_entropy_mean` | Mean `-p log(p)-(1-p) log(1-p)`, endpoints zero | Monitoring, not reliability |

Clipping is **score-only** and recorded; probabilities and served predictions are
never modified. Evaluation entropy is ordinary Shannon entropy; it does not replace
the notebook-compatible clipped entropy used as an inference-controller input.
Calibration uses 10 fixed-width bins by default: edges `j/K`, bins `[left,right)`
except the last includes 1. Empty bins have count zero and null means. No calibration
model is fitted on holdout outcomes. Calibration slope/intercept and their uncertainty
are not implemented in this slice.

**Version clarification:** v0.1 allows a declared single-class AP library convention;
the newer primary v0.4 specifies undefined discrimination on single-class cohorts.
Both AUPRC and AUROC therefore return null with `single_class_cohort` for all-positive
or all-negative inputs. Proper scores, prevalence and calibration remain defined.
No existing notebook/scoring function is rewritten to enforce this new convention.

For patient i with T_i accepted hours, first compute `L_i = mean_t loss_it`.
`hourly_*` scores pool hours; `patient_mean_*` scores average L_i equally across
patients. Long admissions do not dominate the latter. Report linear-interpolated
patient loss p90, maximum patient loss, and empirical upper-tail CVaR90 for both
Bernoulli loss and Brier. CVaR uses exactly

```text
min_eta eta + sum_i max(L_i-eta, 0) / (0.1*N)
```

with an empirical inverse-CDF threshold. This includes fractional boundary mass:
for 11 sorted patient losses 0,...,10, CVaR90 is `(10 + 0.1*9)/1.1`, not the mean
of an arbitrarily rounded top-two group. With one patient CVaR equals that loss.

Paired gains are `L_i(reference) - L_i(candidate)`, so positive improves either
lower-is-better loss. Reports include mean, median, strict win/loss/tie rates, win
count, worst gain, linear p10 and exact fractional bottom-decile mean gain
`-CVaR90(-gains)`. Both policies must score the **same hours of each patient**.

The 95% percentile bootstrap uniformly resamples complete patient gain vectors,
2,000 replicates by default with seed 12345. Patients are sorted for deterministic
row-order invariance; hours are never resampled independently. One patient yields
undefined CI bounds with `fewer_than_two_patients`, not a misleading zero-width CI.
Intervals condition on already fitted policies and do not correct exploratory tuning
or holdout reuse.

An additional two-sided paired sign-flip test randomizes **whole patient** gain
signs, assuming exchangeability of paired policy labels under the null. It enumerates
all `2^N` flips for N<=12; otherwise uses 2,000 seeded Monte Carlo flips and the
`(exceedances+1)/(replicates+1)` correction. Absolute-mean comparisons use a declared
`1e-15` numerical tolerance. This is monitoring evidence, not a reliability guarantee
or an automatic policy promotion rule. Resampling is chunked to bound temporary memory.

Retrospective positive/negative-patient and positive/negative-label-hour strata report
proper scores on their own denominators. Empty strata are explicitly undefined.
Eventual-positive flags are computed only for reporting, never inference or allocation.

## Controller diagnostics

`controller_diagnostics()` needs no labels and consumes one episode's ordered q
values. The patient runner sorts by episode/hour and resets differences between
episodes. It reports population (`ddof=0`) mean/variance/std, min/max, mean absolute
change, mean squared change, total variation and number of within-episode transitions.

Two previously unspecified switching notions are kept distinct:

- `allocation_change_frequency`: fraction of adjacent observations with
  `abs(delta_q)>allocation_change_tolerance` (default zero).
- `expert_switching_frequency`: fraction where `(q>=switching_threshold)` changes;
  threshold defaults to 0.5 and a tie belongs to the temporal expert.

Both are descriptive configured conventions, not validated thresholds. Gap steps
compare adjacent supplied observations; they are not divided by elapsed ICU hours.
Single-hour patients have zero total variation and zero transitions; transition
means/frequencies are undefined rather than fabricated zeros.

`fallback_frequency` counts emergency fallback **hours**, and its count is separately
reported. It differs from `static_mode_fraction` (`q==0`) and `q_near_static_fraction`
(`q<=0.01`, configurable). Consecutive failed hours count separately. Missing temporal
predictions are never fabricated: disagreement uses only available expert pairs and
reports their denominator/availability fraction. Both mean squared and absolute
static/temporal disagreement are supplied. No movement/fallback metric is treated as
evidence of predictive improvement.

Policy-level diagnostics average defined within-patient summaries, with the actual
number of contributing patients recorded. Empty transition/expert denominators are
explicitly undefined. Total emergency and transition counts are also reported.

## Patient-time and leakage contract

The scorer accepts **two separate collections**:

- `PredictionRecord`: dataset, partition, patient-hour key, decision time,
  observation cutoff, frozen policy/artifact/feature versions, static/optional
  temporal/final probabilities, q, status/reason and optional feedback references.
  It forbids outcome labels and unknown fields. The declared mixture is checked to
  absolute tolerance `1e-12`; static/temporal/hard-switch/constant policies enforce
  their declared allocation semantics. Emergency failure requires q=0 and a reason.
- `OutcomeRecord`: the same patient-hour key, binary label, target/event time,
  available-at time and declared availability basis. No inference function receives it.

Times are finite **episode-relative ICU hours**, not mixed UTC/Unix timestamps.
The observation cutoff cannot exceed decision time; decision cannot precede its
observation hour or exceed evaluation cutoff. Outcome availability cannot precede
target/event time, and labels must have matured by the evaluation cutoff.

Feedback may reference only earlier hours of the **same episode** in the ledger.
It must have matured by the relevant decision, not merely by evaluation time.
`verified_availability` additionally requires `verified_independent` availability
basis. Retrospective challenge labels default to `retrospective_only`; using them
as delayed oracle feedback requires the explicit `oracle_delay_simulation` protocol,
which is flagged in the output. No label-offset or clinical-onset timestamp is inferred.

Duplicates, unknown labels, NaN/Inf, inconsistent experts/versions/clocks, mixed
partitions and missing/misaligned keys fail before scores are produced. Default
expected keys are the full supplied outcome ledger. Supply `expected_keys` explicitly
when the ledger also contains historical feedback-only rows, and `expected_policy_ids`
to detect an entirely omitted planned policy. `CoverageError.coverage` reports expected,
supplied and matching counts/fractions without patient IDs. No policy-specific rows
are silently dropped. Accepted coverage is 100% **of the declared key set**, not a
claim of complete real-world cohort follow-up.

Availability, artifact version and split metadata remain source assertions. Validation
cannot prove historical training separation or detect someone forging times/labels
before freezing predictions. There is no label-availability adapter for the local
real cohort yet. No real-patient evaluation was performed for this milestone.

## Synthetic static versus continuous example

The unit fixture has two synthetic patients with one and three hours, fixed experts
S=0.2 and T=0.8. Labels are `[0]` and `[0,1,1]`; continuous weights are `[0.25]`
and `[0.25,0.15,0.25]`. Static uses q=0. No training is performed.

| Score | Static | Continuous |
| --- | ---: | ---: |
| Hourly Bernoulli loss | 0.916291 | 0.787316 |
| Patient-mean Bernoulli loss | 0.685242 | 0.668471 |
| Patient-mean Brier | 0.240000 | 0.236100 |
| Patient CVaR90 Bernoulli loss | 1.147340 | 0.906160 |

Patient mean log-loss gain is `0.016770`, 95% patient-bootstrap CI
`[-0.207639, 0.241180]`; win rate is 50%, worst gain `-0.207639`, and exact paired
sign-flip p-value is 1. This deliberately shows a harmed patient and weak evidence
despite a positive mean. It establishes scoring behavior, not policy effectiveness.

## Frontend coordination and remaining integration

The inspected `frontend/src/App.tsx` is the Vite starter; no fetch/axios client,
evaluation DTOs or metric-rendering API contract exists in the current checkout.
No frontend file was modified. Existing FastAPI routes return inference/run results;
there is **no evaluation HTTP endpoint** added by this task.

For teammate coordination, the proposed JSON consumer shape is:

```text
EvaluationReport
  experiment_id, dataset, split, target_semantics, metric_version, evaluated_at
  config: epsilon, calibration_bins, seeds/resampling and diagnostic conventions
  policies[]: policy_id/kind/versions, declared coverage, metrics, calibration,
              retrospective_strata, authorized local patient summaries
  comparisons[]: reference, candidate, loss, sample counts, resampling settings,
                 metrics, uncertainty_scope
  research_only=true, clinical_use=false, availability_status

MetricValue
  metric: metric_id, direction, role, unit, aggregation, distribution,
          probability_clip, metric_version, undefined_behavior, input shape
  value: number|null
  status: defined|undefined
  reason: string|null
  n_observations, n_patients
```

The frontend must preserve undefined values/reasons rather than coercing null to
zero; distinguish monitoring from optimization and patient-weighted from hourly
scores; display the CI and adverse tail alongside mean gains; and keep the shifted
label/research-only semantics visible. Generate a machine-readable schema with
`EvaluationReport.model_json_schema()` when coordinating a transport adapter.

Current trajectory output is **not** directly an evaluation input: it lacks labels,
outcome availability, partition and policy-version declarations; names such as nested
`temporal_weight` need an explicit adapter to `q_temporal`. That adapter must not
invent provenance or label availability. Coordinate a sanitized, authenticated report
endpoint and frontend types in a future task rather than changing existing routes.

Calibration slope/intercept, automatic constant-blend selection, conditional moments,
HMM/MI estimation, learned hybrid supervision, new persistence and real-cohort scoring
are outside this milestone. Gaussian definitions, inference, API contracts, database
schemas, HMM, covariance optimizer, controller behavior and model artifacts are unchanged.

## Executed verification

Baseline: 136 passed, 15 opt-in database cases skipped. Final targeted suite:
70 passed, zero skipped. Final full database-disabled suite: 206 passed, 15 skipped
(the unchanged opt-in PostgreSQL integration cases). Scoped Ruff and
`git diff --check` passed. Tests cover analytic score/calibration/CVaR values,
Gaussian numerical regression, Brier gain identity, endpoint/invalid inputs,
one-class cohorts, row/patient permutation invariance, patient-cluster bootstrap,
exact/Monte Carlo patient sign-flips, metadata/key/clock failures, delayed feedback,
emergency fallback, missing experts, all six policy kinds and provider-independent
imports. All committed fixtures are synthetic.

No PostgreSQL behavior changed, so live integration was not rerun for this milestone.
Existing trained artifacts were not loaded or retrained for metric evaluation,
and no real patient data were scored. Hosted CI, real-cohort
metric results, HTTP evaluation transport and frontend rendering remain unverified.
