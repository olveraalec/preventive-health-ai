# Decision summaries for the dashboard

The supplied [v0.6 decision-record contract](../markdowns/DECISION_RECORD_AND_DASHBOARD_CONTRACT_v0.6.md)
governs this additive implementation alongside the v0.4 mathematical architecture
and Sepsis evaluation contract v0.1. No model, feature, allocation, fallback,
Gaussian metric, existing route or database schema is replaced. No frontend code
is changed. This is retrospective research tooling, not clinical decision support.

## Existing trajectory responses

Both existing routes retain their request schemas and original response fields:

```text
POST /api/v1/research/sepsis/trajectory
POST /api/v1/research/sepsis/trajectory/raw-hourly
```

Each adds `decision_summaries`, ordered one-for-one with `predictions`. Every item
has `schema_version="decision-summary.v0.6"`. The single-row legacy
`predict-engineered` route remains unchanged. OpenAPI exposes all nested types.

Use `current` and `previous` for the risk chart/allocation display. These contain
the original static, temporal and final probabilities, static/temporal weights,
status and reason. `previous` refers to the previous **observed** hour within this
same request/episode. At the first observation it and `elapsed_hours` are null;
there is no hidden server-side patient history. Send the causal history when a
previous decision is needed. A gap from hour 1 to hour 4 has `elapsed_hours=3`.

`probability_type`/provenance target remains `shifted_challenge_label`; never label
this as a validated next-six-hour sepsis probability. Static/temporal contributions
are `(1-q)*S` and `q*T`. They are arithmetic contributions, not feature importance
or medical explanations. An absent temporal probability/contribution stays null.

## Interpretation and diagnostics

`change` implements the exact ordered v0.6 identity:

```text
probability_delta
  = static_delta
  + disagreement_change_at_previous_allocation
  + reallocation_at_current_disagreement
```

The final two terms are `q(previous)*(d(current)-d(previous))` and
`(q(current)-q(previous))*d(current)`, with `d=T-S`. A residual reports ordinary
floating-point roundoff without changing any served probability. The interval is
always `since_previous_observed_hour`, never an implied one-hour forecast.

- First observation: change status is `unavailable`, reason `first_observation`,
  and all change terms are null.
- Missing expert at either observation: reason `secondary_probability_unavailable`.
  Probability/static deltas remain available; the two expert-dependent terms and
  residual are null. No zero expert or alternate decomposition is substituted.
- `diagnostics` reports signed/absolute disagreement and final-distribution
  Bernoulli entropy in **nats**, with `0*log(0)=0`. Entropy is not reliability.
  This diagnostic does not replace the notebook-compatible `entropy_static`
  feature used by the trained gain model.

`controller` identifies the existing learned gain proposal with configured
heuristic exposure/bounds. It reports the already computed proposal and gain when
available; gain is not verified live benefit. Exposure is not calibrated and the
allocation is not declared empirically validated. Configured constraints are
reported; a binding-constraint trace is explicitly not recorded. The emergency
flag describes fallback precedence over step constraints, not proof a particular
step exceeded the limit. Reasons preserve existing inference semantics.

## Data quality, provenance and unavailable capabilities

On the raw-hourly route, `data_quality.measurements` has one entry for each of the
seven vital signs. `observed` means non-null in the current supplied row;
`age_hours` is elapsed ICU hours since its last non-null observation, or null if
never observed. No imputation, resampling or future record is used. For example,
HR observed at hour 2 and missing at hour 4 yields:

```json
{"observed": false, "age_hours": 2.0}
```

`missing_count`/`missing_fraction` cover these seven vitals, not administrative or
unused Challenge columns. Freshness is based on supplied source times; actual
source availability remains unverified. The engineered route marks raw quality
unavailable (`raw_measurements_not_supplied`) rather than inferring it from filled
features. Both routes count null engineered features. Summaries omit raw values;
those remain in the authorized input rather than being duplicated in explainer
payloads or recorded experiment results.

`provenance` carries dataset/target, existing model identifiers, feature-builder
metadata/status and the manifest's artifact SHA-256 when validated by the existing
service. Without a manifest the hash is null and status `manifest_unavailable`.
An identifier is never substituted for a hash. Legacy reconstructed provenance
does not become historical training attestation.

`next_computational_step` is
`await_next_valid_observation_then_update_causal_features_and_reevaluate`.
`next_hour_forecast`, `learned_hybrid`, `workflow_recommendation`, reactive feedback
and covariance integration have explicit unavailable statuses/reasons. The existing
anticipatory gain model is not represented as the roadmap's learned hybrid.
No clinical recommendation or future risk probability is invented.

## Recorded evaluation endpoint

```text
GET /api/v1/research/evaluation-summary
```

This returns `evaluation-summary.v0.6`, with no request labels, model loading,
database query, dataset access or metric execution. It presents the **existing
document-reported synthetic** conditional-allocation study: five policies, three
recorded loss metrics per policy, and the paired gain/interval versus constant.
It uses separate metadata in `evaluation.registry.SUMMARY_METRICS`; the existing
Gaussian `METRICS` and `evaluate()` remain unchanged.

Provenance includes the exact source document path/SHA-256, six-decimal reporting
precision, split and sample count. It explicitly records unavailable numeric seed,
score clipping, CVaR convention and reference script; results were not reproduced
here. A rounded mean difference need not equal the separately rounded paired gain.
The positive-patient loss harm relative to constant remains visible.

These are simulated binary outcomes, **not Sepsis model evaluation**. Live decision
summaries mark model-bound retrospective evaluation unavailable and link to this
separate endpoint. AUROC/AUPRC/Brier/calibration/win-rate/alert-burden results are
unavailable because this versioned result does not record them. Do not display
missing values as zero, compute patient labels in live inference, or use these
historical gains to alter controller allocations.

## Exact synthetic JSON examples

- [One complete decision-summary item](examples/decision-summary.v0.6.synthetic.json),
  corresponding to `decision_summaries[1]` for the deterministic engineered fixture
  in `tests/test_decision_summary.py`. It shows the hour-1 to hour-4 transition.
- [Complete evaluation-summary response](examples/evaluation-summary.v0.6.synthetic.json).

The examples use synthetic inputs/aggregate published evidence only. Tests verify
their parity with the actual service/endpoint contracts. No raw patient/model
artifact is required or included.

Next integration: the frontend owner can consume these fields and OpenAPI. Real
model-specific evaluation requires a separately reviewed aggregate export with
model/split/metric provenance; next-hour forecasting and the learned hybrid need
their own trained/evaluated implementations. Neither is implemented here.
