# Sepsis decision summaries for the dashboard

This is an additive interpretation adapter, schema version
**sepsis-decision-summary.v1**, guided by architecture v0.4 and the v0.6 Decision
Record & Dashboard Contract supplied in the conversation. The installed prediction
models, feature definitions, controller, metrics registry and existing endpoints
are unchanged. This is retrospective research, not a clinical decision service.

## API and schema

POST /api/v1/research/sepsis/trajectory/decision-summary

Request: the existing RawTrajectoryRequest (same dataset, episode, model metadata
and ordered raw observations as the raw-hourly endpoint). Optional query parameter
include_contributions=true enables fitted expert log-odds explanations. It is false
by default to keep the response compact.

The endpoint reuses predict_raw_trajectory as its sole prediction authority.
It reuses the versioned feature builder for freshness and optional attribution;
there is no new prediction/model-fitting implementation. Compatibility failures
return 422; unavailable artifacts/static inference return 503 without source
records or model tracebacks. A secondary failure returns the existing immediate
static fallback, with unsupported explanations explicitly null.

The complete synthetic request/response is in
[sepsis_decision_summary.synthetic.json](examples/sepsis_decision_summary.synthetic.json).
It has three observations at hours 1, 2 and 4, missing measurements, fixed synthetic
predictors and declared-only artifact provenance. Those predictors are demonstration
arithmetic, not fitted Sepsis models. The fixture is for offline dashboard/test use;
a live server requires the model metadata of its installed trusted artifact.

Response JSON Schema:
[sepsis_decision_summary.schema.json](examples/sepsis_decision_summary.schema.json).
The same typed contract is exposed through the application's OpenAPI components
as DecisionSummaryResponse. JSON null always means unavailable/not applicable,
never a fabricated zero. The API emits JSON numbers and explicit status/reason
fields; no Python callables, DataFrames or NaN values are serialized.

Each record includes:

- current and previous: observed ICU hour, static/temporal/final probabilities,
  static/temporal weights and original controller status/reason.
- change: total probability change, three signed decomposition terms, elapsed
  hours and identity residual. The previous point is the previous **supplied
  observed** hour of this episode, even across gaps. First-observation previous
  and change values are null.
- diagnostics: Shannon predictive entropy in nats, signed T-S and its absolute
  magnitude. These are label-free descriptors, not calibrated reliability.
- measurements: current supplied values, explicit missingness and, for the seven
  builder vitals, last observed value, elapsed age and last observed ICU hour.
  Historical values are distinct from current observations and imputer values.
- information_quality: observed-vital count, completeness fraction, missing
  vital names, and explicit unverified source units/availability status.
- controller: existing policy/mode/reason, emergency status, binding constraints
  and whether emergency failover exceeded the ordinary step limit. Constraint
  settings are read from the existing trajectory response, not reconfigured.
- next_step: await the next valid observation, then update causal features and
  reevaluate inference/allocation. It never peeks at the next row in the batch.
- log_odds_contributions: opt-in per-expert explanations, otherwise null.

Shared response metadata includes dataset/target semantics, model version,
artifact SHA256 when available, feature-builder provenance/transformation hash,
ordered model-feature hashes, configured controller parameters, limitations and
research_only=true, clinical_use=false. A legacy/synthetic bundle without a
verified manifest explicitly reports declared_only and a null artifact hash;
a matching metadata declaration is not independent preprocessing certification.

## Exact arithmetic and optional fitted attribution

The preserved mixture is p = S + q*(T-S). All three signed terms are **added**:

    delta_p = (S_current - S_previous)
            + q_previous*(d_current - d_previous)
            + (q_current - q_previous)*d_current

Here d=T-S. The adapter checks the residual to absolute tolerance 1e-12; seeded
identity tests check 500 random cases plus endpoints to 1e-14. For gaps, changes
are since the previous observed hour, not normalized per elapsed hour.

If either temporal probability is unavailable, total/static movement remain
known, while disagreement_movement, controller_reallocation and identity_residual
are null with partial_temporal_unavailable and explicit unsupported_terms.
The adapter does not invent a temporal prediction even when its multiplier is zero.

For the supported fitted median-imputer -> StandardScaler -> binary
LogisticRegression pipeline, contributions are:

    transformed = fitted_preprocessing.transform(features_in_training_order)
    term_j = fitted_coefficient_j * transformed_j
    expert_log_odds = fitted_intercept + sum_j term_j

The reconstructed sigmoid must match the served expert probability within 1e-12.
Unsupported preprocessing, missing expert predictions or failed introspection
produce explicit unavailable explanations without altering inference. All terms
are returned, without top-k truncation. Imputed features can contribute despite
a missing current observation. These terms explain each expert's log odds;
they are not probability contributions or the blended predictor's log odds.

Arithmetic attribution and fitted coefficients are not causal physiological
explanations. Entropy is not a probability that a model is correct.

## Explicit capability boundary

The shared capabilities object applies to every record:

- next_hour_forecast: unavailable, probability/model version null; horizon 1 hour
  and target shifted_challenge_label_at_t_plus_1 are declared placeholders for a
  separate, independently evaluated model. Current risk is never substituted.
- learned_hybrid: unavailable for serving. The earlier allocation experiments
  do not establish a validated supervisor.
- clinician_review: unavailable/abstain because no validated recommendation
  policy exists. No clinical thresholds, diagnoses, test orders or treatments
  are inferred by this adapter.

The existing prediction remains the shifted Challenge label at the observed
decision time, not a validated next-hour or next-six-hour clinical-onset risk.
Measurement ages are proxies based on supplied observation hours. Actual event/
availability timestamps and source units are not independently verified. Vital
freshness comes directly from the existing builder; administrative and optional
unused Challenge columns retain their current values but report not_tracked
freshness. Optional unused measurements are explicitly marked as unused by the
feature builder, and never flow backward into earlier records.

## Reproducible fixtures and retrospective export

From the repository root, in the existing verified Python 3.12 environment:

~~~bash
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m healthcare_ai_lab.experiments.export_decisions --source synthetic --output results/dashboard/synthetic-01.json
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m healthcare_ai_lab.experiments.export_decisions --source schema --output results/dashboard/schema-01.json
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m healthcare_ai_lab.experiments.export_decisions --source retrospective --output results/dashboard/retrospective-01.json
~~~

Choose a fresh output name: exports refuse to overwrite files. Synthetic output
contains request and response; retrospective output is directly the response DTO.

Retrospective mode requires the existing trusted local model and checksum-bound
manifest plus the manifest-matching raw and engineered development Parquet files.
It reruns the existing raw/offline validation-patient parity smoke, chooses that
same deterministic validation episode, replaces its source identifier before the
API call, and invokes the new endpoint. No training or artifact activation occurs.
The untouched test cohort is not selected.

Before writing, it removes all physiological measurement values and all log-odds
contribution vectors. Only derived prediction/controller/change and aggregate
information-quality summaries remain. The episode uses a constant demo alias.
Retrospective exports are restricted to the existing ignored results directory,
created exclusively with mode 0600, and cannot be exported into docs or frontend.
The committed example/schema are entirely synthetic/data-free.

**Derived patient trajectories are still sensitive and are not anonymous.**
Keep retrospective output local/ignored; do not commit, publish, upload or log
it as a raw API response. Source identifiers, labels, demographics, physiology,
per-feature transformed values and contribution vectors are not exported.
The CLI prints aggregate verification evidence only.

Missing/mismatched artifacts or data stop export with static diagnostics.
The existing strict model/software compatibility rules remain active; the
model-free ECR image and a mismatched Python patch version do not establish
model readiness.

## Frontend handoff

No frontend files were edited. The teammate can:

1. Consume the synthetic wrapper's response or the sanitized retrospective DTO,
   using schema_version to select the contract.
2. For live local inference, post the full legal observation prefix using installed
   model metadata. The service is stateless; a one-row request cannot recover a
   previous decision or omitted history from another request.
3. Plot current.final_probability against current.icu_hour, show previous/change,
   distinguish observed from last-known values, and retain null/unsupported states.
   Use elapsed_hours for gap labeling; do not label every transition one hour.
4. Show the shared capability states, shifted-label semantics, provenance and
   research limitations. Do not turn next_step into a clinician recommendation.
5. Opt into contributions only when needed; preserve signed terms and intercept.
   Sanitized exports intentionally omit those potentially invertible explanations.
6. Coordinate the API base URL, authenticated/private access and data handling
   before any remote dashboard use. No authentication, CORS, public exposure,
   persistence or deployment changes are included in this milestone.

## Executed verification and remaining gaps

Baseline: 280 passed, 15 opt-in PostgreSQL cases skipped. Added 34 deterministic
tests covering identities, gaps, missing sources, fallback/rate-limit override,
active constraints, patient/request/prefix isolation, fitted imputation/log-odds
identities, API compatibility/errors/OpenAPI, unavailable capabilities, output
redaction/permissions/exclusive writes, source-integrity gates and provider-free
numerical imports. Existing Gaussian metric definitions were not edited.

Final executed checks: targeted decision-summary/export pytest **34 passed**;
full non-database pytest **314 passed, 15 skipped** (unchanged opt-in PostgreSQL
cases). Scoped Ruff on all changed Python modules and the CI clean scope passed;
git diff --check and additional new-file whitespace checks passed. The committed
synthetic example and JSON Schema match the runtime JSON contracts. The private
retrospective export's redaction, 0600 permissions and Git-ignore status were checked;
the model binary and manifest hashes remain unchanged.

The local retrospective export covered 37 hours. Prior raw/offline feature,
static/temporal/final prediction and allocation differences remain zero; independent
batch temporal difference is 1.2143e-17. Maximum change-decomposition residual:
4.11997e-18. All 74 static/temporal log-odds explanations were available before
redaction. These are engineering parity results, not calibration, clinical validity
or evidence of adaptive superiority. The generated local payload is
results/dashboard/decision_summary_v1_retrospective.json and is Git-ignored.

Read: root AGENTS, research reading map/index, architecture v0.4, v0.6 supplied
in the conversation, and the existing feature/artifact contracts and implementation.
Mixture, controller bounds/fallback precedence, HMM, covariance optimization,
Gaussian/Bernoulli metrics, dataset distinctions, trained artifacts, PostgreSQL
schemas and AWS configuration are preserved. Hosted CI and live PostgreSQL were
not exercised for this change. Forecasting, learned hybrid serving, calibrated
uncertainty and clinician-review policy validation remain outside this adapter.
