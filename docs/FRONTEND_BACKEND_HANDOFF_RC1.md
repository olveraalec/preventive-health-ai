# Release-candidate frontend handoff

**Readiness:** verified for local/private retrospective dashboard integration.
Not ready for public exposure or clinical use. No new model/controller, API behavior,
AWS configuration, database schema or frontend code was changed during this audit.

## Start FastAPI

From the repository root, use the existing verified environment (Python 3.12.13):

~~~bash
PYTHON_DOTENV_DISABLED=1 SEPSIS_MODEL_BUNDLE=artifacts/models/prototype_bundle.joblib uv run --no-sync uvicorn healthcare_ai_lab.api.main:app --host 127.0.0.1 --port 8000 --no-access-log
~~~

The trusted bundle and adjacent checksum-bound manifest are local delivery
prerequisites, not Git contents. Strict verified software versions remain enforced.
For a new environment, install the locked project/dev dependencies using Python
3.12.13 before running; do not silently upgrade the verified runtime.

- Health: http://127.0.0.1:8000/health
- OpenAPI: http://127.0.0.1:8000/openapi.json
- Local API explorer: http://127.0.0.1:8000/docs
- Generic /ready intentionally retains inference_enabled=false; it is not a
  capability-specific model-readiness test.

## Exact request

POST /api/v1/research/sepsis/trajectory/decision-summary
Content-Type: application/json

Optional query parameter include_contributions=true; default false.
The request model is RawTrajectoryRequest. This fully synthetic one-hour request
uses the installed model's verified metadata:

~~~json
{
  "dataset": "sepsis2019",
  "episode_id": "synthetic-frontend",
  "model": {
    "artifact_id": "notebook09-prototype",
    "model_version": "sha256:16ac3e6deb025a82de6806ecefe5c1af4b31d2742563e600385113d4caf5dde2"
  },
  "observations": [{
    "episode_id": "synthetic-frontend",
    "icu_hour": 1,
    "measurements": {
      "HR": 80, "O2Sat": 97, "Temp": null, "SBP": 120,
      "MAP": 75, "DBP": 65, "Resp": 18,
      "Age": 55, "Gender": 1, "HospAdmTime": -12,
      "Unit1": 1, "Unit2": null
    }
  }]
}
~~~

All twelve measurement keys shown above are required, even when their value is
null. OpenAPI describes a nullable numeric dictionary; these required source keys
and the allowed-name set are additionally enforced by runtime validation.
Do not send labels, engineered features, split fields or embedded patient IDs.
Optional standard unused Challenge columns are accepted but do not enter this model.

Send the full legal observation prefix on each request. The API is stateless:
previous means the previous supplied observation in the same episode. ICU hours
must be unique, strictly increasing positive integers; gaps are allowed. A single
observation produces previous=null. Model metadata must match the installed bundle.

## Exact response contract

The complete JSON Schema is
[sepsis_decision_summary.schema.json](examples/sepsis_decision_summary.schema.json);
OpenAPI exposes the same response type as DecisionSummaryResponse. The complete
synthetic request/response fixture is
[sepsis_decision_summary.synthetic.json](examples/sepsis_decision_summary.synthetic.json).
That fixture has a request/response wrapper and toy predictors; it is not a real
model result. The retrospective export below is the response DTO directly.

Top-level keys:

~~~text
schema_version, dataset, episode_id, target_semantics, model,
artifact_sha256, artifact_integrity, provenance, feature_transform_sha256,
model_feature_hashes, configured_parameters, records, capabilities,
payload_kind, research_only, clinical_use, limitations
~~~

records is an array with these exact keys:

| Key | JSON shape / meaning |
|---|---|
| current | HourPrediction object |
| previous | HourPrediction object or null |
| change | status, since, elapsed_hours, total, static_movement, disagreement_movement, controller_reallocation, identity_residual, unsupported_terms |
| diagnostics | predictive_entropy_nats, expert_gap, absolute_expert_gap, interpretation |
| information_quality | observed_vitals, total_vitals, vital_completeness, missing_vitals, availability_status |
| measurements | measurement-name map or null in sanitized exports |
| controller | policy, mode, reason, emergency_fallback, step_limit_overridden, active_constraints, evidence, feedback |
| next_step | status and then (two computational steps) |
| log_odds_contributions | static/temporal attribution map or null |

HourPrediction has exactly icu_hour, static_probability, temporal_probability,
final_probability, temporal_weight, static_weight, status and reason. Temporal
probability is nullable on fallback. Each measurement object has current_value,
missing, last_value, age_hours, last_observed_hour, freshness_status and
used_by_feature_builder. Each requested attribution has status, reason, intercept,
terms, total_log_odds, reconstructed_probability and interpretation.

capabilities contains next_hour_forecast, learned_hybrid and clinician_review.
All currently report unavailable with explicit reasons. Forecast probability and
model_version are null; horizon_hours=1 is a placeholder for a separate model.

Render null as unavailable/not applicable, never zero. On temporal failure,
unsupported change terms stay null. All three signed decomposition terms are
added. Arithmetic attribution and log-odds contributions are not medical causality.
Entropy is descriptive, not calibrated confidence. Plot against observed ICU hours
and label gaps as since the previous observed hour. The target remains the shifted
Sepsis Challenge label, not a clinically validated next-hour/six-hour onset risk.

422 means invalid/incompatible input or artifact; 503 means artifact/static
inference unavailable. Preserve the research-only flags and limitations.

## Sanitized 37-hour demonstration

Generated during this audit:

results/dashboard/release_candidate_retrospective_37h.json

To regenerate with a new filename:

~~~bash
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m healthcare_ai_lab.experiments.export_decisions --source retrospective --output results/dashboard/retrospective-rc-02.json
~~~

The exporter rechecks the existing validation replay before writing. Files are
exclusive (no overwrites), owner-readable only (0600), and Git-ignored. Source
identifiers are replaced with a demo alias; all measurement values and feature
contribution vectors are omitted. Derived trajectories remain sensitive, not
anonymous; keep the retrospective file local/private and out of Git/public hosting.
Only the fully synthetic fixture and data-free schema belong in the release.

## Audit evidence and blockers

Actual loopback Uvicorn HTTP, with the installed checksum-verified bundle:
37 matching hours; zero difference between decision-summary current predictions,
raw-hourly and engineered trajectories; prefix equality at 1/8/20 hours; episode
alias isolation passed; malformed/schema/model-version requests rejected. Normal
replay had zero fallback events; synthetic fault-injection regression tests verify
immediate static fallback and step-limit override.

Maximum decomposition residual: 4.119968255444917e-18.
Available fitted expert log-odds explanations: 74 of 74.
Prior raw/offline replay parity was reverified; independent batch temporal
difference remained 1.214306433183765e-17. Model binary and manifest hashes unchanged.
The original Gaussian registry implementation remains byte-for-byte as its source
prefix; new Bernoulli entry points are additive.

Targeted integration/metrics/fallback pytest: 147 passed.
Full workspace non-database pytest: 314 passed, 15 PostgreSQL tests skipped.
Scoped Ruff and whitespace checks passed. No live database or remote AWS/hosted CI
was exercised in this checkpoint; the owner reports ECR completion separately.
The exact staged release tree was also tested in an isolated snapshot with no
local patient data, model or notebooks: 312 passed, 17 skipped. The extra two
skips are the optional local notebook fingerprint and processed-data schema checks;
the other 15 are PostgreSQL. Scoped Ruff passed in that snapshot too.

Remaining frontend work: wire the DTO, retain null and unavailable states, use full
prefix requests, and configure a same-origin development proxy or separately
reviewed CORS/access controls. Do not enable wildcard CORS or publish physiological
responses. Authentication/authorization, request bounds, capability readiness and
production data handling remain release blockers for public use.

No forecast, hybrid controller or clinician-review policy is promoted. Real source
units/availability and historical fitting provenance retain their documented
limitations. The model-free container also requires approved model delivery and a
compatible runtime (its previously audited Python 3.12.16 differs from this model's
verified Python 3.12.13). These are integration blockers, not grounds to weaken
artifact validation.
