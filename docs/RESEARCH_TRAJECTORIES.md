# Engineered Sepsis research trajectories

`POST /api/v1/research/sepsis/trajectory` accepts one ordered patient episode.
It does not accept raw hourly measurements as a substitute for engineered features.
The prediction target remains the shifted Sepsis Challenge label, not clinical onset.

The request declares dataset `sepsis2019`, an episode ID, feature schema,
preprocessing version, pipeline ID, causal preprocessing declaration, and requested
artifact/model version. Each observation repeats the episode ID, supplies a finite
positive ICU hour, and supplies numeric engineered features (null denotes missingness).
Hours must be strictly increasing; gaps are permitted and no hours are interpolated.
The service rejects reserved identity/time/label columns inside feature dictionaries.

The trusted local bundle must additionally supply `trajectory_metadata` matching
the request's `dataset`, `provenance`, and `model` objects exactly. The existing
single-row endpoint does not require this new metadata. No binary was generated or
migrated for this change; legacy artifacts require a deliberate provenance review
before an owner adds metadata. Clients cannot choose artifact paths through this API.

Compatibility is checked against artifact declarations and required feature names.
It is **not independent verification** of the client's preprocessing, measurement
availability, or absence of leakage. Responses explicitly report
`declared_compatible_not_independently_verified`. The service cannot establish
prefix causality of features generated outside this repository.

Example synthetic request (feature names must match the installed artifact):

```json
{
  "dataset": "sepsis2019",
  "episode_id": "synthetic-a",
  "provenance": {
    "feature_schema": "synthetic-v1",
    "preprocessing_version": "1",
    "pipeline_id": "synthetic-test",
    "declaration": "current_and_past_only"
  },
  "model": {"artifact_id": "synthetic", "model_version": "1"},
  "observations": [
    {"episode_id": "synthetic-a", "icu_hour": 1,
     "features": {"HR_current": 90, "HR_last": 88}}
  ]
}
```

The response repeats dataset, episode, provenance, and model metadata, and includes
per-hour static/temporal/final probabilities, static/temporal weights, status and
reason. It declares `research_only: true`, `clinical_use: false`, and
`probability_type: shifted_challenge_label`.

The application service reuses `predict_engineered_trajectory`, with request-local
weight history. It never imports the covariance optimizer, HMM, or database layer.
Normal inference uses `(1-q)*p_static + q*p_temporal`.

Configured research parameters, **not empirically calibrated guarantees**:

- Static fallback `q=0`; artifacts requesting another fallback are rejected.
- Reliability exposure `0.25`, not calibrated confidence or probability of correctness.
- Maximum departure from fallback `0.25`; maximum consecutive weight change `0.10`.
- Gain sigmoid temperature from the artifact, default `0.01`, finite and positive.
- First observation has no step constraint, matching the existing controller.
- Step limits apply per supplied observation, not per elapsed hour across gaps.

No verified outcomes are used. `no_verified_feedback` is the normal-operation reason.
These settings are independent of the covariance/PSD allocation policy.

Static inference runs first. Any subsequent model/controller inference exception or
invalid secondary output triggers immediate `q=0`, `final_probability=p_static`,
`status=static_only_fallback`, `reason=secondary_inference_failed`. The temporal
probability is null in this mode: a full secondary result is unavailable, including
when temporal scoring succeeded but its controller failed. The next observation
resumes from zero with the usual step limit. No prediction is fabricated when
static inference itself fails. Model exception text is not returned by the new route.
Invalid schema/provenance is rejected before prediction, not hidden as fallback.

The additive raw-hourly builder and gated endpoint are documented in
[SEPSIS_FEATURES.md](SEPSIS_FEATURES.md). Training/export provenance, source unit
and availability verification, held-out calibration/evaluation, and outcome-availability
contracts before reactive feedback remain prerequisites for trusted use.
BIDMC seconds-based telemetry remains a separate task and adapter.
