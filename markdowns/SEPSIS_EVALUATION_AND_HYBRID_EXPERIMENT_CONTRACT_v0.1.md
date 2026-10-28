# Sepsis Classification Evaluation & Hybrid-Policy Experiment Contract — v0.1

**Project:** `hackathon-starter` / `healthcare_ai_lab`  
**Date:** 2026-10-28  
**Status:** Research engineering specification; not clinical validation.  
**Scope:** PhysioNet 2019 hourly Sepsis Challenge *shifted-label* probability, candidate static/temporal policies, future hybrid supervisor.  
**Basis:** Uploaded source snapshot `evaluation/registry.py`, `evaluation/suite.py`, `notebooks/09_nested_ctonroller_benchmark.ipynb`; Codex-reported additions to raw trajectory preprocessing (not independently inspected). Align details with the current local code before implementation.

## 1. Design invariant: preserve existing regression metric semantics

The current `evaluation.registry.evaluate(y, prediction, sigma=None)` is a continuous-response/normal-model metrics function. **Do not modify or relabel** `rmse`, `mae`, `bias`, `p90_absolute_error`, `nll` (Gaussian NLL), `crps` (Gaussian CRPS), `coverage_90`, or `width_90`. In particular **`nll` must remain Gaussian NLL**; the sepsis score is `bernoulli_log_loss`.

Add *task-specific namespaced functions* (for example `evaluation/classification.py`, `evaluation/patient_classification.py`, `evaluation/controller_diagnostics.py`) and make the existing registry discover or invoke them through an explicit `task="classification"` or a dedicated `evaluate_classification` entry point. Backward compatibility is required; registration should record metric name, score direction, input type, grouping, and aggregation. Missing optional information yields a skipped/undefined metric with its reason, never silently a zero or a replacement metric. `observability.registry` remains distinct from the statistical registry.

## 2. Frozen evaluation records and primary probability semantics

Each prediction record minimally has:

```
dataset, episode_id, icu_hour, y_true, policy_id,
static_probability, temporal_probability, final_probability,
q_temporal, status, fallback_reason, artifact_version,
feature_schema_version, prediction_timestamp_or_replay_marker
```

`y_true` is **evaluation-only** and may not reach inference, gating, preprocessor, or HMM filtering. A `y_true`-absent version of this schema is valid for serving. `episode_id` is the patient-level independent unit for resampling. Keep *exactly aligned* patient-hour keys and targets across policies; report invalid/missing prediction coverage, do not silently drop different hours for different policies. Store predictions before evaluating labels. Label name: `shifted_challenge_label`, *not* clinical onset probability.

All probabilities must be finite and lie in `[0,1]` before scoring; fail on invalid inputs. For a Bernoulli log score use numerical-only clipping to `epsilon=1e-10` (matching Notebook 09), with epsilon recorded. Clipping for scoring must not secretly alter served risk probabilities.

## 3. Classification metrics and aggregation

Define `ell_it(p)=-[y_it log(p)+(1-y_it)log(1-p)]`, using score-only clipping as above. Define `b_it(p)=(y_it-p)^2`.

For `N` patients and `T_i` labeled observed hours per patient:

- `hourly_bernoulli_log_loss = sum_i sum_t ell_it / sum_i T_i` — **hour-weighted**; long admissions count more.
- `patient_mean_bernoulli_log_loss = (1/N) sum_i [(1/T_i) sum_t ell_it]` — **patient-weighted**; primary robustness objective proposed for hybrid research.
- `hourly_brier_score = sum_i sum_t b_it / sum_i T_i`.
- `patient_mean_brier_score = (1/N) sum_i [(1/T_i) sum_t b_it]`.
- `auprc` = average precision across evaluation hours, with explicit prevalence baseline. Rare-event discrimination; not equivalent to log loss.
- `auroc` only when labels contain both classes; otherwise record undefined and reason. Similarly handle one-class AP according to declared library convention; report prevalence.
- `mean_predicted_probability`, `observed_prevalence`, and `prediction_count` on the same key set.
- Calibration: fixed-width or predeclared quantile bins with counts, observed fractions, mean probabilities, and `ece` with its explicit binning rule. ECE is descriptive and sensitive to small bins/class imbalance. Calibration curves must not use validation labels to recalibrate a predictor and then report performance on the same validation set.
- Retrospective strata: `eventual_sepsis_positive_patient` versus negative patient, and positive-label versus negative-label hours, **evaluated only after prediction**. These must never be online supervisor inputs.

For BIDMC forecasting retain separate regression/Gaussian metrics and units; never report Gaussian CRPS or interval width as if they were direct Bernoulli sepsis scores. Bernoulli CRPS for a binary event equals Brier under a particular discrete definition; use the explicit `brier_score` name rather than mixing distributional conventions.

## 4. Patient-level paired analysis and tails

For reference policy `r` and candidate `c`, compute each patient's *same-hour* losses `L_i(r), L_i(c)` and define improvement `Delta_i = L_i(r)-L_i(c)` (positive is better). Report:

- `mean_paired_gain = mean_i Delta_i`, `median_paired_gain`, `win_rate = mean_i 1[Delta_i > 0]`, and optionally `tie_rate`.
- `paired_gain_ci95`: paired bootstrap at **patient level**, resampling the *entire paired patient vector* and recomputing mean gain. Default 2,000 replicates with a fixed seed (e.g., 12345). Do not resample hours.
- `worst_paired_gain`, lower p10 of `Delta_i`, bottom-decile mean gain, and descriptive worst-decile patient label composition.
- `patient_loss_p90`, `patient_loss_max`, `patient_cvar90` with patient losses sorted and **upper-tail** risk. Use the empirical Rockafellar–Uryasev representation `min_eta { eta + [1/(0.1 N)] sum_i max(L_i - eta,0) }`, or an exactly equivalent fractional-tail weighted mean. Document finite-sample convention and test it.
- For gain signs, report denominator and count; CI estimates sampling uncertainty **conditional on the fitted policies**, not the bias caused by reusing exploratory folds or tuning on the holdout.

Existing `compare_paired` assumes *minimize* directions. Add a classification-aware adapter or new comparison function rather than changing its behavior for existing regression callers.

## 5. Controller and failure diagnostics (available without labels)

From each patient sequence, report `q_mean`, `q_std`, `q_min`, `q_max`, `q_near_fallback_fraction` with predeclared threshold (provisional `abs(q-q_F)<=0.01`), `q_mean_squared_step`, `q_total_variation = sum_{t>1}|q_t-q_{t-1}|`, `allocation_step_count`, and `q_emergency_fallback_count`.

Also report candidate expert availability flags, missingness/age quality fields, `rho` exposure values, reason codes, source/model exceptions, constraint overrides, and percentage of evaluated hours with valid predictions. All controller summaries must be computed within patient before aggregation. **Weight movement is not evidence of predictive improvement.** If reliability is not calibrated, report it as `exposure` or `heuristic_trust` and not a probability of correctness.

## 6. Covariance/moment, information, and HMM diagnostics

Maintain both `Sigma = Cov(e)` and `M = E[ee^T] = Sigma + mu_e mu_e^T`; include matrix estimation population, shrinkage, eigenvalue checks, effective sample size/conditioning, and any change from retrospective to time-valid information. Estimating a patient's residual requires a *matured* outcome; no future labels for online decisions. Moment-based mixture optimization targets **Brier**, not automatically Bernoulli log loss.

Optional information metrics: `bernoulli_predictive_entropy`, residual within-patient lag correlations, specified entropy/conditional-mutual-information estimator with null permutations or bias diagnostics, conditional gain prediction error and calibration. Do not call `H(p)` a probability of model correctness, a proof of useful memory, or causal information value.

HMM state belief `b_it(k)=P(Z_it=k|X_i,<=t)` is permitted only from **filtering**, not retrospective smoothing. Record emission conventions, irregular-gap handling, fitted-on-training-only parameters and ablation provenance. Do not attach HMM information metrics to a primary deployment claim without a patient-disjoint comparison.

## 7. Candidate policies and conservative experiment

Frozen main comparison (all using identical already-fitted static and temporal base predictors):

1. `static`, `q=0`.
2. `constant_blend`, `q=q_const` selected *only on inner training/validation patients*.
3. `anticipatory_bounded`, existing time-valid learned-gain proposal plus static fallback.
4. `covariance_bounded`, uncentered-second-moment proposal with training-only shrinkage, conditioning and minimum-support fallback.
5. `hybrid_regularized` **only after** 1–4 are reproducible; optional nonnegative regime weights sum to 1 and may put full mass on fallback.

Reactive/oracle-delay is a **separate explicitly retrospective simulation**, not a clinically valid online policy absent outcome availability metadata.

Use grouped/patient-disjoint nested splits or separately frozen development/holdout splits. Tune proposals, hyperparameters, calibration, and gating **within training**. Keep a genuinely untouched final patient cohort. Report both patient-weighted and hour-weighted loss, positive-label tails, and paired gains vs static **and** the strongest training-selected simple baseline. The original nested study did not establish robust anticipatory superiority: negative evidence must remain visible.

Stop/go: new hybrid remains experimental if its paired patient gain is not credible or it worsens predeclared tail-risk constraints. Do not automatically replace static. If two experts yield nearly identical q proposals, regime weights may be non-identifiable; do not interpret them as empirical expert correctness.

## 8. Test and integration acceptance criteria

- Existing `evaluation.registry.evaluate(...)` behavior and numerical regression tests unchanged; new classification entry points do not pollute Gaussian metrics.
- Unit fixtures verify exact Bernoulli loss and Brier values; explicit clipping, invalid probability, NaN/inf, unknown label, unpaired policy, missing key, duplicate patient-hour and one-class handling.
- Same-hour alignment across all policies and correct patient/pooled aggregation on unequal-duration synthetic patients.
- Paired-bootstrap resamples patients, is deterministic under seed, returns correct sign and handling of ties; no hour-wise resampling.
- CVaR90 matches analytically checked toy patient losses; controller total variation resets between patients; fallback event count is correct.
- Synthetic prefix-causality test: extending future observations does not alter prior predictions/controller outputs; batch versus streaming equivalents on an identical preprocessor version.
- Metric reports include metric units, direction, aggregation, sample size, and reasons for undefined values.
- No implicit changes to HMM, covariance optimizer, database tables, or existing API contracts.

## 9. Engineering execution order and gate

**Gate 1 — Real artifact compatibility:** verify `artifacts/models/prototype_bundle.joblib` exists in the user's *local* environment; inspect Notebook 09's export cell, add authenticated local artifact metadata/provenance **without retraining on held-out patients**, check exact feature ordering, check package versions, and run a controlled single-patient raw-hourly vs offline-engineered prediction parity smoke test. Never load untrusted joblib files; do not commit data or model binaries. If bundle is missing, produce it using the existing training-only notebook export logic and keep dataset split records.

**Gate 2 — Reusable evaluation:** implement task-specific evaluation metrics and reports in additive modules, with tests. If artifact is missing or data are inaccessible, continue implementing metrics using synthetic fixtures, not fabricated patient performance.

**Gate 3 — Comparative policies:** train-only moment estimation and predeclared static / constant / anticipatory / covariance comparisons; no hybrid supervisor until the baseline comparisons are solid.

**Gate 4 — Demonstration and reporting:** connect one real retrospective patient trajectory (research only) to the frontend, display predicted risk, availability, q allocations, fallback, and historical evaluation context; keep label meaning clear.

## 10. Covariance-controller identifiability: an exact simplification

Let `e_S = Y-p_S` and `e_T = Y-p_T`, for the **same realized** binary label and observation. Then `e_S-e_T=p_T-p_S`, so for conditional or unconditional uncentered moments defined on the same distribution:

\[
D = m_{SS}+m_{TT}-2m_{ST} = \mathbb E[(p_T-p_S)^2].
\]

The unregularized Brier-optimal unconstrained coefficient is therefore

\[
q^* = \frac{m_{SS}-m_{ST}}{D}
= \frac{\mathbb E[(Y-p_S)(p_T-p_S)]}{\mathbb E[(p_T-p_S)^2]},
\]

followed by the designated feasible-set projection. This explicitly reveals why high residual correlation and near-identical predictions make the allocation **poorly identified**: `D` is small even though predictive risk may remain high. The numerator *does* require outcome information from valid training/cross-fitted historical examples. On a fully conditioned individual input with a deterministic gap `p_T-p_S`, the optimum depends on unknown conditional event probability; one cannot infer which model is superior from disagreement alone.

For experimental moment estimators, report the empirical denominator and supporting patient count per estimated regime, use training-only shrinkage and an explicit degeneracy threshold, and require fallback when the estimate is unreliable. Compare to a training-selected constant mixture; avoid claiming a confidence threshold is calibrated without held-out evidence.
