# Research Middleware Architecture and Controlled Experiments — v0.4

**Project:** `hackathon-starter` / `healthcare_ai_lab` / `preventive-health-ai`  
**Date:** 2026-10-28  
**Status:** Proposed mathematical and engineering architecture, not an implemented or clinically validated system.  
**Scope:** Causal retrospective PhysioNet 2019 *shifted challenge-label* probability prediction; BIDMC second-scale continuous forecasting is a distinct model family.  
**Supersedes:** Nothing. This is a new architectural iteration and an experiment specification alongside the prior v0.1–v0.3 contracts.

## 0. Research objective and evidence boundary

**Central hypothesis:** *A model-allocation or information-selection policy can improve the reliability of a simple reference predictor only in identifiable contexts where the expected gain of adaptation exceeds estimation uncertainty, risk, and operating cost.*

The architecture should be valuable even if the empirical result is negative: it exposes faithful uncertainty diagnostics, causal patient replay, policy comparisons, and safe static fallback.

**Existing reported evidence:** A static/temporal prototype bundle and a 37-hour validation-patient replay have reportedly demonstrated runtime/offline parity; 136 non-database tests reportedly passed after artifact activation. Those are Codex-reported engineering results, not independent clinical validation. Prior outer patient-disjoint comparison did **not** establish a reliable patient-average log-loss advantage for the learned anticipatory policy. Positive-patient tails were problematic. Other proposed layers below remain research hypotheses.

## 1. One-way causal boundary: observation, availability, labeling

For patient/episode \(i\) and decision hour \(t\):

- \(\mathcal F_{i,t}\): all data **available for use at decision time** \(t\). Each measurement has an event timestamp and, when different, an availability timestamp.
- \(X_{i,\le t}\): legal hourly telemetry history. \(\phi_t(X_{i,\le t})\): exact versioned, training-compatible engineered features.
- \(Y_{i,t}\in\{0,1\}\): retrospectively observed *shifted Sepsis Challenge label*. This is not an observed real-time clinical event probability or a treatment recommendation.
- \(S_{i,t},T_{i,t}\): frozen static and temporal model probabilities; both must be measurable with respect to \(\mathcal F_{i,t}\).
- \(q_{i,t}\in[0,1]\): final temporal expert allocation; \(p_{i,t}=S_{i,t}+q_{i,t}(T_{i,t}-S_{i,t})\).

An observed outcome can only support a reactive controller at decision \(t\) if its availability timestamp is \(\le t\). A retrospectively shifted label cannot be used as online feedback unless its independent availability mechanism is justified. Replaying a patient trajectory never authorizes exposing their future label to the policy.

**Causal invariant:** For two sequences with identical legal prefixes through \(t\), all emitted features, filtered states, allocations, probabilities, and explanations through \(t\) must agree (subject to declared deterministic numerical tolerance), regardless of their different futures.

## 2. Numerical inference core vs. agentic orchestration

**Deterministic data plane** (normal path):

```text
Raw hourly episode
  -> timestamp and provenance validator
  -> exact causal Sepsis feature builder (versioned)
  -> frozen static + temporal prediction experts
  -> optional HMM filtered belief / reliability descriptors
  -> independently testable controller policy candidates
  -> policy supervisor + feasibility + emergency fallback
  -> versioned probability/decision output
  -> metrics event / research-only audit record
```

**Agentic control plane** (LangGraph; complementary, not predictive authority):

```text
validate_request -> load_research_context -> invoke_numerical_service
         -> verify_provenance_and_contracts
         -> [optional] build_read_only_explanation
         -> [optional] require_human_review_for_external_action
         -> record_audit_summary -> END
```

Existing `agent/workflow.py` already implements a *different* `observe -> estimate -> allocate -> execute -> verify` research workflow based on provided error moments and memory selection. Preserve that workflow and use adapters or a separate graph for the Sepsis inference path; do not silently replace its semantics.

### Why this separation matters

The LLM may: retrieve approved research documents; summarize audited outputs; generate explanations that identify uncertainty and limitations; orchestrate approved read-only evaluation queries. It must **not** invent vitals, infer labels, overwrite probabilities, choose safety thresholds by free-form prose, authorize clinical action, or silently change model weights. Numerical predictions and fallback are implemented as ordinary Python services that operate without any LLM provider.

A LangGraph checkpoint stores *workflow state*, not a model's learned HMM belief and not necessarily long-term patient memory. Use typed immutable/versioned artifacts for model state and an explicit timestamp-aware memory service. For replay, graph state and database queries must be scoped by episode and availability time. Use a persistent checkpointer only after defining privacy, retention, migration, and thread isolation policies; development-only in-memory state is acceptable for synthetic tests. The production-side database schema remains opt-in and unmodified at this phase.

### Minimal typed output contract

`dataset`, `episode_id` (pseudonymous), `decision_icu_hour`, `observation_cutoff`, `feature_builder_version`, `feature_schema_hash`, `model_manifest_sha256`, `target_semantics`, `static_probability`, `temporal_probability|null`, `final_probability`, `final_temporal_weight`, `policy_id`, `fallback_status`, `fallback_reason`, `state_belief|null`, `diagnostics`, `research_only=true`, `clinical_use=false`, and optionally a sanitized `trace_id`.

No arbitrary JSON from a language model can replace these validated fields. Check finite probability bounds, time ordering, patient isolation, and actual artifact compatibility.

## 3. A hierarchy of prediction and policy proposals

Set \(d=T-S\). Define **distinct concepts**:

1. **Operational safety fallback:** \(q_F=0\) (static-only); immediately invoked when the secondary path fails. If the static model itself fails, return a documented unavailable/error outcome, not a fabricated risk.
2. **Statistical reference:** \(q_0\), a training-only selected feasible **constant** blend, potentially different from zero. It is the challenging nonadaptive research comparator.
3. **Anticipatory proposal:** \(q_A(z_t)\), trained from prior cross-fitted evidence; only time-legal inputs.
4. **Covariance-aware proposal:** \(q_C(z_t)\), estimated from training-only error second moments, with shrinkage and denominator support safeguards.
5. **Reactive proposal:** \(q_R(z_t)\), enabled only with independently verified, matured outcome feedback. Until then it is a separately labeled simulated/oracle-delay experiment, not a live policy.

A proposed future supervisor uses \(\pi_t=\operatorname{softmax}(g_\varphi(z_t))\) over *available* proposals (mask missing ones), \(\tilde q_t=\sum_k\pi_{k,t}q_k\), and applies the jointly feasible projection \(q_t=\Pi_{\mathcal Q_t}(\tilde q_t)\). Softmax weights are proposal-combination coefficients, **not** calibrated probabilities that any expert is correct. When proposals are equal or nearly equal, weights may be nonidentifiable even if predictions are stable.

**Staged adoption:** static -> train-selected constant blend -> one preregistered conditional covariance rule -> optional HMM diagnostic -> small regularized supervisor. Do not implement all policies in one unconstrained optimization search.

## 4. Candidate value: exact math for Brier and log score

For a candidate \(q_1(z_t)\), reference \(q_0\), and proposed movement \(\delta=(q_1-q_0)d\), let \(p_0=S+q_0d\) and \(p_a=p_0+a\delta\), \(a\in[0,1]\).

### Exact Brier gain

\[
\Delta_B(a\mid z)=\mathbb E[(Y-p_0)^2-(Y-p_a)^2\mid Z=z]
=2aN(z)-a^2D(z),
\]

\[
N(z)=\mathbb E[(Y-p_0)\delta\mid Z=z],\qquad
D(z)=\mathbb E[\delta^2\mid Z=z].
\]

For \(D>0\), \(a_B^*=\operatorname{clip}(N/D,0,1)\). When \(D\approx0\), abstain from adaptation; an unstable ratio is not evidence. For the global direct static/temporal mixture this reduces to the constant Brier-optimal blend and is mathematically equivalent to unregularized global error-second-moment optimization under identical observation weights.

### Bernoulli log score: the primary sepsis outcome

Define \(\ell(y,p)=-y\log p-(1-y)\log(1-p)\), with a **predeclared** probability-clipping rule for numerical evaluation. Along the blend path:

\[
\frac{\partial}{\partial q}\mathbb E[\ell(Y,S+qd)\mid Z=z]
=\mathbb E\!\left[\frac{d\,(S+qd-Y)}{(S+qd)(1-S-qd)}\middle|Z=z\right].
\]

The second derivative is nonnegative whenever \(0<S+qd<1\), so the population conditional log-loss objective is convex in the scalar mixture weight. The Brier-optimal and log-loss-optimal allocations need not agree. Therefore **fit** a covariance candidate with Brier-consistent moments, but **select and compare** actual policies using prespecified patient-mean Bernoulli log loss on data outside the candidate-fitting partition.

A claimed conditional gain must survive comparison to the strongest inner-training-selected **constant** mixture, not just static. Patient-level harms and uncertainty intervals remain mandatory.

## 5. A more precise role for the HMM: sufficient representation, not magic new information

Suppose \(H_t\) is the complete legal observation history and \(b_t=f_{\widehat\theta}(H_t)\) is a **deterministically computed** filtered HMM belief after fitting \(\widehat\theta\) using permitted training data.

**Data-processing fact:** Conditioned on the entire \(H_t\), a deterministic \(b_t\) carries no additional information:

\[
I(Y_t;b_t\mid H_t)=0.
\]

This does **not** imply the HMM is useless. If the existing predictor sees only a compressed feature vector \(C_t=\phi(H_t)\), the HMM may capture different aspects of the history, and in general

\[
I(Y_t;b_t\mid C_t)\ge0
\]

may be strictly positive. The HMM can create a **useful inductive bias or compressed state representation**, not information from nowhere. A finite sample model may benefit even when the population information content of the full history is unchanged.

### Filtering equations

For discrete latent state \(Z_t\in\{1,\ldots,K\}\), transitions \(A\) and observed compatible emission \(L_k(x_t)\):

\[
b^-_t(k)=\sum_j b_{t-1}(j) A_{jk},\quad
b_t(k)=\frac{L_k(x_t)b^-_t(k)}{\sum_l L_l(x_t)b^-_t(l)}.
\]

Missing dimensions contribute no observation likelihood (under explicitly modeled assumptions); entirely missing observations yield prediction-only updates. Verify the original module's first-observation convention. If ICU hour gaps occur, either use a declared transition operator for elapsed time (such as \(A^{\Delta t}\) for an integer-hour Markov model) or explicitly declare the one-transition-per-observation interpretation. **Do not silently change this convention.** Fit emissions/transitions only on training patients, then filter forward on holdouts; never smooth backward with future observations for live-style evaluation.

### HMM ablation

Compare (A) existing features/predictions, (B) same base expert augmented by *filtered* belief features under fresh fitting, (C) HMM-only diagnostics used by the reliability gate. Keep the training effort and evaluation splits explicit; compare patient-mean log loss and tail risk, not entropy alone. A negative result legitimately leaves the HMM as an interpretable diagnostic.

## 6. Information theory: what exactly do we mean by "value of information"?

Under the **true** optimal Bernoulli predictive distributions using context \(C\) and extra permitted observation \(U\), the expected reduction in optimal log score is exactly conditional mutual information:

\[
H(Y\mid C)-H(Y\mid C,U)=I(Y;U\mid C)\ge0.
\]

**Caveat:** Our models are finite, imperfect, and estimated. Their observed held-out log-loss improvement is an *empirical predictive-value estimate*, not necessarily an unbiased estimate of population CMI. Entropy of a model's forecast \(h(p)=-p\log p-(1-p)\log(1-p)\) is a forecast diagnostic, **not** confidence that the forecast is correct. Likewise mutual information does not certify the clinical value of acquiring an extra measurement.

For an optional memory item \(m\), a research utility proxy may be defined as

\[
V(m\mid C_t)=\widehat{\Delta\mathrm{LogScore}}(m\mid C_t)
-\lambda_c\,\mathrm{cost}(m)-\lambda_r\,\mathrm{redundancy}(m,C_t).
\]

Every constituent must be specified and empirically fitted using training-only observations. If the data were observational and measurement missingness is selective, **do not** claim the counterfactual benefit of actively requesting the measurement; that requires stronger causal/selection assumptions or randomized measurement policies. Memory is eligible only if `available_at <= decision_time`, episode access is authorized, and privacy/resource rules pass.

**Practical controlled experiment:** Add filtered belief \(b_t\) or a freshness descriptor \(u_t\) to a *small* gain model. Test whether its out-of-patient-fold log-score improvement over a baseline using \((S,T,\mathrm{ICULOS})\) is positive with patient-cluster uncertainty. Report entropy/CMI estimates as secondary diagnostics, not as the reason for declaring success.

## 7. Reliability supervisor: evidence, abstention and constraints

For a frozen candidate, define a patient-level paired improvement over reference \(q_0\):

\[
\Delta_i=\frac{1}{T_i}\sum_t\big[\ell(Y_{it},p_{0,it})-\ell(Y_{it},p_{1,it})\big].
\]

A candidate can be activated in a predeclared regime \(g\) only after inner-training estimation and disjoint inner-validation evaluation; e.g. use \(\widehat G_g-c\widehat{\mathrm{SE}}_g>\tau_g\) with patient-cluster uncertainty and minimum patient/event support, both fixed before outer evaluation. This is a **heuristic selection safeguard**, not an anytime-valid statistical guarantee. The gate itself is fitted/selected within outer training data.

Ordinary constraints: \(q\in[0,1]\), bounded departure from declared reference, and switching penalty/limit when jointly feasible. Emergency static fallback takes precedence: an invalid secondary model means \(q=0\) immediately, even if it violates an ordinary rate limit; if static is invalid, return a failure and avoid a prediction.

Candidate SRM-style objective (for supervised outer-training development only):

\[
\widehat{\mathcal J}(\varphi)=
\frac1n\sum_i\frac1{T_i}\sum_t\ell(Y_{it},p_{it}(\varphi))
+\lambda_\varphi\|\varphi\|_2^2
+\lambda_s\widehat{\mathbb E}(q_t-q_{t-1})^2
+\lambda_f\widehat{\mathbb E}(q_t-q_0)^2
+\lambda_T\widehat{\mathrm{CVaR}}_{0.9}(L_i).
\]

`L_i` denotes patient-mean log loss. An empirical objective with regularizers is **not** by itself a PAC/SRM generalization theorem. Show actual out-of-patient confidence intervals and tail outcomes instead of claiming certified robustness. Different objectives/aggregators (hour vs patient vs event-positive) must be named, not swapped silently.

## 8. Metrics registry as a typed experiment contract

Extend `evaluation.registry` **additively**, preserving `nll` as Gaussian NLL and `crps` as Gaussian CRPS. Do not have the sepsis endpoint call the old regression `evaluate(...)` indiscriminately.

Proposed task-specific metric names and interpretations:

| Family | Example registry IDs | Unit of aggregation / caveat |
|---|---|---|
| Binary proper scoring | `bernoulli_log_loss`, `brier_score` | clip convention recorded; hourly and patient-mean distinct |
| Discrimination | `auprc`, `auroc` | undefined for single-class cohorts/folds; record undefined rather than invent |
| Calibration | `ece_binned`, `calibration_slope`, `calibration_intercept` | preprocessing and bins fixed by training partition; small positives -> uncertainty |
| Patient robustness | `patient_mean_log_loss`, `patient_cvar90_log_loss`, `patient_p90_log_loss` | aggregate within patient first |
| Paired comparisons | `paired_patient_gain_mean`, `paired_patient_win_rate`, `paired_patient_gain_ci_low`, `paired_patient_gain_ci_high` | paired patient cluster bootstrap; gain = ref minus candidate |
| Policy dynamics | `temporal_weight_mean`, `temporal_weight_std`, `weight_delta_abs_mean`, `weight_delta_sq_mean`, `fallback_fraction`, `static_mode_fraction` | per-episode reset; separate operational failover from statistical abstention |
| Dependence | `residual_corr`, `residual_autocorr_lag1`, `model_disagreement_sq_mean` | diagnostic; autocorrelation depends on label structure |
| Information | `predictive_entropy_mean`, `conditional_gain_surrogate` | diagnostic, not certified reliability; estimated CMI needs separate assumptions |
| Regression-only | existing `rmse`, `mae`, `bias`, Gaussian `nll`, `crps`, `coverage_90`, `width_90` | BIDMC/continuous tasks only, unchanged |

Registry schema should record: metric ID; task; outcome distribution; objective direction; unit/weighting; probability clipping; aggregation level; valid input shape; missing/undefined behavior; policy/model version; data partition; and metric version. The evaluation **runner** should emit long-form structured records per experiment, policy, split, subgroup and metric, with provenance and uncertainty. Keep metrics used for **selection** separate from post-selection **monitoring** diagnostics.

## 9. Preregistered 4.5-hour research execution plan

**P0 — Ensure working demo is not broken**
- Confirm GitHub CI actually runs (local YAML parsing is not enough).
- Activate disposable PostgreSQL integration separately; no cloud provisioning or schema migrations without approval.
- Save model manifest/hash, successful synthetic and real-patient parity smoke-test outcomes, and test count.

**P1 — Metrics extension (Codex's next math-adjacent implementation task)**
- Add Bernoulli metrics and per-patient aggregations behind new task-specific registry interface; preserve old functions; synthetic hand-calculated fixtures and edge-case tests; all non-DB pytest; scoped Ruff; diff check.
- Failure cases: all-positive/all-negative AUPRC/AUROC, exact 0/1 probabilities, NaN, patient with one hour, repeated patients, mismatched split IDs, constant predictions, polarity of paired gains.

**P2 — Simple conditional moment hypothesis (one variable at a time)**
- Freeze preexisting experts and patient-disjoint outer folds.
- Compare static, a patient-mean-log-loss-selected constant blend, unregularized global Brier blend (an analytical identity/control), and one low-complexity conditional moment rule (e.g. predeclared measurement freshness OR expert disagreement regime, not dozens of regimes).
- Fit moment estimates exclusively on inner cross-fitted training predictions; shrink and select thresholds in outer-training data; report outer-fold patient scores, patient bootstrap, CVaR, positive-patient tail, log loss, Brier, AUPRC and gate variability.
- No trusted gain -> do not promote the new policy. A numerically different allocation without outcome gain is not a success.

**P3 — HMM ablation (only after P2 and resources permit)**
- Fit HMM on allowed training patients, filter forward, use states only in a separately identified experiment. Compare the simple representation against a state-augmented model using paired patients and tail metrics. No smoothing or on-the-fly peeking at outcomes.

**P4 — LangGraph research explainer (only once P0–P2 are stable)**
- Make a separate, read-only graph node consume typed, already computed research results and versioned documents. Require citations to model/metric/provenance fields and explicit non-clinical language. Deterministic fallback summary if LLM unavailable; no LLM-generated numerical predictions. Human review before any external action.

**P5 — Database and AWS (in parallel, not blocking math demo)**
- Use local PostgreSQL for experiment metadata, decisions, artifact SHA, and sanitized traces with opt-in integration tests; avoid storing raw patient rows unless there is a clear need and approved privacy boundary.
- GitHub Actions CI -> artifact manifest verification -> immutable Docker image -> ECR -> approved staging -> health/readiness/auth check -> candidate evaluation -> rollback. HTTPS ingress via Nginx or ALB; true blue/green only with adequate dual capacity and backward-compatible DB migration. No AWS provisioning or public exposure without separate authorization.

**Recommended demo story:** Show a real retrospective patient trajectory and synthetic policy comparison; display the static/temporal probabilities, bounded mixture weight, fallback explanation, feature availability, patient-level metrics and research limitations. Explain honestly if the learned adaptive gate did not beat static in nested evaluation.

## 10. Invariants and tests before promoting any policy

1. Prefix causality across all predictions, features, HMM beliefs, memory selections, and graph explanations.
2. Patient and agent-thread isolation; no cross-patient memory or state; DB queries filtered to `available_at <= t`.
3. Numeric agreement of batch vs incremental feature paths; artifact hashes/schemas verified.
4. Stable emergency static fallback even if optional model/HMM/LLM/database components error.
5. Deterministic property tests: identical experts imply identical final probability for every allocation; \(q=0\) equals static, \(q=1\) equals temporal; no invalid probabilities.
6. Independent evaluation: disjoint fit/selection/outer evaluation; policy refitting uses only data allowed by that stage; seed and data hash recorded.
7. On cohort shift, missingness or numerical instability, report coverage/abstention and degrade safely; do not claim invariance from one replay.
8. Every modified production module has targeted tests plus full non-DB pytest and scoped Ruff; opt-in PostgreSQL tests for DB behavior changes; CI must truly execute remotely before being called verified.
9. Agent output must not inject new probabilistic values or clinical advice; provenance, research-only tag, and audit trace always present.
10. No cloud credentials in code, screenshots, or Git. Public endpoints are authenticated and TLS-protected; localhost can be demonstrated without HTTPS.

## 11. Decision ledger / unresolved research assumptions

- **Accepted:** A clean numerical/agentic separation, static failure precedence, constant-mixture comparator, additive task-specific metrics, retrospective non-clinical target semantics, patient-level outer evaluation.
- **Approved for testing, not for claims:** Conditional covariance, HMM belief contribution, reliability gating, conditional log-score value, agentic explanation quality, blue/green infrastructure.
- **Open:** Actual cohort-B freshness and holdout provenance; exact outcome availability for reactive feedback; quality of information estimators with rare positives; whether HMM filtering contributes beyond fixed temporal features; memory acquisition's counterfactual value under missing-not-at-random data; model artifact historical training reproducibility; privacy design for stored trajectories.
- **Engineering unknowns:** Whether actual GitHub Actions job passes; live PostgreSQL integration; deployed HTTPS/auth/ingress; resource budget for two simultaneously healthy model releases.

## 12. Related versioned documents and ownership

- `markdowns/RELIABILITY_GATED_HYBRID_AND_INFORMATION_THEORY_v0.3.md`: detailed exposure, information and SRM derivations.
- `markdowns/COVARIANCE_AWARE_CONDITIONAL_ALLOCATION_v0.2.md`: conditional second-moment estimator, shrinkage and Brier diagnostics.
- `markdowns/SEPSIS_EVALUATION_AND_HYBRID_EXPERIMENT_CONTRACT_v0.1.md` (or file currently located at repository root): experiment scoring definitions and comparisons.
- `markdowns/HYBRID_CONTROLLER_MATHEMATICAL_CONTRACT_v0.1.md` (or root): earlier controller math.
- `markdowns/AWS_API_SECURITY_AND_DEPLOYMENT_ADR_v0.2.md`: access, networking, deployment and secrets.
- `docs/MATHEMATICAL_FRAMEWORK.md`: existing code-facing math reference. Do not overwrite historical documents to force agreement; record conflicts explicitly.
- `AGENTS.md`: persistent Codex change-control and testing instructions; `markdowns/CODEX_RESEARCH_READING_MAP.md`: task-specific reading map.

**Next decision checkpoint:** once the metric registry is implemented, inspect its test cases and compare a preregistered low-complexity conditional policy on patient-disjoint predictions. Do not turn hypothetical estimates into deployed clinical claims.
