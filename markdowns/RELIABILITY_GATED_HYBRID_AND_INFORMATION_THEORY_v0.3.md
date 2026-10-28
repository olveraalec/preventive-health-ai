# Reliability-Gated Hybrid Controller and Information-State Mathematics — v0.3

**Project:** `hackathon-starter` / `healthcare_ai_lab`  
**Date:** 2026-10-28  
**Status:** Research specification; only the existing static/temporal inference, bounded controller, trajectory API, feature builder, and artifact parity are reported implemented. **The hybrid learned supervisor, conditional estimator, HMM ablation, and reliability certification are not yet validated or activated.**  
**Related documents:** `HYBRID_CONTROLLER_MATHEMATICAL_CONTRACT_v0.1.md`; `SEPSIS_EVALUATION_AND_HYBRID_EXPERIMENT_CONTRACT_v0.1.md`; `COVARIANCE_AWARE_CONDITIONAL_ALLOCATION_v0.2.md`; `docs/MATHEMATICAL_FRAMEWORK.md`.  
**Task:** PhysioNet 2019 Sepsis Challenge **shifted-label** prediction; not clinical onset prediction, decision support, or a clinically certified risk estimate. BIDMC second-scale forecasting remains a separate task.

## 0. Decisions and non-decisions

**Freeze now:**

- One causal, timestamp-aware information contract per patient/episode. No training labels, smoothed HMM states, retrospective outcomes, or late-arriving SQL memories enter live decisions before their availability time.
- An operationally executable static predictor and immediate static-only failover if optional temporal/gain/HMM/controller components fail. If static fails, return an error, not an invented probability.
- Strong **reference policies**, including static (`q_F=0`) and an inner-training-selected constant blend (`q_0`). Distinguish an *ordinary statistical reference* from the *emergency operational fallback*.
- Patient-disjoint model fitting, policy fitting, hyperparameter selection, and outer evaluation; patient-cluster uncertainty; classification-specific scoring names separate from Gaussian regression metrics.
- The old covariance optimizer, anticipatory controller, reactive experiments, and HMM remain independent, testable components.

**Keep experimental:** the reliability gate, state-conditioned moments, learned expert combinations, HMM information contribution, latency tradeoffs, and any SRM generalization claim. No candidate is promoted solely because its blend weight varies over time.

## 1. Information sets and prediction contracts

For patient/episode `i`, ICU hour `t`, define a filtration `F_{i,t}` containing **only** telemetry actually available by decision time, fixed model/artifact parameters, permitted patient metadata, available pre-trained cohort-level estimates, and mature outcome feedback. Observation time and *availability time* must be distinct where relevant.

Let `Y_{it} in {0,1}` denote the shifted benchmark label, evaluated only retrospectively; let the two frozen predictive experts be

\[
S_{it}=\Pr_{\widehat\theta_S}(Y_{it}=1\mid\mathcal F_{i,t}),\qquad
T_{it}=\Pr_{\widehat\theta_T}(Y_{it}=1\mid\mathcal F_{i,t}),
\]

with `0 <= S,T <= 1`, and `d_{it}=T_{it}-S_{it}`. Allocation produces

\[
 p_{it}(q)=S_{it}+q_{it}d_{it},\qquad 0\le q_{it}\le1.
\]

**Schema rule:** episode, ICU hour, data-source identity, feature-builder version/hash, fitted feature order, model binary SHA-256, and timestamp availability must be checked; matching metadata is a necessary consistency check, not independent proof of historical training provenance.

## 2. Incremental-value formulation: a bridge from covariance to reliability

### 2.1 Candidate allocation relative to a preselected reference

Let `q_0` be a **constant**, selected using training patients only (e.g. minimize patient-average log loss, with any operational exposure limits accounted for). Define

\[
 p_0=S+q_0d,
 \qquad q_1(z)\in[0,1], \qquad
 \delta(z)=[q_1(z)-q_0]d.
\]

The candidate may be covariance-aware, anticipatory, or a small pre-specified hybrid. Rather than adopt `q_1` immediately, use an exposure `a(z) in [0,1]`:

\[
 \boxed{p_a=p_0+a\delta,\quad q_a=q_0+a(q_1-q_0).} \tag{1}
\]

This *interpolates allocation*, not a randomized discrete memory-selection policy. If `q_1=q_0`, exposure is unidentifiable but irrelevant to prediction.

### 2.2 Exact Brier incremental gain

For conditional observable context `Z=z`, define

\[
N_z=\mathbb E[(Y-p_0)\delta\mid Z=z],\quad
D_z=\mathbb E[\delta^2\mid Z=z]\ge0.
\]

Direct expansion gives the **exact** conditional Brier improvement over the reference:

\[
\boxed{G_B(a\mid z)=\mathbb E[(Y-p_0)^2-(Y-p_a)^2\mid z]
=2aN_z-a^2D_z.} \tag{2}
\]

When `D_z>0`, the optimum over `[0,1]` is `a^*_z=clip(N_z/D_z,0,1)`. When `D_z≈0`, the policies are prediction-equivalent or weakly identifiable; use the declared reference, **not** an unstable ratio.

**Crucial insight:** covariance/moment optimization determines a *candidate*, but deciding **whether to expose** that candidate can itself be expressed as a lower-complexity conditional incremental-gain problem.

The direct two-model Brier blend is a special case with `q_0=0`, `q_1=1`, and `delta=T-S`. The global unpenalized second-moment optimizer equals the constant-Brier blend under identical patient/hour weights and prediction data; do not claim novelty from that identity.

### 2.3 Log-loss derivative: why Brier gains do not imply log-loss gains

For Bernoulli `ell(y,p)=-y log p-(1-y)log(1-p)` and interior `p_a`, differentiation yields

\[
 \frac{\partial\ell(y,p_a)}{\partial a}
 =\delta\frac{p_a-y}{p_a(1-p_a)}, \tag{3}
\]
\[
 \frac{\partial^2\ell(y,p_a)}{\partial a^2}
 =\delta^2\left[\frac{y}{p_a^2}+\frac{1-y}{(1-p_a)^2}\right]\ge0. \tag{4}
\]

The **conditional first-order benefit** of a small exposure is

\[
 B_{\log}(z)=-\mathbb E\left[
 \delta\frac{p_0-Y}{p_0(1-p_0)}\middle|Z=z\right]. \tag{5}
\]

A positive `B_log` suggests a small move may help, but not that full exposure is optimal; curvature, sample uncertainty, and rare-positive tails matter. Values are numerically scored with declared clipping `epsilon` and exact objective-specific conventions. In an initial experiment, *fit moment candidates and choose their hyperparameters using patient-mean Bernoulli log loss on independent inner folds*; report Brier as a mechanistic secondary metric.

### 2.4 Candidate-gain estimators with patient weighting

For a fixed, frozen proposed candidate and predeclared low-dimensional regime `g`, use **training-only cross-fitted** patient sequences:

\[
\widehat N_g=\frac{1}{n_g}\sum_{i\in g}\frac{1}{m_{i,g}}
\sum_{t\in g(i)}(Y_{it}-p_{0,it})\delta_{it}, \quad
\widehat D_g=\frac{1}{n_g}\sum_{i\in g}\frac{1}{m_{i,g}}
\sum_{t\in g(i)}\delta_{it}^2. \tag{6}
\]

Here `n_g` counts distinct patients contributing to that regime; `m_{i,g}` counts that patient's applicable hours. This targets **equal patient weight within a regime**. If instead one wants the global patient-mean objective restricted to a regime, weight each hour by `1/T_i` and normalize according to the declared target; the two estimands differ. Regimes cannot be defined using eventual sepsis outcome or retrospective label status for runtime allocation.

Shrink toward global training moments:

\[
\gamma_g=\frac{n_g}{n_g+\kappa},\quad
\widetilde N_g=\gamma_g\widehat N_g+(1-\gamma_g)\widehat N_0,\quad
\widetilde D_g=\gamma_g\widehat D_g+(1-\gamma_g)\widehat D_0,
\]
\[
\boxed{a_g=\operatorname{clip}\left(
\frac{\widetilde N_g+\lambda a_{\rm ref}}{\widetilde D_g+\lambda},0,1\right).} \tag{7}
\]

For experimental exposure over `q_0`, use `a_ref=0`; do **not** confuse it with the static emergency weight `q_F=0` (the two reference *parameters* act in different spaces). `kappa`, `lambda`, regime boundaries, minimum patient support, and denominator tolerance are **chosen only on inner training folds**. Bootstrap full patients, not individual hours. The shrinkage formula is a heuristic, not a posterior confidence guarantee.

## 3. Reliability as evidence, not entropy

Let `g` be one predeclared regime (e.g. measurement freshness or expert disagreement); let a policy `a_g` be selected exclusively from a training partition. For each *independent* inner-validation patient, compute the paired gain in patient-mean log loss relative to `q_0`, yielding `Delta_i` for that frozen policy. One cautious decision statistic is

\[
 \operatorname{LCB}_g=\widehat{\mathbb E}[\Delta_i\mid g]
 -c_g\widehat{\operatorname{SE}}_{\rm patient}(\Delta\mid g), \tag{8}
\]

or a patient-cluster-bootstrap lower quantile with the procedure fixed in advance. Activating adaptation only where `LCB_g > tau_g` (complexity/resource threshold) creates a *selection rule* that itself must be evaluated on fresh patient-disjoint outer folds. **This estimate is neither a calibrated probability of improvement nor an anytime-valid confidence sequence.** If expert/gate fits and LCB estimation use the same observations, optimistic selection is likely; require separate inner folds or full nesting.

Policy outcomes for rare positive patients may be highly variable. Record patient support, fraction of positive episodes in a regime *retrospectively*, LCB stability across outer training splits, proportion of episodes using the reference, and tail loss. Refuse conditional adaptation if support, fit provenance, or numerical conditioning is inadequate.

**Operational precedence:**
1. Invalid static prediction -> no prediction, clear failure.
2. Missing/invalid secondary expert or incompatible artifact -> immediate valid static prediction (`q_F=0`), ignoring normal switching bounds.
3. Valid experts but unreliable adaptive *gain* -> use the predeclared feasible statistical reference `q_0` (often static; it need not be 0 in comparative research).
4. Strong evidence -> allow bounded adaptive exposure; project onto jointly feasible allocation constraints.

If `q_0` lies outside operational fallback-deviation bounds, the implementation must define a **feasible reference** `q_0^{feas}` in advance and evaluate that policy, rather than quietly projecting it after comparing to an unconstrained baseline.

## 4. Hierarchical hybrid, if simpler conditional policies earn it

Maintain **candidate proposals**, not incompatible direct outputs:

- `F`: static-only `q_F=0` (always eligible if static predictor valid).
- `A`: anticipatory candidate from cross-fitted estimated conditional benefit.
- `C`: covariance-aware candidate from uncentered **error second moments**, not centered covariance alone.
- `R`: reactive candidate **only if feedback has a verified availability timestamp**; for current sepsis replay, keep reactive policy simulation-only without an observational protocol.

A small frozen supervisor can assign `pi_j(z)>=0`, `sum_j pi_j=1` across available experts, yielding `q_mix=sum_j pi_j q_j`. Let `q_base` be the declared operational reference. Apply a separate reliability exposure `rho in [0,1]`:

\[
 q_{proposal}=q_{base}+\rho(z)(q_{mix}-q_{base}). \tag{9}
\]

Do **not** call `pi_j` calibrated expert correctness or `rho` calibrated trust without independent tests. Because candidate proposals may be nearly equal, `pi` can be non-identifiable. Compare final predictions, paired gain, expert ablations and policy stability, not raw `pi` interpretation.

For normal operation, form **one feasible intersection**

\[
\mathcal Q_{it}=[0,1]\cap[q_F-d_{max},q_F+d_{max}]\cap
[q_{i,t-1}-s_{max},q_{i,t-1}+s_{max}], \tag{10}
\]

and project `q_proposal` onto it. Log infeasible intersections; priority: validated fallback constraints above normal step smoothing. For model failure or hard reliability-unavailability requiring emergency failover, force `q=0` immediately even when `|q-q_prev|>s_max`. Current runtime tests treat switching as **per observation**; elapsed-hour-aware switching is a separate, testable design change, not implicit behavior.

This is an *experimental hierarchical architecture*, not a mandatory always-changing gate.

## 5. Causal latent state and HMM mathematics

Let `Z_{it} in {1,...,K}` represent a latent statistical regime (not necessarily a biological or clinically meaningful state), `X_{it}` the observed physiology at hour `t`, and `b_{it}(k)=Pr(Z_{it}=k | X_{i,<=t})` the **filtered** belief.

A conventional discrete-time HMM with transition `A`, where `A_{jk}=Pr(Z_t=k|Z_{t-1}=j)`, admits

\[
 b^-_{it}(k)=\sum_j b_{i,t-\Delta}(j)(A^{\Delta})_{jk},\qquad
 b_{it}(k)=\frac{L_k(X_{it}^{obs})b^-_{it}(k)}{
\sum_h L_h(X_{it}^{obs})b^-_{it}(h)}. \tag{11}
\]

For integer ICU-hour gaps `Delta`, `A^Delta` is one *candidate* gap policy. The existing implementation's transition and initial-observation convention must be preserved or explicitly migrated and retested; **do not assume it already uses `A^Delta`**. An empty observed feature subset should contribute a neutral likelihood; for a diagonal-Gaussian emission, `L_k` is the product over *observed* dimensions, computed in log space for stability. Fit HMM parameters and feature transforms on training patients only.

Important safeguards: reset `b` per episode; only filtering `X_{<=t}` in runtime; no backward smoothing; do not condition on future challenge labels; record unit/feature definitions; test gaps, all-missing observations, first observation, and prefix invariance.

**Ablations:** `baseline static/temporal` -> `+HMM filtered belief as diagnostic` -> `+HMM belief as a small supervised gate feature`. If HMM beliefs add no out-of-patient predictive value, retain them as optional uncertainty diagnostics, not as an automatic controller driver.

## 6. Information theory: measurable questions rather than decorative entropy

The Bernoulli entropy of a valid probability `p` is

\[
H_B(p)=-p\log p-(1-p)\log(1-p). \tag{12}
\]

This describes predictive dispersion, **not** observed error, epistemic confidence, or model correctness. Two confident, wrong experts can have low entropy.

Proposed quantity of interest: does an HMM belief or freshness feature `B` improve predictive information about the shifted label after accounting for *declared baseline summary* `U`?

\[
 I(Y;B\mid U)=H(Y\mid U)-H(Y\mid U,B). \tag{13}
\]

If `B` is a deterministic function of *the complete history* and `U` already includes that complete history, then **the true conditional mutual information is zero**. For a meaningful ablation, condition on a limited baseline (e.g. `U=(S,T,availability summary)`), and test whether the *additional summary* `B` contributes on held-out patients.

Estimate using **cross-fitted held-out predictive log scores** for two properly specified conditional models (with and without `B`):

\[
\widehat\Delta_{CE}=\frac1N\sum_i\frac1{T_i}\sum_t
\left[\ell(Y_{it},\widehat p_U)-\ell(Y_{it},\widehat p_{U,B})\right]. \tag{14}
\]

This is **empirical predictive log-score improvement**, *not automatically a consistent estimator of conditional mutual information*. Equality to CMI requires idealized Bayes-optimal conditionals under compatible weighting. The same distinction applies to models based on cross-entropy and conditional entropy. Preserve joint entropy and pairwise/interactions as diagnostics with explicit estimator bias and sample-size caveats; do not optimize every information metric simultaneously.

Residual serial dependence `Corr(e_{t},e_{t-k})` must be estimated **within patients**, not across concatenated episodes; strongly serial challenge labels are a possible contributor and do not prove a physiological latent state. Centered residual covariance `Sigma` and uncentered moments `M=Sigma+mu mu^T` must be separately logged.

## 7. SRM, tails, and non-vacuous reporting

For patient `i` with `T_i` labeled hours, fix `epsilon` and define a numerically clipped Bernoulli loss

\[
 L_i(h)=\frac1{T_i}\sum_t \ell_\epsilon(Y_{it},p^h_{it}). \tag{15}
\]

Candidate training objective:

\[
\begin{split}
 J(h)=\;&\widehat R_{patient}(h)
 +\lambda_{par}\Omega(h)
 +\lambda_{switch}\frac1N\sum_i\frac1{\max(T_i-1,1)}\sum_{t=2}^{T_i}(q_{it}-q_{i,t-1})^2\\
 &+\lambda_{fallback}\mathbb E[(q-q_{base})^2]
 +\lambda_{tail}\operatorname{CVaR}_{0.90}(L_1,\ldots,L_N).
\end{split} \tag{16}
\]

For `T_i=1`, switching contribution is 0. Define upper-tail empirical CVaR by

\[
 \operatorname{CVaR}_\alpha(L)=\min_\eta\left[\eta+
 \frac1{(1-\alpha)N}\sum_i\max(L_i-\eta,0)\right]. \tag{17}
\]

These penalties are **regularization choices**, not generalization certificates. If optimizers use patient-mean loss but report pooled hourly loss, document their divergent target distributions. Never use retrospective positive-patient status to define a live gating regime.

A valid but typically loose finite-policy SRM illustration: if patients are independent, `L_i(h) in [0,B]` with `B=-log(epsilon)`, and a **fixed finite** candidate class `H` has `K` elements (with candidate definitions frozen independently of the evaluation sample), then with probability at least `1-delta`,

\[
 R(h)\le\widehat R(h)+B\sqrt{\frac{\log(K/\delta)}{2N}}
 \quad \forall h\in H. \tag{18}
\]

For illustration, `N=1842`, `K=5`, `delta=.05`, and `epsilon=1e-6` imply a complexity term of approximately **0.488**, much larger than previously observed patient-mean loss differences. This is **vacuous for the hoped-for small gains**. The independence assumption and predeclared finite class matter; the bound does not cover unlimited adaptive hyperparameter searching, correlated patients, adaptive deployment, or distribution shift. Do not claim numerical guarantees from the empirical penalty in (16).

### Decision rule for next experiments

Do not select the final hybrid using an outer holdout. Predeclare an inner-fold candidate budget and choose reference/thresholds there. For research promotion require positive *paired out-of-patient* gain against the training-selected constant blend, stable calibration, and no unacceptable worsening of predeclared patient-tail criteria. A null or negative outcome is a valid finding and should retain the simpler policy.

## 8. Standardized, experiment-aware metrics (additive to existing registry)

Do not rename or alter the existing Gaussian `nll`, `crps`, `rmse`, `mae`, `bias`, coverage, or width metrics for continuous forecasting. Add explicit Bernoulli score functions and time/patient aggregation semantics:

| Family | Required statistics / metadata |
|---|---|
| Bernoulli classification | `bernoulli_log_loss`, `brier_score`, AUPRC, AUROC when defined, prevalence, calibration bins/ECE diagnostics |
| Aggregation | Hour-pooled and per-patient mean scores reported separately; prediction coverage & unique episode count |
| Paired comparison | `delta_i=L_i(reference)-L_i(candidate)`, mean/median, patient win rate, patient-cluster bootstrap CI, p10 delta, worst delta |
| Adverse tails | patient-loss CVaR90, positive-patient and positive-label-hour results **retrospectively only** |
| Controller | `q_mean`, `q_std`, mean square switching, total variation, near-static fraction, emergency/static fallback counts and reasons, infeasible-projection overrides, optional expert availability |
| Uncertainty | predictive entropy, gain-estimator calibration/error, condition/sample-support diagnostics, centered covariance vs uncentered moments, within-patient residual lag dependence |
| Optional HMM | filtered-belief entropy, state-switch counts, missing-emission fraction, incremental held-out predictive log score |
| Continuous forecasting | inherited regression metrics; Gaussian NLL/CRPS/coverage only when actual predictive sigma is defined |

**Registry conventions:** metric name, task type, units, `direction` (min/max/target), aggregation key, required inputs, availability/undefined policy, registry version, scorer version, preprocessing/model hash, evaluation split/seed, and whether a metric is an **optimization target** or a **monitoring diagnostic**. AUPRC/AUROC undefined for a one-class slice; do not silently assign 0. Uncertainty via *patient-cluster* resampling, with interval construction and multiple-comparison caveat logged.

## 9. Minimum controlled research sequence

1. **Freeze artifact and data/provenance.** Verify binary hash and feature parity; protect a truly unseen patient cohort if available; use matched `(episode_id, icu_hour)` predictions and frozen outcome joins.
2. **Metrics implementation.** Add classification, paired-patient, tail, and allocation diagnostics with analytic synthetic tests; preserve the existing Gaussian registry.
3. **Algebra unit checks.** Verify global unpenalized second-moment optimizer equals constant Brier blend under identical weighting; verify `M=Sigma+mu mu^T`, `D=E[(T-S)^2]`; test zero disagreement and ill-conditioned denominator.
4. **Small finite comparison:** static, inner-trained constant log-loss blend, inner-trained constant Brier blend, global shrunken moment, *one* predeclared conditional moment regime, and the existing anticipatory baseline.
5. **Reliability gate ablation:** fixed candidate vs conservative inner-trained evidence threshold; log loss, Brier, confidence, patient CVaR90, worst-patient harm, fallback proportion and variation.
6. **Optional information-state ablation:** diagnostics-only HMM then filtered HMM belief in small gate; no assumption of state clinical meaning or usefulness.
7. **Only if positive evidence:** compare a tiny hierarchical hybrid against the **best preselected simple policy**, not just static. Reactive branch requires a separate defensible feedback-timing study.

**Research null:** The most sophisticated hybrid may lose to a static or constant mixture. The architecture must report that result and keep the simpler policy available.

## 10. Software acceptance tests derived from the mathematics

- All returned probabilities finite in `[0,1]`; test invalid model predictions and legitimate zeros/ones with declared scoring clipping.
- Patient prefix invariance; no cross-patient state or temporal imputations; no unobserved label or smoothed HMM state as a runtime feature.
- Correct sample vs population standard deviation in six-record features, three-record deltas, and irregular ICU-hour age logic according to notebook parity tests.
- Numerical identity of second-moment and direct Brier solutions under the same row/patient weights; test degenerate `D=0` and tiny/nonfinite moments.
- Reliability threshold cannot use outer-test outcomes; no model/provenance-dependent activation without compatible fitted features and training-only estimators.
- Simultaneously feasible projection under normal operation; deterministic explicit static emergency fallback overrides rate limits; stateless request isolation.
- Every new metric has a known expected value fixture, invalid-input tests, undefined-policy behavior, and aggregation test; tests never invoke raw patient data or untrusted model files.
- Repeatable comparison with frozen random seed, fold IDs, patient keys, hyperparameter grid, scorer version, and paired-bootstrap sampling unit.

## 11. Evidence and status ledger (as of 2026-10-28)

| Claim | Status |
|---|---|
| Causal Sepsis feature builder synthetic notebook parity | Codex reported passing |
| Real 37-hour validation-patient feature and inference parity | Codex reported passing; no independent re-run by this document |
| 136 tests passed, 1 DB skipped | Codex reported; not independently rerun in this document |
| Original outer-fold anticipatory advantage | **Not demonstrated** on patient-mean log loss; static 0.153758, anticipatory 0.153855 (reported from prior notebook results) |
| Hierarchical hybrid superiority | Not established; design only |
| Conditional gain or HMM state reliability | Not established; experiment planned |
| Statistically meaningful bounded-SRM guarantee | Not established; crude bound above is loose |
| Clinical or onset-time validity | Not established; shifted-label benchmark research only |

## 12. Versioning and design handoff

- Keep this v0.3 as an *append-only research iteration* in `markdowns/`; do not delete v0.1/v0.2.
- Add a short pointer and decisions to `00_RESEARCH_INDEX.md`; do not rewrite historical outcomes.
- Next Codex math-related task should implement **metrics only**, with exact test cases and no implicit controller changes. Next research experiment should implement one *training-only* conditional second-moment estimator under patient-disjoint folds; do not implement all proposed branches at once.
- Deployment is an independent operational layer. Blue/green or rolling release cannot substitute for empirical predictor validation, artifact provenance, database migration discipline, or patient-data controls.
