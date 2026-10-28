# Covariance-Aware Conditional Allocation — Mathematical and Experimental Specification v0.2

**Project:** `hackathon-starter` / `healthcare_ai_lab`  
**Date:** 2026-10-28  
**Status:** Mathematical derivation + preregistered research design; **not an implemented or validated improvement**.  
**Related documents:** `HYBRID_CONTROLLER_MATHEMATICAL_CONTRACT_v0.1.md`, `SEPSIS_EVALUATION_AND_HYBRID_EXPERIMENT_CONTRACT_v0.1.md`, original `docs/MATHEMATICAL_FRAMEWORK.md` and Notebook 09.  
**Task:** PhysioNet 2019 hourly *shifted challenge-label* binary prediction; **not** a clinically validated sepsis-onset model.

## 0. Executive research decision

**Keep the covariance optimizer as an independent proposal mechanism. Do not claim that a global covariance optimizer is novel adaptive control.** When its matrix consists of *uncentered* error second moments, its unpenalized solution is exactly the constant mixture minimizing the same weighted training Brier objective. The prospect for genuinely adaptive value is from **conditional, time-valid estimation**, conservative statistical shrinkage, and gating/abstention when estimates are unstable. These are hypotheses to test, not guaranteed improvements.

**Priority:** (1) verify the real trained artifact and raw/engineered prediction parity, (2) additive classification/paired metrics, (3) grouped cross-fitted moment experiment, (4) hybrid supervisor only after credible incremental evidence. The mathematical work here can proceed while the separate Codex artifact-provenance task finishes.

## 1. Exact two-expert derivation

Let episode/patient `i` at hour `t` have binary shifted challenge label `Y_it`, and frozen static and temporal model probabilities `S_it` and `T_it` that depend only on information available by that hour. Set

\[
 d_{it}=T_{it}-S_{it},\quad e^S_{it}=Y_{it}-S_{it},\quad e^T_{it}=Y_{it}-T_{it}=e^S_{it}-d_{it},
\]

and `p_it(q)=S_it+q d_it`, `0<=q<=1`. For any precisely specified target distribution and weighting of patient-hours, define

\[
 N=\mathbb E[e^S d],\qquad D=\mathbb E[d^2]\ge 0,\qquad C=\mathbb E[(e^S)^2].
\]

The **Brier risk identity** is exact:

\[
 R_B(q)=\mathbb E[(Y-p(q))^2]=C-2Nq+Dq^2. \tag{1}
\]

For `D>0`, its unconstrained minimizer is `N/D`; the constrained minimizer for `q in [a,b]` is

\[
 q_{B}^*=\operatorname{clip}\left(\frac ND,a,b\right). \tag{2}
\]

For `D=0`, both experts agree almost surely on the target distribution; predictive Brier risk is independent of `q` and the allocation is **not identifiable from predictions**. Select a declared baseline (`q_F=0` for static-only safety, or a previously preselected constant in a non-emergency experiment).

Let `M=E[ee^T]` for `e=(e^S,e^T)`. Then

\[
 m_{SS}-m_{ST}=N,\qquad
 m_{SS}+m_{TT}-2m_{ST}=D=\mathbb E[(T-S)^2]. \tag{3}
\]

**Crucial distinction:** `M` is *uncentered second moment*, whereas `Sigma=Cov(e)` is centered. If estimating only `Sigma`, add `E[e]E[e]^T` to recover `M`. Consequently, the present `controller/optimizer.py` `estimated_risk` has the Brier interpretation when the supplied moments are `M` and the same aggregation is used.

### 1.1 New implication: global covariance = constant Brier blend

Using the **same fixed out-of-fold predictions and same observation weights**, directly optimizing one constant `q` by empirical Brier loss yields equation (2) with sample `N_hat`, `D_hat`. It is therefore algebraically equivalent to the unpenalized global second-moment optimization, not an independent outperforming controller. An additional global *regularized* moment optimizer becomes a **regularized constant-blend baseline**, not a conditional policy.

This gives a necessary ablation: `global_moment_blend` vs `constant_Brier_blend` should match numerically under identical conventions. A discrepancy indicates an implementation, centering, weighting, or penalization mismatch.

### 1.2 Shrinkage toward a simple reference

Add `lambda >= 0` times `(q-q_ref)^2` to equation (1), where `q_ref` is chosen on training data, not on held-out patients:

\[
 q_{\lambda}=\operatorname{clip}\left(\frac{N+\lambda q_{ref}}{D+\lambda},a,b\right). \tag{4}
\]

If `lambda` is large relative to `D`, the result approaches `q_ref`. For the existing optimizer, time-step and resource penalties produce the previously documented extension with separate stability and linear cost terms. **Do not equate** that code's default `reference_weight=0.5` with the sepsis emergency static fallback `q_F=0`.

For reference-weight `q_ref` and proposed `q`, the unregularized Brier improvement is

\[
 R_B(q_{ref})-R_B(q)
 =2N(q-q_{ref})-D(q^2-q_{ref}^2). \tag{5}
\]

This permits analytic checks of estimated gains but is not a substitute for out-of-patient empirical scoring.

## 2. Conditional allocation, the actual adaptive hypothesis

Let `Z_it` contain **only decision-time observable information**, e.g. model disagreement magnitude, measured-vital availability and ages, ICU elapsed time, and separately validated filtered latent-state beliefs. Never use `Y_it`, future onset, patient-end statistics, labels from later hours, or immature residuals in `Z_it`.

Define conditional moments:

\[
 N(z)=\mathbb E[(Y-S)(T-S)\mid Z=z],\quad
 D(z)=\mathbb E[(T-S)^2\mid Z=z]. \tag{6}
\]

Then a conditional squared-error-optimal policy is `q_B(z)=clip(N(z)/D(z),0,1)` where identifiable. **A policy can legitimately equal zero in most states.**

**Important subtlety:** if conditioning includes the exact pair `(S,T)`, the model difference `d` becomes deterministic. For `d!=0`, 

\[
 q_B^*(S,T)=\operatorname{clip}\left(\frac{\mathbb E[Y\mid S,T]-S}{T-S},0,1\right). \tag{7}
\]

Thus learning a fine-grained conditional allocator is closely related to learning a calibrated conditional event probability—a statistically hard problem in a rare-event setting. A more complex gate can overfit even when the algebra is correct. Prefer low-dimensional, predeclared regimes and explicit shrinkage in the first experiment.

## 3. Patient-weighted training estimators

**Independent evaluation unit: patient, not hour.** In a frozen training-only out-of-fold prediction table with `n` patients, `T_i` observed labeled hours, and identical keys across expert predictions, estimate

\[
 \widehat N = \frac1n\sum_{i=1}^n\frac1{T_i}\sum_{t=1}^{T_i}(Y_{it}-S_{it})(T_{it}-S_{it}),\\
 \widehat D = \frac1n\sum_{i=1}^n\frac1{T_i}\sum_{t=1}^{T_i}(T_{it}-S_{it})^2. \tag{8}
\]

This targets **patient-mean Brier**, not hour-pooled Brier; replacing `1/(n T_i)` by `1/sum_i T_i` changes the estimand. Report both, but select one before fitting.

Requirements:

1. Static and temporal models predict each OOF patient from a **different patient group used for fitting**. Each prediction must have a recorded base-model training-fold ID and feature/artifact version.
2. If fitted policy parameters, subgroup boundaries or shrinkage hyperparameters are learned from these OOF predictions, evaluate them on **separate patient-disjoint outer folds** (or an untouched final cohort). Never use the same outer test patients to fit the gate or select its threshold.
3. Use the patient's own prediction/outcome rows only to fit a research policy **after that episode is part of training data**. For online deployment on a new patient, `N(z)` comes from past *training* patients or matured permitted observations, never their future challenge labels.
4. Records must align on `(episode_id, icu_hour)`; duplicate or missing policy-specific keys are errors or explicit coverage shortfalls, not silent joins.
5. Grouped resampling for estimator uncertainty must draw patients/episodes, preserving their entire temporally correlated sequences.
6. Keep the challenge shifted-label semantics explicit; scoring after retrospective outcome join does not make those labels available to the controller at prediction time.

### 3.1 Low-complexity conditional regimes

Start with **one** predeclared partition at a time, for instance:

- Observability regime: vital measurements sufficiently fresh versus stale, using availability/freshness thresholds chosen in inner training; or
- Disagreement regime: a low/high `|T-S|` split with threshold chosen only in inner training; or
- ICU elapsed-time regime using a declared, training-selected cut.

Do **not** use eventual sepsis-positive patient status, positive label at time `t`, or oracle knowledge as a runtime regime. Test multiple partitions as separate predeclared candidates and account for the selection budget.

For regime `g`, let `n_g` be the number of contributing **distinct patients**, `N_g,D_g` the same patient-weighted estimators within that regime, and `N_0,D_0` the global training estimators. A conservative family is

\[
 a_g=\frac{n_g}{n_g+\kappa},\quad
 \widetilde N_g=a_g\widehat N_g+(1-a_g)\widehat N_0,\quad
 \widetilde D_g=a_g\widehat D_g+(1-a_g)\widehat D_0,\tag{9}
\]

\[
 q_g=\operatorname{clip}\left(\frac{\widetilde N_g+\lambda q_{ref}}{\widetilde D_g+\lambda},0,1\right). \tag{10}
\]

`kappa` and `lambda` are preselected/tuned on **inner patient folds**; `n_g/(n_g+kappa)` is a **heuristic shrinkage rule**, not a Bayesian posterior or theoretical confidence guarantee. Distinct-patient support and per-patient effective hours must both be recorded. Shrinking `N` and `D` jointly avoids artificially producing a zero denominator from a tiny subgroup; PSD moment checks remain appropriate where using the optimizer's matrix interface.

**Reliability and abstention:** If model provenance fails, a candidate prediction is missing/nonfinite, the regime has insufficient independent-patient support, the denominator is near-degenerate under a **predeclared scale-aware rule**, or bootstrap instability is excessive, declare the conditional policy unavailable. Revert to the preselected valid reference `q_ref` in a research comparison; use `q_F=0` for explicit runtime static fallback after a secondary inference failure. Do not silently substitute a plausible probability for a failed static model.

### 3.2 Bootstrap stability without leaking the holdout

Within **training only**, resample full patient vectors, recompute `(N_g,D_g,q_g)`, and estimate both the variability of proposed weights and the **paired incremental Brier gain** against `q_ref` on an inner-validation patient set. If a conservative lower confidence bound for gain is not positive, prefer the reference. This confidence-based choice itself requires nested tuning/selection and can be conservative; report its selection frequency and outer-fold behavior.

Simply bootstrapping fitted observations within the same development sample does not prove generalization, because subgroup selection and model fitting also contribute uncertainty.

## 4. Brier-optimal is not log-loss-optimal

For `p(q)=S+qd`, Bernoulli log loss is `ell(y,p)=-y log p-(1-y)log(1-p)`. With valid interior probabilities,

\[
 \frac{\partial\ell}{\partial q} =
 d\,\frac{p-y}{p(1-p)},\quad
 \frac{\partial^2\ell}{\partial q^2}=
 d^2\left(\frac{y}{p^2}+\frac{1-y}{(1-p)^2}\right)\ge0. \tag{11}
\]

Therefore an unconstrained fixed-data log-loss mixture objective is convex in one scalar `q`, but its minimizer generally differs from `N/D` because it weights errors by current probability. We have already observed differences between hour-pooled and patient-mean rankings; never claim Brier-optimal allocations are log-loss-optimal or automatically improve positive-patient tails.

The primary **evaluation** target for the prototype is patient-mean Bernoulli log loss, supplemented by Brier, AUPRC, calibration and patient CVaR90. Compare the moment policy against a **training-selected constant log-loss blend** as well as a constant Brier blend. Model-selection losses and regularizer scales must be defined before opening the final holdout.

## 5. Proposed falsifiable experiment (no full hybrid gate yet)

**H0:** After patient-disjoint testing, low-dimensional conditional moment allocation does not improve patient-mean Bernoulli log loss beyond the strongest preselected constant mixture; apparent gains on exploratory validation arise from noise/overfitting.

**H1:** At least one predeclared, adequately supported and shrunk conditional moment policy yields positive held-out paired patient-mean log-loss gain over that same baseline, with acceptable predeclared tail-risk behavior.

**Fixed policy matrix:**

| ID | Policy | Core purpose |
|---|---|---|
| S | `q=0` static | Operational fallback and scientific reference |
| CL | Training-optimized constant `q` by patient log loss | Strongest simple log-loss baseline |
| CB | Training-optimized constant `q` by patient Brier | Required algebra check for global moments |
| GM | Unregularized global second-moment optimizer | Should equal CB with identical weights/folds |
| GR | Shrunk global moment reference | Regularization / estimation stability, **not** conditional adaptivity |
| CM | One low-dimensional, shrunk conditional-moment policy | Tests actual state-dependent adaptation |
| A | Existing bounded anticipatory policy | Existing baseline; preserve previous negative outer-fold evidence |

Reactive feedback and a multi-expert learned supervisor remain **separate later experiments**, since their feedback and complexity assumptions differ.

**Procedure:**

1. Freeze patient-disjoint outer folds and feature/artifact versions before looking at policy outcomes; optionally preserve an untouched cohort B for independent reporting if it has not been used in decisions.
2. In each outer training partition, generate/capture patient-grouped OOF base-model predictions. Fit base static/temporal models as specified by the chosen training protocol; do not reuse evaluation labels for fitting.
3. In inner patient folds choose any `q_const`, `lambda`, `kappa`, regime threshold and fallback criterion. Set a small fixed search budget and record every attempt.
4. Fit all policy parameters exclusively using outer-training data, refit experts on permitted outer training patients, and emit **frozen** predictions for the outer held-out patients.
5. Evaluate aligned hours and patient sequences. Primary: paired patient-mean Bernoulli log-loss difference vs CL (positive gain means improvement). Secondary: vs S, patient Brier, hour-pooled loss, AUPRC, calibration, positive-label-hour diagnostics, CVaR90, p10 gain, worst-patient harm, q mean/switching and fallback frequency.
6. Patient-cluster paired bootstrap of *held-out predictions*, with explicitly stated conditional uncertainty limits. Do not tune thresholds or select policies on outer test folds after viewing outcomes.
7. Report the numerical identity `GM ≈ CB` as a **unit/integration check**, not a research breakthrough. If it fails, investigate weighting, covariance centering, clipping, and penalty differences.
8. Report failure, ambiguity and null results. No clinical performance or superiority claim without credible independent evidence.

**Acceptance for further hybrid exploration:** Positive credible paired gain against CL on patient-disjoint evaluation **and** no unacceptable worsening in predeclared adverse-tail criteria, with all provenance and coverage checks passing. A result that improves Brier but not log loss is a useful *metric-specific* finding, not a general win. Exact numerical thresholds and risk margins are not yet calibrated; do not invent them.

## 6. Testable mathematical invariants for implementation

- **Prediction invariants:** `p(q=0)=S`, `p(q=1)=T`, `p(q) in [0,1]` for valid probability inputs and `q in [0,1]`.
- **Brier equivalence:** Direct weighted empirical Brier optimization yields same `q` as the unpenalized `N_hat/D_hat` optimizer for identical predictions and aggregation; if `D_hat=0` both risks are equal and declared fallback wins.
- **Moment identity:** `mSS - mST == N_hat` and `mSS + mTT - 2mST == D_hat` (within numeric tolerance), with **uncentered** matrix moments.
- **Reweighting:** Unequal patient lengths demonstrate patient-mean estimators differ, intentionally, from hour-pooled estimates.
- **Shrinkage:** `lambda -> infinity` yields `q_ref` after feasible projection; `kappa -> infinity` yields global moments for all regimes; no data leads to no learned proposal.
- **No leakage:** Shuffle future observations without changing prefix decisions; remove unavailable outcomes without changing currently valid controller inputs; a patient never influences its own fitted OOF expert parameters.
- **Low identification:** If `S=T` throughout, `D=0` and no claim of model preference is made; if predictions disagree only slightly, report estimate sensitivity and uncertain gain.
- **Failure isolation:** A missing secondary model returns a finite valid static prediction, `q=0`, and a reason; missing static model yields explicit prediction unavailable, never a fabricated zero.
- **API/status:** All sepsis risk outputs retain `shifted_challenge_label`, `research_only=true`, and `clinical_use=false`.
- **Scoring:** Bernoulli log loss is a separately named metric; never overwrite the registry's Gaussian `nll` or `crps`.

## 7. Research-result registry template

Every new experiment result should append a versioned note containing:

```
experiment_id:
status: proposed | implemented | evaluated | rejected | inconclusive
hypothesis:
policy_id:
reference_policy:
patient_count_by_fold:
event_patient_count_by_fold:
raw_model_artifact_versions:
feature_builder_version:
OOF_policy_fitting_and_outer_split_definition:
regime_and_tuning_search_budget:
primary_patient_mean_log_loss_and_paired_gain_CI:
secondary_hourly_loss_brier_auprc_calibration:
CVaR90_p10_gain_worst_patient_gain:
controller_weight_and_fallback_statistics:
source_of_feedback_and_availability:
known_limitations:
next_decision:
```

**Documentation convention:** Keep v0.1 and this v0.2 as distinct historical artifacts, label untested equations as proposals, and never replace negative nested-fold results with optimistic exploratory numbers.

## 8. Implementation boundary for Codex

**Do not interrupt the ongoing artifact-integrity and real patient smoke-test task.** Once that checkpoint is reviewed, a future coding assignment can implement a small, additive `moment_estimation` research module plus synthetic tests from sections 1, 3 and 6. There is no approval yet to activate conditional covariance gating in the default API or to add a learned hybrid supervisor. Preserve the existing regularized optimizer and `evaluation.registry` behavior.
