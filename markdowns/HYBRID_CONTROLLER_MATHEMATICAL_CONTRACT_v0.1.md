# Hybrid Controller Mathematical Contract — Draft v0.1

**Project:** `healthcare-ai-lab` / `hackathon-starter`  
**Date:** 2026-10-28  
**Status:** Design specification and research hypotheses, **not** a validated clinical system.  
**Scope:** Sepsis Challenge 2019 hourly classification, with an explicitly separate pathway for BIDMC forecasting and agent-memory utilities.  
**Code basis:** Original uploaded source snapshot (`controller/optimizer.py`, `controllers/adaptive.py`, `evaluation/registry.py`, `docs/MATHEMATICAL_FRAMEWORK.md`) and Codex's reported trajectory-service contract. New Codex changes were not independently reviewed in this draft.

## 1. Design principle and prediction target

The control objective is to use **the simplest trustworthy policy for the current context**, including remaining static indefinitely when adaptation has no demonstrable advantage. Decisions may be computed continuously without allocation weights continuously changing.

For patient `i` at ICU hour `t`, let `\mathcal F_{it}` contain observations and metadata *available by decision time t*, as well as any previously matured feedback. It must not include future observations, future challenge labels, or outcomes that have not yet become observable.

- `p^S_{it} = P(Y^{challenge}_{it}=1 | \mathcal F_{it}; S)` is the *static-feature* predictor.
- `p^T_{it} = P(Y^{challenge}_{it}=1 | \mathcal F_{it}; T)` is the temporal predictor.
- `Y^{challenge}` is the **shifted PhysioNet Challenge label**, not an independently validated probability of clinical sepsis onset or a recommendation for care.
- The deployed mixture is `p_{it}(q) = (1-q_{it})p^S_{it} + q_{it}p^T_{it}` for `q_{it}\in[0,1]`.
- The default Sepsis fallback is `q_F=0`, yielding `p_{it}=p^S_{it}`. The original covariance controller's `reference_weight=0.5` is a *different* notion of fallback/reference; do not silently equate them.

The **covariance optimizer**, **anticipatory gate**, **reactive gate**, and **fallback** are candidate allocation mechanisms. They do not automatically correspond to distinct models; several candidates may propose different `q` for the same pair `(p^S,p^T)`. Discrete memory-selection policies are separate and cannot be interpreted as a probability blend without defining a common utility and action space.

## 2. Moment-based risk and the existing optimizer

For a binary outcome `Y` and predictions `p^S`, `p^T`, define errors `e^S=Y-p^S` and `e^T=Y-p^T`. Let the **uncentered** second-moment matrix be

\[
M_t = \mathbb E\left[
\begin{pmatrix}e^S\\e^T\end{pmatrix}
\begin{pmatrix}e^S & e^T\end{pmatrix}
\middle|\mathcal F_{it}\right]
=\begin{pmatrix}m_{SS}&m_{ST}\\m_{ST}&m_{TT}\end{pmatrix}.
\]

Then the mixture's expected **Brier/squared-error** risk is

\[
R_B(q)=(1-q)^2m_{SS}+2q(1-q)m_{ST}+q^2m_{TT}.
\]

By contrast, the **centered covariance** is `\Sigma = M-\mu_e\mu_e^\top`, where `\mu_e=\mathbb E[e|\mathcal F]`; using `\Sigma` alone omits conditional bias. The existing `optimize_weight(moments, ...)` computes exactly this quadratic *algebraically* but names its inputs covariance-related; before deployment its estimator must explicitly declare whether its inputs are centered covariance or uncentered second moments. The existing `project_covariance` enforces PSD but neither establishes calibrated estimates nor corrects bias.

With no penalties and `D=m_{SS}+m_{TT}-2m_{ST}>0`, the unconstrained squared-error-optimal allocation is

\[
q_C^{raw}=\frac{m_{SS}-m_{ST}}{D}.
\]

Constrain it to a feasible interval and fallback conservatively when `D` is negligible or moment estimates are unreliable. **This optimizes Brier risk under assumed moments, not Bernoulli log loss.** A low estimated Brier objective does not guarantee the best log loss or patient-tail performance. Estimate moments from patient-disjoint *training-only cross-fitted* predictions; regularize/shrink with held-out training folds; never use the current patient's not-yet-available label.

For the existing quadratic code, with `\lambda_c` complexity penalty to reference `q_ref`, `\lambda_s` switching penalty to previous `q_prev`, and `\lambda_m c` resource cost per unit weight, the stationary point is

\[
q_C^{raw}=\frac{m_{SS}-m_{ST}+\lambda_c q_{ref}+\lambda_s q_{prev}-\frac12\lambda_m c}{D+\lambda_c+\lambda_s}.
\]

This reproduces the existing `controller/optimizer.py` objective when those entries are supplied as its `moments`. Its degenerate-denominator fallback to `reference_weight` does **not** equal a verified emergency static fallback; policy-level orchestration must distinguish them.

## 3. Candidate allocation policies

Each policy produces a proposal `q^{(j)}_{it}\in[0,1]` and an availability flag, on a common observable-information contract.

- **Fallback F:** `q_F=0` for the Sepsis mixture; always eligible when the static predictor is valid.
- **Anticipatory A:** a bounded proposal derived from predicted *conditional* incremental held-out utility, e.g. `\widehat\Delta_{it}`. The gain estimator is speculative until patient-disjoint validation shows it forecasts gains; a sigmoid transformation is **not** a calibrated probability of benefit.
- **Reactive R:** proposal from matured, timestamped outcome feedback only. No assumption that both policies' counterfactual outcomes are observable online. If there is no mature feedback, mark unavailable rather than fill it with simulated future residuals.
- **Covariance-aware C:** proposal minimizing a well-defined moment-based risk and penalties; eligible only if moments are estimated from valid historical training information and quality checks pass.

**Optional HMM:** Filtered belief `b_{it}(k)=P(Z_{it}=k|X_{i,\le t})` may be a supervisor input after training-only parameter fitting and a separate causal-filtering ablation. It is not a validated reliability score or prediction improvement by itself. First-emission convention, irregular hour gaps, and emission assumptions must be explicit.

## 4. Candidate hierarchical supervisor (not yet implemented)

Let `z_{it}\in\mathcal F_{it}` be decision-time features: missingness and measurement ages, model probabilities/entropy, source provenance, validated feedback availability, and quality diagnostics. The optional supervisor proposes

\[
\pi_{it}=\operatorname{softmax}(g_\phi(z_{it})+m_{it}),\quad
\widetilde q_{it}=\sum_{j\in\{F,A,R,C\}}\pi^{(j)}_{it}q^{(j)}_{it},
\]

where masks `m_j=-\infty` mark unavailable experts and valid weights sum to one. F remains available when static inference is healthy. The softmax weights are **allocation coefficients**, not calibrated probabilities of expert correctness.

Apply a separate, possibly calibrated **trust/exposure parameter** `\rho_{it}\in[0,1]`:

\[
q^{proposal}_{it}=q_F+\rho_{it}(\widetilde q_{it}-q_F).
\]

`\rho` must not be assumed calibrated merely because it is based on entropy. In a conservative first experiment, it may be a fixed or training-selected coefficient; later experiments may train and calibrate a reliability estimator on held-out data. An untrusted signal can produce `\rho=0` and static allocation.

**Feasibility:** On normal healthy operation, define

\[
\mathcal Q_{it}= [0,1]\cap[q_F-d,q_F+d]\cap[q_{i,t-1}-s,q_{i,t-1}+s]
\]

with deviation limit `d` and step limit `s`. If nonempty, `q_{it}` is the Euclidean projection of `q^{proposal}` onto `\mathcal Q_{it}`. If empty, **prioritize the static/fallback band over the step limit**, log the override, and return a safe point in the fallback band. Do not run independent sequential clipping that can violate an earlier constraint. If `\rho=0`, or a required secondary model/estimate fails, override normal rate limits and select `q_F=0` immediately, provided static prediction is valid. If the static predictor itself fails, report *prediction unavailable*; do not invent a valid risk probability.

**Alternative to gating:** Retain all policies as experimental alternatives; an empirically learned hybrid should not replace static until holdout evaluation warrants it.

## 5. Candidate regularized learning objective

For independent evaluation units *patients*, let `T_i` be the number of observed hours and `\ell_\epsilon(y,p)` the Bernoulli log loss with probabilities clipped to `[\epsilon,1-\epsilon]` **for numerical scoring**, with `\epsilon` fixed before evaluation. Define

\[
L_i(\phi)=\frac1{T_i}\sum_{t=1}^{T_i}\ell_\epsilon(Y_{it},p_{it}(q_\phi)).
\]

A research candidate objective on **training/inner-validation patients only** is

\[
\begin{aligned}
\widehat J(\phi)=&\frac1N\sum_iL_i(\phi)
+\lambda_\phi\|\phi\|_2^2\\
&+\lambda_s\frac1N\sum_i\frac1{T_i-1}\sum_{t>1}(q_{it}-q_{i,t-1})^2\\
&+\lambda_F\frac1N\sum_i\frac1{T_i}\sum_t(q_{it}-q_F)^2
+\lambda_T\operatorname{CVaR}_{0.90}(L_1,\ldots,L_N).
\end{aligned}
\]

For `T_i=1` define the switching sum as zero. `CVaR_\alpha` here is empirical upper-tail **patient loss**:

\[
\operatorname{CVaR}_\alpha(L)=\min_\eta\left[\eta+\frac1{(1-\alpha)N}\sum_{i=1}^N\max(L_i-\eta,0)\right].
\]

These are **candidate regularizers**. The objective alone is *not* a proven generalization bound or guarantee of patient safety. Scale and ablate terms; preselect hyperparameters using grouped inner folds and compare on independent grouped outer folds. Selection bias requires keeping the final holdout untouched through development.

For theoretical SRM, one possible *illustration* is a finite-policy-class patient-level concentration inequality if patients are independent, each patient's clipped loss lies in `[0,B]`, and policy class size `|\mathcal H|` is finite:

\[
R(h)\le\widehat R(h)+B\sqrt{\frac{\log|\mathcal H|+\log(1/\delta)}{2N}}
\quad \text{for all }h\in\mathcal H
\]

with probability at least `1-\delta` under those assumptions. With clipped Bernoulli log loss, `B=-\log\epsilon`. This bound can be loose and does **not** automatically apply to continuous unrestricted policy classes, adaptively chosen model classes, dependent patients or distribution shift. Do not label the empirical penalty expression a *proven SRM bound*.

## 6. Experiment and evaluation contract

Keep the existing regression registry **unchanged**, particularly `nll` (Gaussian NLL), `crps`, `rmse`, `mae`, `bias` and interval metrics. Add a task-specific **Bernoulli** registry or namespaced metric set; avoid reusing `nll` to mean binary log loss.

**Classification metrics** (with explicit aggregation): `bernoulli_log_loss`, `brier_score`, `auprc`, `auroc` (if both classes), calibration plot/ECE with declared binning, mean predicted risk, and class prevalence. Compute hour-weighted and patient-weighted variants distinctly; for rare-event classification, compare against constant-prevalence and static predictor baselines.

**Patient-level diagnostics:** `patient_mean_log_loss`, paired patient difference `L_i(baseline)-L_i(candidate)`, patient win fraction, bootstrap CI by resampling patients, bottom decile improvement, `patient_cvar90`, sepsis-positive patient versus negative patient strata, positive-label-hour log loss, and pre-specified calibration subgroups. Report positive/negative strata *retrospectively*; do not use patient eventual outcome for online control.

**Controller diagnostics:** `q_mean`, `q_std`, `q_total_variation`, `q_mean_squared_step`, near-static proportion, emergency fallback count, expert availability and exposure, constraint override count, probability of non-finite input/failure, conditional realized gain and estimation error. Changes in `q` do not imply improved quality.

**Information/reliability diagnostics:** Bernoulli predictive entropy `H(p)=-p\log p-(1-p)\log(1-p)`, residual lag correlation (within patient), centered covariance and second-moment diagnostics, mutual/conditional information with estimator and bias declared, calibration of any predicted gain `\widehat\Delta`, error-versus-availability analysis, distribution shift and patient-tail concentration. Entropy measures uncertainty under the forecast, not prediction accuracy or causal information gain.

**Primary design comparison:** Static, training-selected best *constant* mixture, anticipatory fallback, covariance fallback, and regularized hybrid. Reactive is an additional candidate only in a specified retrospective/oracle-delay simulation until real feedback availability is defined. Compare each to the strongest preselected simple baseline; report uncertainty and tail harm rather than relying on mean-only improvements.

**Splitting:** Split/group by patient, never by individual hours; use nested cross-fitting for supervised gain and moment estimates, tune only inside training folds, preserve genuinely unseen final cohort(s), and bootstrap entire patients, not rows. Report both patient mean and pooled hourly losses, including when they disagree. The already inspected outer-fold analysis did **not** establish a robust advantage for anticipation over static; preserve this negative evidence.

**No data leakage:** Inference at `(i,t)` may use only `\mathcal F_{it}`. Do not use labels, residuals, HMM smoothed state estimates, future-fill features, patient-end summaries or SQL memories created after the decision timestamp. Outcomes are joined for evaluation only after predictions are frozen.

## 7. Implementation boundaries, checkpoints and interfaces

- **Codex current task:** exact causal Sepsis raw-to-engineered temporal features, version/provenance checks and batch/incremental notebook parity tests; no redesign of optimizer, HMM, or evaluation math.
- **Next mathematical checkpoint:** review `trajectory_metadata`, feature provenance, one-patient state reset, temporal missingness semantics, and model bundle feature agreement against the Codex output.
- **Next engineering task after review:** implement an **experiment-only** policy interface (`propose(context) -> {q, available, reason, diagnostics}`), maintaining backward-compatible routes. Do not activate learned hybrid as default without grouped comparison.
- **Next experimental task:** training-only moment estimator (with shrinkage and min-count fallbacks) and a preregistered comparison vs static and training-selected constant blend.
- **Later:** HMM filtered-belief ablation, conditional-information estimators, delayed-feedback protocol, extended metric registry, database ingestion, frontend, CI/CD and AWS deployment after baseline functionality.

## 8. Decisions already fixed versus research questions

**Fixed:** static fallback is permitted and can be the best policy; raw Sepsis and BIDMC have different time units and feature pipelines; Bernoulli classification and Gaussian forecasting use different scoring distributions; feedback has an availability timestamp; patient-disjoint evaluation; emergency fallback overrides switching smoothness; information-based proposals are not assumed calibrated.

**Research questions:** whether moment estimates are informative enough for covariance gating; whether anticipatory gain is predictable; whether HMM beliefs provide incremental information; whether a learned supervisor beats a constant blend without worsening tail risks; which regularizers achieve a favorable bias-variance-risk trade-off under distribution shift.

**Stop/go rule:** Keep hybrid experimental unless its incremental benefit against a simple policy is credible on patient-disjoint validation and adverse-tail metrics remain acceptable. Lack of observed benefit is itself a result; do not force adaptation for the sake of visual variation.
