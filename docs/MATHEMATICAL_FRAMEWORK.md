# Mathematical framework and assumptions

## 1. Allocation and covariance risk
For 2 pathways, let `w = (w1, w2)` with `w1 + w2 = 1`, `w_i >= 0` and covariance matrix `Σ`.

$$R(w)=w^\top\Sigma w=w_1^2\sigma_1^2+w_2^2\sigma_2^2+2w_1w_2\sigma_{12}.$$

Project or shrink empirical covariance to a positive semidefinite matrix before risk optimization. Covariance can represent redundancy or prediction-error dependence; its relevance to utility is *not guaranteed*.

## 2. Conditional value and regret
Let `S` be a selected set of memories with budget `|S| <= B` and `V(S|x)` its realized downstream value.

$$\mathrm{Regret}(S)=\max_{|A|\le B}V(A\mid x)-V(S\mid x).$$

When the value oracle is synthetic, this is a simulation metric, not observed agent improvement. In new environments define task utility and feasible counterfactual evaluation *before* running experiments.

## 3. Incremental usefulness of covariance

$$\Delta_t=\mathcal L_{\text{static},t}-\mathcal L_{\text{cov},t},\qquad
\widehat\Delta_t=\widehat{\mathbb E}[\Delta_t\mid x_t,\mathcal H_{t-1}].$$

Positive `Δ` means the covariance policy improves a lower-is-better loss. `x_t` must be fully observable before the decision; `H_{t-1}` is past outcome history only.

## 4. Trust-aware allocation (candidate, unvalidated)

$$q_t=\mathrm{sigmoid}\!\left(\frac{\widehat\Delta_t-\kappa\widehat u_t-\tau}{T}\right),\qquad
w_t=(1-q_t)w_{\mathrm{static}}+q_tw_{\mathrm{cov}}.$$

`q_t` is an exposure weight, **not** automatically a calibrated probability. `u_hat` is useful as a lower-confidence adjustment only after independent calibration. Blending pathway weights is not equivalent to randomizing between two discrete top-k policies.

## 5. Hybrid controller objective (research candidate)

$$\min_{w\in\mathcal W}\; -\widehat V(w\mid x_t,b_t)+\lambda_r w^\top\widehat\Sigma_{\mathrm{PSD}}w+\lambda_s\|w-w_{t-1}\|_2^2+\lambda_m C(w)+\lambda_u U_{\mathrm{epistemic}}(w).$$

Treat terms as ablatable components, not a validated combined model. Scale and validate each term. Optional HMM belief `b_t` requires identifiable/observable regimes.

## 6. Experimental hygiene
Use patient/recording-level grouping, temporal causality, separate tuning and testing, paired comparisons, ablations, bootstrap at independent cluster level, shift checks, and appropriate uncertainty metrics. Fallback when reliability is weak or uncalibrated.
