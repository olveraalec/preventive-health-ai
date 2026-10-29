# Discrete versus continuous allocation — frozen experiment protocol v1

Date: 2026-10-28. Written before this runner's evaluation. Research only; this
does not change runtime controllers, Gaussian/Bernoulli metrics or API behavior.

Read: root AGENTS, research reading map/index, architecture v0.4, evaluation v0.1,
reliability v0.3, conditional allocation v0.2, synthetic sanity v0.1, Notebooks 09/10,
and existing continuous-memory/covariance experiments. The old synthetic experiments
score continuous-response lag specialists and may not be reported as sepsis evidence.
Notebook 10 uses 1e-12 clipping; this experiment uses the standardized registry's
1e-10 convention without rewriting historical results.

## Fixed design and hypotheses

Every policy uses identical frozen S/T probabilities and evaluation patient-hours.
Only experimental allocation parameters are fitted; no base predictor is fitted or
model binary written. Primary contrast: reliable continuous versus hard switch.
Required strong reference: training-selected constant mixture. Static and temporal
endpoints and the existing bounded anticipatory controller are retained.

Original training patients alone supply OOF base probabilities. Deterministic
patient-disjoint inner roles are 60% gain/constant fitting, 20% parameter selection,
20% reliability checking (seed 12345). These are experiment-internal roles, not
changes to the existing train/validation/test assignments. No final refit after the
reliability check: the policy whose gain is checked is the policy evaluated.

Fit a Notebook-09-style median-imputer/scaler/Ridge(alpha=10) on static-minus-temporal
Bernoulli log-loss gain, with equal total weight per fitting patient. Inputs are
exactly the existing seven meta features: ICU hour, HR/Resp/Temp availability,
HR/Resp age and notebook-compatible static entropy. Outcomes are excluded from
allocation inputs. This is an experimental conditional-gain model, not calibrated trust.

- `static`: q=0.
- `temporal`: q=1.
- `constant_mixture`: q selected on fitting OOF patients from 101 evenly spaced
  values in [0,1] by patient-mean log loss. Ties choose the smaller q.
- `hard_switch`: q in {0,1}; threshold chosen on selection patients from zero and
  fitting-gain quartiles (plus always-static/always-temporal candidates).
- `continuous`: start with `q0 + rho*(sigmoid((gain-threshold)/T)-q0)`;
  use the same finite threshold grid as hard switching, T from {0.005, 0.02, 0.1},
  and rho from {0, 0.5, 1}, selected on selection patients by patient log loss.
  The rho=0 candidate is evaluated once; the other candidates share the fixed grid.
  A whole-patient bootstrap of incremental gain against q0 on reliability patients
  enables adaptation only if the lower 95% percentile bound is positive and at
  least 20 patients contribute. Otherwise retain q0, possibly static indefinitely.
- `bounded_anticipatory`: use the existing adaptive_allocation function, sigmoid
  temperature 0.01, exposure 0.25, static fallback, max departure 0.25 and step 0.10.
  Real-data replay uses the existing trusted artifact's gain model; synthetic replay
  fits the same estimator form on all synthetic training OOF rows for this baseline.

Predeclared ablations: selected continuous before reliability abstention; that same
ungated proposal with per-observation step cap 0.10; and two ungated continuous
parameter selections minimizing log loss plus lambda times patient-mean squared
allocation change, lambda in {0.01, 0.1}. A one-hour patient's selection penalty is
zero, as in the research contract. No ablation is selected using evaluation labels.
All settings are experimental/configured, not empirically calibrated probabilities
or safety thresholds. Full-range hard/soft candidates and the bounded existing
controller have different action constraints; report this explicitly.
Scores within 1e-12 of the minimum are numerical ties; choose the first candidate
(smallest constant, always-static hard switch, or zero-exposure continuous reference).
This prevents spurious allocation preferences when the experts agree.

## Evaluation and provenance

Use evaluation.registry.evaluate_classification_policies: patient and hourly log
loss/Brier, AP/AUC, calibration/ECE, retrospective strata, CVaR90, worst/p10/bottom
decile gains, patient wins, cluster-bootstrap CI and controller variation. Compare
all policies with q0, and continuous with hard switching and static directly.
Default resampling: 2,000 patient replicates, seed 12345. No hourly bootstrap.
Nominal pairwise intervals are exploratory, not multiplicity-adjusted guarantees.
Promotion requires credible gain against hard **and** constant and no increase in
CVaR90 or positive-patient mean log loss relative to both; otherwise keep adaptation
unpromoted. An already inspected real validation set cannot authorize promotion.

Synthetic experts are fixed label-independent stochastic forecast generators, not
trained sepsis models. Their empty fitting sets make their training predictions
out-of-sample with respect to labels by construction. Context-informative and
context-uninformative controls share frozen seeds/design; an optional flipped-context
shift tests negative transfer. Synthetic holdout labels are opened only for scoring
after allocations are frozen. Results do not establish real-cohort effectiveness.

Local real-data source: checksum-verified Notebook 09 training OOF table and the
existing artifact's base forecasts on the **previously inspected validation** split.
The reserved test split is never read/scored. No claim that validation is a fresh
confirmatory holdout is permitted. Inner OOF tables from Notebook 09 were not
persisted, so globally OOF training experts may have fitted patients belonging to
another inner policy role. All such patients remain inside original training; no
outer validation patient enters fitting. The inner reliability bound is consequently
a non-nested selection heuristic, not independent/certified reliability evidence.
Do not repartition the global OOF table and call that a new nested outer evaluation.

Real shifted-label availability is not inferred from a clinical onset offset. Scoring
uses a declared offline replay cutoff after all supplied outcomes; no outcome is
provided to allocation and reactive feedback is disabled. This is retrospective
evaluation-only availability, not an online feedback protocol.

Only aggregate metrics, settings, counts and whole-file hashes may be saved or
printed. No patient IDs, raw features/labels, per-patient probability tables, model
binaries or coefficients are exported. Existing evidence/manifests remain unchanged.
