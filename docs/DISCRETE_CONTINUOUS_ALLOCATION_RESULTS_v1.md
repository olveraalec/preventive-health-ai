# Discrete versus continuous allocation: experiment v1

Executed 2026-10-28. Retrospective research only; the target remains the Sepsis
Challenge **shifted label**. Continuous superiority and clinical improvement were
not established. No serving policy is promoted or changed.

The experiment was fixed in [protocol v1](DISCRETE_CONTINUOUS_ALLOCATION_PROTOCOL_v1.md)
before evaluation. The six requested policies and four predeclared ablations use
identical frozen expert probabilities, patient-hours and evaluation splits. Only
small experimental controllers are fitted in memory. No base model is trained,
artifact overwritten, frontend edited, API changed, or reserved real test row selected.

## Executed design and provenance

- Seed 12345; 2,000 paired whole-patient bootstrap replicates; nominal 95%
  percentile intervals. Gains are reference loss minus candidate loss, so positive
  favors the candidate. Intervals condition on fitted policies and do not correct
  exploratory reuse, controller search, or multiple comparisons.
- Synthetic: 900 training / 300 fresh evaluation patients, 12 hours each. Three
  context conditions share exactly the same expert forecasts, outcomes and patient
  splits: informative, uninformative, reversed context at evaluation only. Expert
  generators never fit labels; they are not trained Sepsis models.
- Real: original 1,842 training patients / 71,182 OOF hours; original, previously
  inspected validation: 395 patients / 15,409 hours. Training roles contain
  1,105 fit / 368 selection / 369 reliability patients. The original reserved test
  cohort (395 patients per existing manifest) remains unselected by this runner.
  Its historical untouched status cannot be independently certified here.
- Original Notebook 09 outer training OOF folds are reconstructed and matched;
  expert targets, patient-hour keys and source hashes are checked. Inner OOF tables
  were not persisted. Global OOF experts can have fitted other inner-role patients,
  so the internal reliability bound is a **non-nested selection heuristic**.
  Original validation patients never enter controller fitting/selection/checking.
- The existing artifact/manifest provides reconstructed consistency evidence,
  not independent training attestation. Frozen training OOF experts and the
  all-training fitted artifact experts necessarily have different fit sizes.
- Hard and continuous share the same seven decision-time metadata inputs and
  Ridge gain estimator. Search budgets: 101 constant weights, 8 hard candidates,
  37 continuous candidates per variation penalty. Continuous has a larger search
  budget; this experiment does not isolate continuity from search complexity.
- Existing bounded anticipation uses the unchanged trusted artifact controller
  on real data; synthetic uses the same Notebook 09 estimator form fitted on all
  synthetic training patients. New controllers use the fit role only. Its bounds
  (q ≤ 0.25, step ≤ 0.10) differ from the full-range experimental candidates.
- Allocation is frozen without an outcome column. Outcome availability is annotated
  at an offline replay cutoff solely for scoring; it is not measured clinical
  availability. No reactive branch or delayed-label simulation is used.

## Real validation results

These are exploratory validation results, not fresh confirmatory holdout results.
Log loss, Brier and CVaR below use equal patient weighting. ECE uses the existing
registry's pooled ten-bin definition. Lower is better for these columns.

| Policy | Patient log loss | Patient Brier | ECE | Patient CVaR90 log loss | Mean squared q step |
|---|---:|---:|---:|---:|---:|
| Static | 0.151740 | 0.034195 | 0.003211 | 1.375451 | 0 |
| Temporal | 0.151848 | 0.034239 | 0.003164 | 1.382215 | 0 |
| Constant q=0.21 | 0.151520 | 0.034192 | 0.003324 | 1.374447 | 0 |
| Selected hard switch (static) | 0.151740 | 0.034195 | 0.003211 | 1.375451 | 0 |
| Reliable continuous (retains q=0.21) | 0.151520 | 0.034192 | 0.003324 | 1.374447 | 0 |
| Existing bounded anticipation | 0.151538 | 0.034186 | 0.003436 | 1.374084 | 0.00008920 |

The training reliability check rejects continuous adaptation: incremental patient
log-loss gain against q0 is −0.000313, CI [−0.001116, 0.000488]. The fitted ungated
candidate uses exposure 1, temperature 0.005, threshold 0.01218509; the gate retains
q0 rather than presenting this candidate as supported adaptation.

Reliable continuous minus-reference results (gain = reference minus continuous):

| Reference | Mean patient gain | 95% patient-cluster CI |
|---|---:|---:|
| Constant mixture | 0 | [0, 0] |
| Hard switch / static | 0.00021940 | [−0.00037519, 0.00080119] |

Against static, 66.84% of patients improve, but the worst patient gain is
−0.04292448 and the bottom-decile mean gain is −0.00612903. Positive-patient mean
loss is 1.503064 versus static's 1.504837. These descriptive tail differences do
not override the uncertainty interval or certify safety. Existing anticipation
versus constant yields −0.00001752 [−0.00030340, 0.00029021]. All real policy
differences remain small; no evidence here justifies adaptation over the constant.

## Synthetic controls and negative findings

All three cases use the same 300 held-out patients and base predictions. In the
uninformative case the hard controller selected temporal-only; this is an eligible
training-selected endpoint, not an implementation failure.

| Policy / patient log loss | Informative | Uninformative | Reversed context |
|---|---:|---:|---:|
| Static | 0.352422 | 0.352422 | 0.352422 |
| Temporal | 0.364584 | 0.364584 | 0.364584 |
| Constant q=0.43 | 0.327357 | 0.327357 | 0.327357 |
| Hard switch | 0.304019 | 0.364584 | 0.412986 |
| Reliable continuous | 0.304383 | 0.327357 | 0.407934 |
| Existing bounded anticipation form | 0.329135 | 0.340702 | 0.352991 |

| Case / reference | Continuous patient gain | 95% CI |
|---|---:|---:|
| Informative / constant | 0.02297408 | [0.01624590, 0.02986655] |
| Informative / hard | −0.00036396 | [−0.00124787, 0.00068128] |
| Uninformative / constant | 0 | [0, 0] |
| Uninformative / hard | 0.03722631 | [0.02663569, 0.04854625] |
| Reversed / constant | −0.08057682 | [−0.09185691, −0.06974273] |
| Reversed / hard | 0.00505230 | [0.00171307, 0.00902857] |

Informative context supports adaptation over a constant in this constructed
environment, but does **not** establish continuous superiority over hard switching.
With uninformative context the training check abstains; reliable continuous is
exactly the constant mixture, and its advantage over hard is diversification,
not justified contextual adaptation. Ungated continuous instead worsens loss to
0.329278. Under reversed context the training gate remains enabled and both adaptive
policies are harmful versus constant/static. A training evidence gate does not
detect distribution shift or provide a calibrated reliability probability.

## Accuracy versus variation ablation

Parameters were selected using training roles only; no evaluation-driven retuning.
One-hour patients contribute zero to the **selection penalty**. The existing
registry excludes their undefined step means when reporting controller dynamics.

| Informative synthetic policy | Patient log loss | Mean squared q step | CVaR90 log loss |
|---|---:|---:|---:|
| Hard switch | 0.304019 | 0.134242 | 0.747960 |
| Ungated continuous, lambda=0 | 0.304383 | 0.136384 | 0.747602 |
| Continuous, lambda=0.01 | 0.304383 | 0.136384 | 0.747602 |
| Continuous, lambda=0.1 | 0.306653 | 0.096596 | 0.743409 |
| Ungated proposal with step cap 0.10 | 0.322399 | 0.005439 | 0.792354 |

A continuous-valued policy is not automatically smoother than a hard policy. The
stronger variation penalty lowers variation with a modest log-loss cost; the step
cap lowers variation substantially but worsens accuracy and CVaR in this example.
On real validation both penalty searches select the same ungated parameters as
lambda=0: loss 0.151523, squared step 0.000480. Step capping yields 0.151523 and
0.000456. No credible advantage over constant emerges; zero adaptation is valid.

## Reproduce and inspect

From the repository root, using the existing locked Python 3.12 environment:

```bash
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m healthcare_ai_lab.experiments.allocation_comparison --scenario informative --output results/experiments/allocation_informative_01.json
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m healthcare_ai_lab.experiments.allocation_comparison --scenario uninformative --output results/experiments/allocation_uninformative_01.json
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m healthcare_ai_lab.experiments.allocation_comparison --scenario shifted --output results/experiments/allocation_shifted_01.json
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m healthcare_ai_lab.experiments.allocation_comparison --source local-validation --output results/experiments/allocation_validation_01.json
```

Each command refuses to overwrite an existing output. Optional `--seed`,
`--bootstrap`, synthetic cohort sizes and `--root` are exposed; these results use
the defaults. No network, database, credentials or LLM provider is required.
The local source requires the existing trusted artifact plus checksum-bound
manifest, Notebook 09 training OOF cache, and processed temporal Parquet; it fails
closed if these are missing, mismatched, or folds/labels/expert coverage disagree.

Canonical local outputs are ignored aggregate JSON files at
`results/experiments/discrete_continuous_v1_audited_{informative,uninformative,shifted,local_validation}.json`.
They include all metric directions/statuses, AUPRC/AUROC, calibration bins, coverage,
paired Brier/log-loss gains and tails, win/tie rates, sign-flip tests, allocation
variance/changes/switching/fallback and expert disagreement, seeds/counts/search
budgets, software and input/runner/protocol hashes. There are no patient tables,
identifiers, controller coefficients, raw outcomes, or model binaries in exports.
All four runs have complete prediction coverage, zero emergency fallback events
and no undefined aggregate metric records. Synthetic failure tests exercise fallback.

Runtime preserves `(1-q)*S + q*T`, q in [0,1], existing bounded anticipation and
immediate static failover even when it violates ordinary step constraints. This
is an additive experimental scalar gate, not the proposed hybrid/SRM supervisor.
Covariance/PSD optimization, HMM mathematics, Gaussian and Bernoulli definitions,
feature engineering and artifact provenance are unchanged. All gate thresholds,
temperatures, constraints and regularization strengths are configured research
choices, not empirically calibrated safety/trust parameters.

The next research step is a prospectively frozen protocol with truly nested
training OOF evidence for reliability, fixed controller complexity/bounds and
multiple seeds before any use of a certified untouched real holdout. These results
support keeping adaptation unpromoted. They do not authorize base-model retraining,
using the reserved test set for tuning, or modifying serving/API behavior.

## Verification receipt

Baseline: 206 passed, 15 opt-in PostgreSQL tests skipped. Final executed checks:

```bash
PYTHON_DOTENV_DISABLED=1 RUN_DB_TESTS=0 uv run --no-sync pytest -q tests/test_allocation_comparison.py tests/test_allocation_sources.py
# 56 passed
PYTHON_DOTENV_DISABLED=1 RUN_DB_TESTS=0 uv run --no-sync pytest -q -rs
# 262 passed, 15 skipped (tests/test_database_integration.py: Requires local PostgreSQL)
uv run --no-sync ruff check src/healthcare_ai_lab/experiments/allocation_comparison.py src/healthcare_ai_lab/experiments/allocation_sources.py tests/test_allocation_comparison.py tests/test_allocation_sources.py
uv run --no-sync ruff check --config ruff-ci.toml .
git diff --check
# All pass; new untracked files also checked individually against /dev/null.
```

Local commands used `UV_CACHE_DIR=/private/tmp/healthcare-uv-cache`. PostgreSQL was
not exercised in this milestone; no persistence code changed. GitHub CI was not
run remotely. Tests cover deterministic/permutation invariance, label-free
allocation, evaluation and reliability-role leakage, OOF fit witnesses, patient
overlap, patient state/prefix isolation, Brier gain/log-loss convexity and mixture
identities, feasible weights/steps, single-reference abstention, failure overrides,
malformed inputs and local split/source integrity. Two added regressions first
reproduced numeric-hour ordering/duplicate normalization and infinite gain-signal
failover errors in the new runner; both pass after correction. The final results
retain exactly the original run's allocations and metrics after these guards and
reporting additions; no parameters were changed after evaluation.

All four audited JSON outputs were checked against current runner/adapter/protocol
hashes, original allocation digests and scalar metrics, complete shared coverage,
finite JSON and absence of episode identifiers. Software: Python 3.12.13, NumPy
2.5.4, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0.

| Evidence | SHA256 |
|---|---|
| Runner | `4b799e040b4011771f4d384acbe12e788448100395e2f9d1518a6d340c00cfe1` |
| Source adapter | `16011513afe725cc0f300f8fe3854d4b529259fee4d290f3aa6207444d61bb4a` |
| Protocol | `2f4569524466b8933f3cb6653272d0e2b033ffae4d95f9d7cafe7207a1bbdcec` |
| Unchanged model bundle | `16ac3e6deb025a82de6806ecefe5c1af4b31d2742563e600385113d4caf5dde2` |
| Unchanged local artifact manifest | `e22a0dcaa8490b1d477d9a5dcbae8f2e273888637bb2404df98ad425c987855e` |
| Frozen training OOF source | `e9ca7afa91241ab43c1e7360840065d87b24b17101ffea0eb823c120ed4a72f9` |
| Frozen engineered source | `6a732f4e7f3f51d370836ef676659ba83187d023e98b2f4e830f864c4d490967` |

Added files: `experiments/allocation_comparison.py`, `experiments/allocation_sources.py`
inside the installed package; their two matching pytest modules; this result note
and protocol. `ruff-ci.toml` adds these modules/tests to the existing clean scope.
Pre-existing evaluation implementation/worktree changes were reused and preserved.

Consulted root `AGENTS.md`, research reading map/index, mathematical architecture
v0.4, Sepsis evaluation v0.1, reliability/information theory v0.3, conditional
allocation v0.2, synthetic allocation sanity v0.1, Notebooks 09/10 and existing
patient-disjoint/continuous-memory/covariance experiments. Prior research documents
and their mathematical definitions remain unchanged. No fresh real-cohort
confirmation, nested reliability certification, online feedback validation,
multi-seed robustness or clinical improvement is claimed.
