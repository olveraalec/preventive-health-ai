# Trusted local retrospective Sepsis demo

The existing `artifacts/models/prototype_bundle.joblib` was inspected against
Notebook 09's export cell. No model was retrained or binary rewritten. Activation
uses `prototype_bundle.manifest.json`, a local sidecar bound to the binary SHA-256.
Both files are ignored by Git. This is a research prototype, not clinical validation.

## Compatibility and input order

Notebook: `notebooks/09_nested_ctonroller_benchmark.ipynb` (existing filename).
The source digest for cells 0, 1, 2 and 5 is
`304d743becef8a8e587899e7a55f2b42e4bea7a356d697cf30207b150fd18670`.

Fitted input order agrees exactly with the stored feature lists:

- Static, 13 features: `Age`, `Gender`, `HospAdmTime`, `ICULOS`, `Unit1`, `Unit2`,
  then `HR_current`, `O2Sat_current`, `Temp_current`, `SBP_current`, `MAP_current`,
  `DBP_current`, `Resp_current`.
- Temporal, 41 features: the static sequence, then `last`, `mean6`, `std6`,
  `delta3` for each vital in order `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`,
  `Resp`. Full names and ordered hashes are stored in the manifest.
- Controller, seven features: `ICULOS`, `HR_observed`, `Resp_observed`,
  `Temp_observed`, `HR_age`, `Resp_age`, `entropy_static`.

All three pipelines use training-median `SimpleImputer` (NaN missingness,
no added indicators, `keep_empty_features=False`) followed by centered/variance
scaled `StandardScaler`. Base models are `LogisticRegression(C=0.01,
max_iter=1000, random_state=12345)` with classes `[0, 1]`. The controller is
`Ridge(alpha=10.0)` fitted to training-patient OOF Bernoulli log-loss gains with
equal total weight per patient. The verifier permits the export cell's removal of
training-all-missing base features, retaining order.

The dataset is `sepsis2019`; the target is `shifted_challenge_label`, not a verified
clinical-onset probability. Feature builder metadata is defined in
[SEPSIS_FEATURES.md](SEPSIS_FEATURES.md).

One necessary runtime compatibility correction aligns `entropy_static` with
Notebook 09's exact `binary_entropy`: clip log arguments at `1e-10`, retaining
original probabilities as weights. The previous runtime clipped both at `1e-12`.
Extreme-probability tests cover this. Allocation, covariance, HMM and evaluation
definitions remain unchanged.

## Reconstructed provenance, not certification

The legacy artifact contains no historical model/run version, environment manifest
or fit logs. Its new model version is the observed binary content hash, not an
invented training version. The manifest stores the verified current environment;
historical training software versions are explicitly `not_recorded_in_legacy_export`.
sklearn loading must produce no `InconsistentVersionWarning`. Sidecar loading
requires exact verified runtime versions; changed environments require re-verification.

Observed training consistency:

- Patient-disjoint source splits: 1,842 train, 395 validation, 395 test.
- All 71,182 training rows have paired stored static/temporal OOF predictions.
- OOF coverage, original outcomes and three grouped outer-fold assignments reproduce
  the notebook's splitter exactly.
- Static, temporal and controller medians, scaler means and population variances
  match reconstruction from training rows exactly.
- The controller satisfies the weighted-ridge normal equations for those OOF gains:
  maximum stationarity residual `3.2871e-13`; intercept residual `2.6645e-14`.
- Verification makes no fitting calls and does not fit/tune on validation/test rows.

These checks support consistency with the training-only export but cannot prove
every historical fit or OOF job. The status is
`reconstructed_consistency_not_training_attestation`. The manifest records whole-file
evidence hashes and aggregate diagnostics, never patient identifiers, raw rows,
per-patient predictions or labels. A checksum is not a signature and does not make
an arbitrary joblib file trustworthy.

## Upgrade procedure and retrospective demo

From the repository root:

```bash
uv run --no-sync python -m healthcare_ai_lab.experiments.verify_sepsis_bundle --activate
```

This performs the metadata upgrade and patient demo. It checks the trusted binary,
pinned notebook source, splits, OOF evidence and preprocessing; selects one local
validation patient; compares every hour's generated features and raw-hourly API
predictions with the offline-engineered history; and independently checks batch base
probabilities. Only aggregate diagnostics are printed. The sidecar is written
atomically after all checks pass, then loaded through the application's loader.
The model binary remains unchanged. Omit `--activate` for a read-only audit.

Required existing files:

```text
artifacts/models/prototype_bundle.joblib
notebooks/02_temporal_feature_engineering.ipynb
notebooks/09_nested_ctonroller_benchmark.ipynb
data/processed/sepsis_development.parquet
data/processed/sepsis_temporal_features.parquet
data/processed/sepsis_patient_splits.csv
results/experiments/nested_controller_predictions.parquet
```

Use `--root /path/to/repository` and `--artifact /path/to/trusted/bundle.joblib` to
locate another copy. Missing evidence, changed source, incompatible software,
invalid metadata, checksum mismatch or numerical mismatch stops verification.
Restore original evidence or use an explicitly reviewed Notebook 09 export in its
recorded environment. The verifier never retrains automatically. Any reviewed
replacement export must preserve cells 0/2/5's training-only data, grouped
cross-fitting, gain labels and patient weights; never substitute validation/test
rows into training or controller fitting.

The verified runtime is Python 3.12.13, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1,
scikit-learn 1.9.1 and joblib 1.6.0. The real one-patient test covered 37 hours:
features, static/temporal/final probabilities and controller weights matched exactly.
Independent batch temporal probabilities differed by at most `1.2143e-17`, below
the predeclared `1e-12` tolerance. No fallback occurred. No patient rows were persisted.

To start the research API after verification:

```bash
uv run --no-sync uvicorn healthcare_ai_lab.api.main:app --host 127.0.0.1 --port 8000
```

Use `POST /api/v1/research/sepsis/trajectory/raw-hourly`, with `model` metadata from
the local sidecar or verifier output. The default loader attaches sidecar metadata
automatically; engineered endpoints remain available. Restart any running API after
a sidecar change because its loader is cached. `/ready` remains the original stub,
not an artifact readiness check. No deployment was performed.

Parity establishes retrospective inference compatibility, not clinical validity,
calibrated reliability or a hybrid-controller benefit. Bounded exposure/step settings
remain configured parameters. Raw units and actual observation availability remain
source assertions.
