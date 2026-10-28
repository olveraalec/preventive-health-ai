# Versioned causal Sepsis features

The `healthcare_ai_lab.preprocessing.sepsis` module extracts Notebook 02 cells 9–10.
It is separate from BIDMC seconds-based telemetry. The hybrid supervisor is not
implemented: static, anticipatory, reactive and covariance-aware allocations remain
distinct candidates for a future hierarchical supervisor.

Version identity:

- Feature schema: `sepsis2019.notebook02.temporal.v1`
- Preprocessing version: `1`
- Pipeline: `notebook02.cells9-10`
- Transformation-cell SHA-256:
  `46b9a84a5d38188de7c5ddc0423bf7fb65e1e95865c419456411c901044b8527`

The batch builder accepts a DataFrame for exactly one patient with `patient_id`,
`ICULOS`, seven vitals (`HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`), and
`Age`, `Gender`, `HospAdmTime`, `Unit1`, `Unit2`. Columns are required even when all
their values are missing. Explicit NaN/null measurements remain missing; omission
of a required source column is an error. Other columns never enter features.

Every vital produces the notebook's exact names and transformations:

| Suffix | Transformation |
| --- | --- |
| `_current` | Current raw value, including NaN |
| `_observed` | Current nonmissing indicator, int8 |
| `_last` | Forward-fill within this patient's observed prefix |
| `_age` | Current `ICULOS` minus latest observed measurement's ICU hour |
| `_mean6` | Mean of forward-filled values over six trailing records; minimum two nonmissing |
| `_std6` | Sample standard deviation over the same window (`ddof=1`) |
| `_delta3` | Forward-filled current value minus value three records earlier |

Initial last-value/age and insufficient-history statistics remain NaN. Repeated
forward-filled values count as records in the trailing summaries, exactly as in
the notebook. There is no separate slope, slope normalization, sparse-lab feature,
backfill, or imputation in these transformation cells.

Interactions are `HR_last * MAP_last`, `Resp_last * O2Sat_last`,
`HR_delta3 * MAP_delta3`, and `Resp_std6 * Resp_delta3`. Names are respectively
`HR_x_MAP`, `Resp_x_O2Sat`, `HRtrend_x_MAPtrend`, `Respstd_x_Respdelta`.
`ICULOS_sin24` and `ICULOS_cos24` use `2*pi*ICULOS/24`; they retain the notebook's
experimental admission-time encoding, not a claim of physiological periodicity.

All source units are retained without conversion. Ages are ICU hours; standard
deviations and deltas have their source measurement's units, and products retain
product units. The notebook does not independently verify measurement units.
`feature_sources()` records the source columns for every predictive feature.

`ICULOS` must contain strictly increasing, unique, finite positive integers. Gaps
are allowed: no rows are inserted; six-record windows and three-record deltas keep
their notebook meaning. Measurement ages use elapsed hours across gaps. Histories
may begin after hour one; no history before the first supplied observation is
assumed. An update requires a complete source-column set, with nulls where unmeasured.

`IncrementalSepsisFeatures(patient_id).update(observation)` returns a one-row feature
DataFrame. Each instance owns one patient's history; invalid updates leave that
history unchanged. The reference implementation recomputes each prefix using the
batch builder, preserving pandas rolling arithmetic. It retains O(n) history and
uses O(n²) total work over n updates; it is not an optimized long-running stream.

The notebook's `patient_id` is retained for grouping, but `SepsisLabel` and `split`
are excluded from predictive output. Numeric nulls are normalized to NaN. Age
columns are float even on an initially complete prefix, avoiding a pandas dtype
change when a later missing measurement appears. These are representation choices,
not changed mathematical features.

## Raw-hourly research API

`POST /api/v1/research/sepsis/trajectory/raw-hourly` accepts:

```json
{
  "dataset": "sepsis2019",
  "episode_id": "synthetic-raw",
  "model": {"artifact_id": "synthetic", "model_version": "1"},
  "observations": [{
    "episode_id": "synthetic-raw", "icu_hour": 1,
    "measurements": {
      "HR": 90, "O2Sat": 97, "Temp": null, "SBP": 120,
      "MAP": 75, "DBP": 65, "Resp": 18,
      "Age": 55, "Gender": 1, "HospAdmTime": -12,
      "Unit1": 1, "Unit2": null
    }
  }]
}
```

Optional standard Challenge measurement columns are accepted but unused by this
version. Unknown/engineered, outcome, and embedded identity columns are rejected.
The API builds features server-side and reuses the existing trajectory service.
The engineered-row and engineered-trajectory endpoints remain available.

The trusted loaded bundle must declare matching `trajectory_metadata` with dataset,
the requested artifact/model version, and the runtime provenance above plus
`declaration: current_and_past_only`. Its static/temporal feature lists must be
subsets of the generated schema; controller features additionally may request
`entropy_static`, calculated by existing inference. Unsupported feature names or
metadata mismatches reject raw inference, rather than substituting missing columns.
No existing artifact was rewritten or automatically attested. For explicit local
checksum-bound verification and activation, see
[SEPSIS_ARTIFACT_VERIFICATION.md](SEPSIS_ARTIFACT_VERIFICATION.md).

Responses retain research/non-clinical labels, configured uncalibrated controller
parameters and static fallback behavior. They add source-column lineage, gap policy,
unit policy and the transformation hash. Provenance status is
`runtime_generated_source_availability_unverified`: generation is server-owned,
but raw units and true measurement availability remain source assertions.

## Parity evidence and limitations

`tests/fixtures/sepsis_notebook02_synthetic.json` was generated by executing the
original cells directly on two entirely synthetic histories. It freezes all 62
runtime columns, including patient identity (61 predictive columns), and covers
missing prefixes, gaps, all-missing channels, rolling-window truncation and products.
Tests compare every value and NaN location, with numerical tolerance 1e-13. Tests
also establish batch/incremental equality, prefix causality, failed-update isolation,
API schema rejection and synthetic inference. Optional tests inspect the local
notebook fingerprint and processed Parquet **schema only**, never patient records.

This establishes synthetic transformation parity with the inspected notebook cells.
It does not establish row-level parity with confidential processed data, fitted model
training provenance, clinical validity, or source measurement timestamps. Raw model
inference remains gated on explicit compatible artifact metadata. The subsequent
[artifact verification milestone](SEPSIS_ARTIFACT_VERIFICATION.md) loads the trusted
local bundle and checks one actual trajectory without certifying historical training.
Verified delayed feedback,
raw-data unit/availability validation and independent model evaluation remain future
work. This feature builder does not implement the hybrid controller supervisor.
