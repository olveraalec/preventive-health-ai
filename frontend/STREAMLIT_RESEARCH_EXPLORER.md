# Research Explorer — pinned three-service demonstration

Verified 2026-10-28 against `origin/dashboard-demo` commit
**`c28e2ad51f2b9304f99637f6f720e41f2be92bdf`**. This supersedes the earlier ccf5b3f
source-only audit. The historical audit remains in frontend commit 9b9c280.

React remains the primary product. Streamlit runs in its own detached worktree and
virtual environment; no Streamlit chart is rebuilt or embedded in React. The React
visual design, canonical API client, 37-hour importer, backend, model artifacts,
mathematical implementations and evaluation source/results are unchanged.

## Reviewed pages and data contracts

| Page | Available exploration | Source boundary |
|---|---|---|
| `dashboard/app.py` — Patient Monitoring | Episode selector, threshold slider, final/static/temporal overlays, HR/MAP/Resp/Temp traces, retrospective threshold-crossing and label-dependent lead/alert summaries | Reads `data/predictions.csv`; without it generates 40 simulated episodes and displays a strong simulated-data warning. Thresholds are exploratory, not treatment triggers. |
| `dashboard/pages/1_Adaptive_Decisions.py` | Probability, temporal allocation, disagreement and entropy traces; fallback shading; stable/adapting/fallback counts; user-controlled movement tolerance | Reads the same CSV; without it generates three hand-made 48-hour episodes and warns the allocation pattern is not model behavior. Stable/adapting is a display rule. |
| `dashboard/pages/2_Model_Evaluation.py` | Policy loss/Brier/approximate AUPRC tables, patient-bootstrap intervals, paired patient change boxplot, positive-patient change, worst-patient/tail summaries and calibration curves | Reads **`data/eval_predictions.csv`**, independently of the monitoring file. Without it generates 300 simulated episodes and explicitly warns constructed blending gains are not experimental results. |

Both monitoring loaders and the evaluation loader now include the CSV modification
time in their cache key. Missing files select simulated mode independently: an
existing monitoring CSV does not make the evaluation page real, or vice versa.

### `data/predictions.csv`

One row per episode/hour, sorted by the application:

- Main page requires `patient_id`, `hour`, `risk`, `HR`, `MAP`, `Resp`, `Temp`.
- Optional `static_probability` and `temporal_probability` add expert overlays.
- Adaptive Decisions additionally requires `static_probability`,
  `temporal_probability`, `temporal_weight`, `status` alongside patient/hour/risk.
- Optional `entropy_static` is consumed; otherwise entropy is computed locally.
- Optional `SepsisLabel` or `onset_hour` enables retrospective detection/lead-time
  displays. Without label columns, those metrics/onset interpretation are unavailable.
  Unknown/partial labels must not be treated as verified negative outcomes.

### `data/eval_predictions.csv`

A **separate**, held-out, patient-disjoint evaluation export is needed for actual
model comparisons. The intended contract has `patient_id`, `hour`, binary
`SepsisLabel` and at least two available policy probability columns:

| Policy | Accepted column(s), in preference order |
|---|---|
| Static | `static_probability` |
| Temporal | `temporal_probability` |
| Hard-switching | `hard_switch_probability` (optional policy) |
| Continuous | `continuous_probability`, otherwise `risk` |

The loader checks patient ID and label column presence; `hour` is documented but
not used for key validation. Missing policies are labeled unavailable. Rows missing
any selected policy/label are dropped jointly, and the dropped count is displayed.
The page itself does not establish patient split provenance, unique patient-hour
keys, label completeness, or valid finite probabilities. Validate those properties
upstream before presenting a real CSV. Do not generate labels in live inference.

The unchanged evaluation definitions include:

- Bernoulli log loss with page-specific clipping `1e-12`; Brier and an approximate,
  order-sensitive-tie average-precision calculation (not a claim of exact registry
  AUPRC parity). No AUROC or ECE output is implemented.
- Patient-mean log loss averages within patient first. Percentile bootstrap
  intervals resample patients; the minimum support is 20 patients. The slider ranges
  from 200–4000 resamples, with a memory-based cap in the implementation.
- Paired change is **candidate minus baseline**, so negative means lower loss. This
  has the opposite sign of a reference-minus-candidate gain report.
- “CVaR90” is the mean of the worst `ceil(0.1*N)` patient losses. It is the page's
  finite-sample convention, not automatically the fractional-tail registry estimator.
- Calibration uses equal-count probability bins. These are descriptive diagnostics,
  not a fitted recalibration or proof of clinical reliability.

These definitions were inspected and exercised without editing them. Simulated
calibration, bootstrap intervals and policy differences do not measure improvement
of the activated trained model.

## Existing converter limitations — no private-data conversion performed

`dashboard/convert_trajectory.py` accepts the older
`{request: RawTrajectoryRequest, response: {predictions: ...}}` wrapper. It does not
accept canonical `records` or the sanitized `retrospective-01.json` DTO. It writes
monitoring CSV only, not the evaluation CSV, and optional labels are merged on hour
without enforcing identity/duplicate alignment. Its default output can overwrite
an existing CSV. It was not run.

The 37-hour sanitized export contains no raw vital values or labels; those cannot be
reconstructed to populate either CSV contract. Keep that private export in its
existing ignored location and use React's local importer. A single unlabeled episode
is not a valid cohort calibration/bootstrap report. No adapter is added here.

The main/Adaptive pages label any existing monitoring CSV “Real predictions” based
on presence, not verified provenance. Do not place a synthetic CSV there and claim
real inference from its caption. The launcher below requires **both CSVs absent**
for this safe, explicitly simulated Explorer demo. The branch's ignore rules do not
specifically protect these two CSV paths; do not add private data to this worktree.

The separate `dashboard/train_baseline.py` trains LightGBM with different features
and writes a different output contract. It is not needed for the Explorer and was
not run. No model is trained, replaced or copied by this integration.

## One-time isolated setup

The teammate's branch does not declare Streamlit/Plotly in its pyproject/uv lock.
This frontend supplies `explorer/requirements.in` and a complete hashed
`explorer/requirements.lock`, installed **only in the Explorer's new .venv**.
Validated direct versions: Python **3.12.13**, Streamlit **1.66.1**, Plotly **7.1.0**,
NumPy **2.5.4**, pandas **3.1.0**. All 35 resolved packages are pinned with hashes.
The verified model environment and repository uv lock are not changed. Windows
package hashes are included, but runtime validation here was macOS only.

From `/Users/olveraalec/code/projects/hackathon-aws-publish`, for a fresh setup:

```bash
git fetch --no-tags origin refs/heads/dashboard-demo:refs/remotes/origin/dashboard-demo
git worktree add --detach ../hackathon-streamlit-demo c28e2ad51f2b9304f99637f6f720e41f2be92bdf
uv venv --python 3.12.13 ../hackathon-streamlit-demo/.venv
uv pip sync --python ../hackathon-streamlit-demo/.venv/bin/python \
  --require-hashes frontend/explorer/requirements.lock
```

**These setup steps were already completed locally.** Do not rerun `worktree add`
against the existing directory or overwrite another checkout. The new worktree is
clean/detached at c28e2ad. Neither dashboard CSV exists. The launcher verifies the
pinned revision, requires that no CSV is present, uses the separate Python, binds
only loopback port 8501, and disables usage-statistics collection. It never installs
packages on startup or alters a model environment. Override `EXPLORER_WORKTREE` only
when using another intentionally prepared separate checkout.

## Exact three-service launch commands

Run each block in a separate persistent terminal. Keep already-running owner
processes rather than starting duplicates on the same ports.

**Terminal 1 — existing verified FastAPI, unchanged configuration:**

```bash
cd /Users/olveraalec/code/projects/hackathon-starter
PYTHON_DOTENV_DISABLED=1 uv run --no-sync uvicorn healthcare_ai_lab.api.main:app \
  --host 127.0.0.1 --port 8003 --no-access-log
```

Retain the owner's approved model configuration/environment. Do not install Explorer
packages or copy model files here. `http://127.0.0.1:8003/health` should return 200.

**Terminal 2 — independent Research Explorer:**

```bash
cd /Users/olveraalec/code/projects/hackathon-aws-publish
npm --prefix frontend run explorer
```

This launches `../hackathon-streamlit-demo/.venv/bin/python -m streamlit run
 dashboard/app.py` in that detached worktree, with the explicit loopback/headless/
usage-statistics options in `explorer/start.mjs`. URL: **http://127.0.0.1:8501**.
Stop the owner terminal with Ctrl+C when finished.

**Terminal 3 — primary React application:**

```bash
cd /Users/olveraalec/code/projects/hackathon-aws-publish
VITE_STREAMLIT_URL=http://127.0.0.1:8501 npm --prefix frontend run dev -- \
  --host 127.0.0.1 --port 5173 --strictPort
```

URL: **http://127.0.0.1:5173**. The local ignored `.env.local` was also updated only
at `VITE_STREAMLIT_URL`; the API/model settings were preserved. Restart an older
Vite process to apply environment changes. The existing `/api` proxy targets 8003.

The URL is a **configurable, same-laptop demo setting**, not a hardcoded public
deployment address. Production/default `.env.example` stays blank with a local-demo
comment. A remote deployment must set its own actual URL. The anchor opens a new
application/tab with `noopener noreferrer`; there is no iframe, data handoff or
backend CORS change. Another person's localhost is not the presenter's laptop.

## Demonstration order and labels

1. React's default three-observation fixture: identify synthetic observations and
   fixed demonstration predictions before showing the interface.
2. **Connect API / export → Run model on synthetic observations:** actual trained
   model inference via HTTP on synthetic inputs. Show source banner, selected hour,
   allocation and exact attribution; do not describe this as patient validation.
3. **Open private retrospective outputs:** select the existing sanitized 37-hour
   export locally. Its distinct retrospective banner, unavailable redacted values
   and imported provenance remain intact. No upload, browser storage or automatic
   substitution with fixtures occurs.
4. **Research Explorer:** open the separate Streamlit app using the configured link.
   Patient Monitoring and Adaptive Decisions run their disclosed simulations.
   Open Model Evaluation from the sidebar; explicitly point to its warning that
   constructed gains are **not experimental results**. Highlight synchronized traces,
   fallback shading, calibration and paired-patient boxplots as research tooling.
5. React's separate document-reported synthetic study and Streamlit's generated
   synthetic evaluation are **different studies**; neither evaluates the 37-hour
   episode or demonstrates improvement of the trained model. React's evaluation
   API is still 404, so model-specific cohort metrics stay unavailable.

## Verification and remaining boundaries

Executed:

- All three unchanged source pages via Streamlit `AppTest`: expected titles/charts,
  simulated warnings and no runtime exceptions. Model Evaluation renders its two
  dataframes, comparison plot and calibration plot.
- Real Chrome navigation from React's configured link into a **separate Streamlit
  tab**, then through all three sidebar pages: health 200, synthetic notices,
  Model Evaluation's non-results warning, no runtime exception or React iframe.
- Frontend build/typecheck/lint/unit/browser regressions; the real FastAPI and private
  retrospective-import regression remain separate checks, with no data sent to the
  Explorer. Full check counts are in the final milestone report.

No dashboard source, backend contract, model artifact, mathematical definition,
evaluation result, database schema or existing React design was edited. The original
37-hour importer and live client are preserved. The Explorer runs independently;
closing it does not replace or interrupt React/FastAPI.

There is **no blocker for the local simulated Explorer demonstration**. Real CSV
use still needs reviewed held-out data/provenance and contract validation; public
hosting/authentication is outside this local demo. No clinical validation or adaptive
superiority is claimed. No branch was merged or pushed automatically.
