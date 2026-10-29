# Independent Streamlit Research Explorer: integration audit and demo runbook

Inspected read-only on 2026-10-28: `origin/dashboard-demo` at
`ccf5b3f7637108a0710362063d4bdd5a7f8ef087` (`WIP: dashboard updates`). The ref was
fetched without merging or switching checkouts. React remains the main product;
no Streamlit charts, Python, model artifacts or backend routes were changed.

The audit reads code/configuration only. No CSV records were opened, converted or
uploaded; no training script was executed. Runtime Streamlit behavior is not
verified: its optional packages are absent from this worktree's locked environment,
and no Streamlit health endpoint was reachable on `127.0.0.1:8501`.

## Pages present in the fetched commit

| Entry point | Implemented functionality | Data and interpretation boundary |
|---|---|---|
| `dashboard/app.py` | Main **ICU Retrospective Research Dashboard**; episode selector; exploratory threshold slider (0.05–0.95); final probability plus optional static/temporal overlays; synchronized HR, MAP, respiratory-rate and temperature traces; threshold/first-crossing markers; retrospective case, lead-time and false-alarm summaries when labels exist | Missing CSV generates 40 synthetic episodes with an intentionally built-in signal and explicit simulated warnings. Label-free files suppress detection metrics/onset interpretation. Label-based event markers are retrospective heuristics, not validated clinical onset. |
| `dashboard/pages/1_Adaptive_Decisions.py` | **Adaptive AI decisions**; episode selector; tolerance slider for display classification; stable/adapting/fallback counts; synchronized static/temporal/final, temporal allocation, disagreement and entropy plots; fallback shading; explanatory expander | Missing CSV produces three hand-made, 48-hour simulated episodes. “Adapting” means absolute allocation movement exceeds the UI tolerance; it is not calibrated reliability or evidence of benefit. Fallback comes from the status field. Entropy is descriptive. |

**A separate Model Evaluation page is not present at this SHA.** The tracked
`dashboard/pages/` directory contains only `1_Adaptive_Decisions.py`. No calibration,
ROC/PR, patient-cluster paired comparison or CVaR page was found. The baseline
training CLI prints row-level AUROC/AUPRC and threshold summaries; those prints are
not a dashboard page or results for the activated LogisticRegression bundle.
Re-inspect the next pushed ref before claiming the teammate's new evaluation page
is integrated. Do not rebuild it in React to compensate for an unpublished file.

## CSV and data dependencies

Both pages read a fixed repository-relative **`data/predictions.csv`**, cached by
Streamlit. There is no CSV path variable or upload control in the inspected code.
Changes to the file require cache invalidation/app restart for a predictable demo.

| Consumer | Required columns | Optional columns/behavior |
|---|---|---|
| Main dashboard | `patient_id`, `hour`, `risk`, `HR`, `MAP`, `Resp`, `Temp` | `static_probability`, `temporal_probability` add overlays. `SepsisLabel` or `onset_hour` enable retrospective detection/lead-time displays. Without labels, only threshold crossing is described. |
| Adaptive Decisions | `patient_id`, `hour`, `risk`, `static_probability`, `temporal_probability`, `temporal_weight`, `status` | `entropy_static` is used if present; otherwise binary entropy is calculated from static probabilities. Final entropy is calculated locally. Missing temporal probabilities remain gaps. |
| `dashboard/convert_trajectory.py` | Legacy JSON wrapper containing `request.observations` and `response.predictions`; predictions include `icu_hour`, `final_probability` and controller fields | Optional labels CSV needs `hour`, `SepsisLabel` for the same episode. Raw vital nulls remain empty; no imputation. Default output is `data/predictions.csv`; `--append` reads and combines an existing CSV. |
| `dashboard/train_baseline.py` | A separate CSV/PSV/Parquet patient-hour dataset with identifiers and `SepsisLabel`; available PhysioNet measurements | Trains a separate LightGBM baseline and writes held-out predictions plus forward-filled plotting vitals. It does not supply the static/temporal/weight/status columns required by Adaptive Decisions. Do not run it for this integration or replace the verified model. |

Important compatibility gaps:

- The converter does **not** accept canonical `sepsis-decision-summary.v1` records
  or the top-level sanitized retrospective export. Passing `retrospective-01.json`
  to it would fail: that file has no `request` or `response.predictions` wrapper.
  Redacted raw vitals cannot be reconstructed to fill its required chart columns.
- A reviewed adapter/export is a future prerequisite if both applications must
  consume the same canonical decision-summary dataset. No such adapter is added
  here, and no physiological values are fabricated for compatibility.
- The main dashboard labels any existing CSV “Real predictions” based on file
  presence, without model/dataset provenance validation. Do not convert a synthetic
  episode to that path and then present the source caption as proof it is real.
  For an unmodified safe layout demo, leave the CSV absent so both pages visibly
  show their simulated mode.
- Label presence is checked by column existence. A column of missing/partial labels
  is not evidence of complete outcomes; do not use it to infer cohort performance.
- The `first positive shifted label + 6` event marker is the page's retrospective
  heuristic, not independently verified real-world onset. It must not become a
  React inference input, future forecast or treatment trigger.
- The converter merges labels on hour and does not enforce duplicate/key/episode
  alignment; the output can overwrite an existing file unless append is requested.
  Do not run it against the current private export or overwrite existing results.
- The branch's `.gitignore` covers raw/processed directories and `results/`, but
  does not specifically exclude `data/predictions.csv` or arbitrary labels CSVs.
  Keep any real/derived episode CSV private and ignored through an approved setup.
  This audit does not change that branch's files.

## Runtime prerequisites

The source imports `streamlit`, `plotly`, `pandas` and `numpy`. Streamlit/Plotly are
not declared in the inspected `pyproject.toml` or `uv.lock`, and no dashboard-specific
requirements file is committed. LightGBM is also undeclared and absent, but it is
needed only for the separate training script—not to display simulated dashboards.
The training script's docstring also names a different entry-point path than its
actual `dashboard/train_baseline.py` location; do not use that as a launch recipe.

The teammate should supply their tested/pinned Streamlit and Plotly environment
and a startup check before claiming reproducibility from a clean checkout. Do not
install optional dashboard packages into or retrain within the checksum-sensitive
verified model environment. Use an independent explorer environment/checkout.

## Linking from React

The existing `VITE_STREAMLIT_URL` link is sufficient for either page and future
Streamlit pages. It is ordinary top-level navigation to a separate tab, with
`noopener noreferrer`, not an iframe or cross-origin API call. No backend, CORS or
React chart changes are needed. Link the application's actual root URL and use its
own sidebar; no unverified deep-link pathname is guessed.

Once the teammate has a running, locally reachable explorer, set only this key in
the ignored `frontend/.env.local`, preserving the other API/model settings:

```dotenv
VITE_STREAMLIT_URL=http://127.0.0.1:8501
```

That is a **proposed same-laptop URL**, not a verified running service. Use the actual
URL if the teammate runs elsewhere. Restart Vite after changing it. On another
computer, `127.0.0.1` refers to that computer, not the presenter's laptop; remote
access/hosting requires its own approved setup. Never put credentials in a public
Vite variable. Until the actual URL is known/reachable, React correctly displays
“Explorer link not configured” rather than opening an invented service.

## Reproducible demonstration sequence

1. **Backend owner terminal:** keep the already verified FastAPI process running on
   `127.0.0.1:8003` with its unchanged model/configuration. Confirm `/health` is 200.
   Do not move the bundle or alter its environment for the explorer.
2. **React terminal, this worktree:** run `npm --prefix frontend run dev`. Open the
   loopback URL printed by Vite. The initial three-hour fixture is explicitly
   synthetic, including its demonstration predictions.
3. **Live model on synthetic inputs:** choose **Connect API / export → Run model on
   synthetic observations**. Explain that the 24-hour inputs are synthetic and the
   outputs are from the real activated model. Show the Network POST returning 200,
   select hours, and inspect allocation, exact change terms, missingness and entropy.
4. **Real retrospective outputs, local/private:** choose **Open private retrospective
   outputs** and select the existing sanitized `retrospective-01.json` from the main
   checkout's ignored results directory. Show the distinct retrospective banner,
   37 decisions, previous/current outputs and exact decomposition. No file upload
   occurs. Explain that raw vitals and log-odds vectors were redacted; do not restore,
   publish, screen-capture or commit them. This is one retrospective episode, not a
   cohort evaluation or clinical validation result.
5. **Separate synthetic evaluation:** switch to Research perspective. The displayed
   document-reported synthetic policy study is distinct from the imported episode
   and the installed model. The current evaluation-summary API is 404; calibration
   and real-cohort paired results remain unavailable.
6. **Explorer owner terminal:** in a separate checkout with the teammate's tested
   dashboard environment and approved data arrangement, run:

   ```bash
   python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
   ```

   For the no-CSV layout demo, both Streamlit pages disclose simulated data.
   Verify `http://127.0.0.1:8501/_stcore/health`, then configure the actual URL in
   React and restart Vite. Do not launch a new training run or copy the private
   retrospective JSON into a static website/CSV directory for this step.
7. **Research Explorer link:** open the separate Streamlit app. Highlight the
   synchronized risk/vital views and the threshold trade-off as retrospective
   exploration, then Adaptive Decisions' allocation/disagreement/entropy traces
   and fallback shading. State whether the page uses simulated or approved loaded
   data before discussing any number. Keep React as the primary product.

Useful independent charts to highlight without rebuilding:

- Main page: synchronized final/static/temporal probability with HR/MAP/Resp/Temp;
  exploratory threshold crossings. Label-dependent counts are not prospective
  detection guarantees or validated treatment thresholds.
- Adaptive page: four aligned probability/allocation/disagreement/entropy plots,
  fallback shading and the user-controlled display tolerance. Synthetic allocation
  patterns can exceed the configured backend's ordinary bounds; they illustrate
  layout rather than reproduce the verified controller.
- The promised Model Evaluation page should be highlighted only after its commit,
  data contract and source/provenance labels have been inspected. It is not present
  in the fetched WIP commit.

## Current integration status

React retrospective import and real FastAPI/browser checks pass. The existing
Streamlit link implementation needs no redesign. Remaining explorer prerequisites
are the pushed evaluation page (if desired), tested dependency pins/environment,
actual running URL, and a reviewed CSV/adapter/provenance arrangement if real data
are used. Streamlit was source-inspected, **not runtime-tested**, in this milestone.
No branch was merged, no backend/model/SQL/AWS change was made, and no Streamlit
training or conversion job was executed.
