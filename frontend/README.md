# Preventive — React ↔ FastAPI research product

React is the primary product interface. Streamlit remains an independent research
explorer, linked explicitly rather than embedded. One responsive workspace provides
Patient Monitoring, Adaptive Intelligence, Model Evaluation and Research Methodology.
Monitoring/research perspective buttons share the same selected episode and decision;
they do not duplicate inference or maintain independent patient histories.

## Preview the complete journey

Use Node 24+ with native TypeScript stripping (validated on Node 26.12.1). From the
repository root:

```bash
npm --prefix frontend ci
# Preserve any existing local configuration:
test -f frontend/.env.local || cp frontend/.env.example frontend/.env.local
npm --prefix frontend run dev
```

Open the loopback URL Vite prints, normally `http://127.0.0.1:5173`.
The initial view uses the **authoritative synthetic example**, three observations
at ICU hours 1, 2 and 4. It includes observed values, missingness, last-observed ages
and demonstration probabilities. Those fixed synthetic predictors are not the
trained Sepsis model and do not establish performance on the PhysioNet cohort.

To use the trained model: select **Connect API**, enter the installed manifest's
public artifact identifier/model version, and select **Run model on synthetic
observations**. This submits the teammate's reused 24-hour synthetic input sequence.
The returned risk, weights, interpretation, entropy and freshness come from the
unchanged backend. The source banner explicitly says **Synthetic inputs · real
API outputs**. No patient record, label or model binary is copied into React.
An ignored `.env.local` was configured locally with the owner's approved public
identifiers when no existing file was present, so this demonstration can be run
without retyping them. It contains no credentials and is not committed.

Select any observed hour to update the chart, previous/current decision, allocation,
measurement trends and evidence. Research perspective reveals the reused controller
arithmetic/reliability panels and a separately labeled recorded synthetic study.
It does not present that study as evaluation of the installed model.

## Configuration

```dotenv
VITE_API_BASE_URL=
API_PROXY_TARGET=http://127.0.0.1:8003
VITE_STREAMLIT_URL=
VITE_DEMO_MODEL_ARTIFACT_ID=
VITE_DEMO_MODEL_VERSION=
```

Restart Vite after changing `frontend/.env.local`.

- Empty `VITE_API_BASE_URL` uses same-origin `/api/v1/...`. The development proxy
  routes `/api` to configurable `API_PROXY_TARGET`, defaulting to the owner's current
  backend on port 8003. No Python CORS or API changes are needed.
- An explicit browser base such as `http://127.0.0.1:8003` requires an already
  approved CORS arrangement. It precedes `/api/v1`; do not append `/api` or include
  credentials, a query string or a fragment.
- Set `VITE_STREAMLIT_URL` to the teammate's actual HTTP(S) address. It opens a
  separate tab with `noopener noreferrer`; there is no iframe. The configured local
  demo uses `http://127.0.0.1:8501`; blank deployments remain visibly unconfigured.
- The optional public model identifiers populate the synthetic-input form. They
  identify the installed model; they are not model artifacts or authentication.
  All `VITE_` values are public build inputs; no credentials belong here.
- Production proxy/authentication/hosting remains a separate reviewed integration.
  `npm run preview` serves the bundle at loopback, usually port 4173, but does not
  apply Vite's development API proxy. The offline fixture works independently.

## Actual API contract and data boundary

The canonical numerical authority is:

```text
POST /api/v1/research/sepsis/trajectory/decision-summary
POST /api/v1/research/sepsis/trajectory/decision-summary?include_contributions=true
```

The UI calls the default compact response. The typed client also supports the
optional contribution query. Input is `RawTrajectoryRequest`; wire schema is
**sepsis-decision-summary.v1**, guided by architecture v0.6. It has `records`, shared
capabilities/provenance and server-computed change terms. It differs from the
previous aws-publish prototype's `decision_summaries` envelope.

The owner-authorized, read-only references were `docs/DECISION_SUMMARIES.md`,
`docs/examples/sepsis_decision_summary.synthetic.json` and
`docs/examples/sepsis_decision_summary.schema.json` in the separate main checkout.
The two data-free synthetic/schema assets are bundled under `src/data/`. No Python,
artifact, patient export or source data was read/copied from that checkout.

`src/api/canonical.ts` validates the supplied JSON Schema and normalizes field names
for the preserved React panels. It copies probabilities, allocations, decomposition
terms and residuals directly; it does not reconstruct an unavailable expert/change
term or run inference. Only weighted expert contribution displays use the elementary
reported-weight × reported-probability arithmetic. Earlier frontend fixtures remain
available for regression compatibility, but the authoritative three-hour fixture is
the initial demonstration.

The client also checks episode/model/history correspondence, increasing observed
hours and previous-hour linkage. Calls time out after 20 seconds; synthetic reset
aborts an outstanding request. Errors never echo backend response bodies. A failed
inference retains the clearly labeled previous view. It does not silently swap real
outputs for fixture probabilities. Requests/responses exist only in React memory;
there is no browser persistence, patient-data logging or model loading.

Monitoring displays current versus last-known values separately, source units,
missing measurements and age. The reused information-quality component and new
trend cards use only the observation prefix through the selected hour. No fill,
clinical expiry threshold, diagnosis, treatment trigger or outcome label is inferred.
Entropy is a distribution descriptor, not calibrated reliability. Next-hour forecast,
learned hybrid and clinician-review policy remain explicitly unavailable.

The optional evaluation call is:

```text
GET /api/v1/research/evaluation-summary
```

**The current port-8003 server returns 404 for this route.** React handles that state
and identifies its bundled, document-reported synthetic conditional-allocation study.
The installed model's cohort evaluation remains unavailable. Calibration, AUROC,
AUPRC, Brier, real-cohort paired intervals and alert burden are never invented or
computed with live labels. The synthetic study retains source checksum, aggregation,
patient bootstrap interval and the adverse positive-patient result.

## Reused remote work — no branch merges

Both requested refs were fetched and inspected without changing checkouts:

- `origin/frontend/trajectory-ui`, commit **e08593d**: reused `Timeline`,
  `InfoQuality`, `ControllerPanel`, `Reliability` from `components/Panels.tsx`.
  `RiskTimeline` now wraps that existing timeline instead of maintaining a competing
  graph. A type bridge adapts it to v0.6. Styles are scoped; the fallback wording was
  corrected to include temporal-model **or gain-controller** failure.
- Reused the raw synthetic observation generator. The older formula-generated
  predictions are deliberately not displayed as current backend outputs.
- Kept the newer typed/error-checked client instead of replacing it with the older
  fetch-only implementation. The App shell and v0.6 interpretation remain intact.
- `origin/dashboard-demo`, commit **c28e2ad**: all three Streamlit pages run in a
  separate pinned worktree/environment. Their retrospective evaluation and chart
  implementations remain independent; no label-dependent calculations moved into
  React inference. See the Explorer runbook for verified contracts and metric caveats.

The pre-inspection React work was preserved in an independent archive and tracked
patch under `/private/tmp/preventive-react-checkpoint-qo0nsfdn`. No branch was merged,
no teammate checkout was modified, and no existing work was blindly replaced.

## Files and components

- `src/App.tsx`, `App.css`, `index.css`, `index.html`, `public/preventive.svg`: product
  shell, view modes, navigation, external explorer and accessible responsive styling.
- `src/components/PatientMonitoring.tsx`, `RiskTimeline.tsx`, `Panels.tsx`:
  shared timeline, observation selection and reused dashboard components.
- `src/components/DecisionInterpretation.tsx`: probabilities, exact signed change
  attribution, allocation/fallback, entropy, freshness and unavailable capabilities.
- `src/components/ObservationEvidence.tsx`, `ResearchAudit.tsx`, `ModelEvaluation.tsx`:
  measurement trends/units, descriptive review evidence, research audit and evaluation.
- `src/components/Icon.tsx`: local SVG icon set, no remote assets/fonts.
- `src/api/{client,contracts,canonical}.ts`: typed, validated HTTP and display adapter.
- `src/hooks/useResearchWorkspace.ts`: loading/error/cancellation and data origin.
- `src/data/`: canonical synthetic/schema assets and recorded/legacy fixtures.
- `src/fixtures/synthetic.ts`, `src/types/trajectory.ts`: remote-input reuse/type bridge.
- `src/lib/{format,observations}.ts`: formatting and prefix-only measurement evidence.
- `.env.example`, `vite.config.ts`, `tsconfig.app.json`, `package.json`, `.gitignore`:
  public configuration, proxy, type/test commands and ignored outputs.
- `tests/`: deterministic contract/client/evidence, browser and real-API checks.

## Verification and successful response

```bash
npm --prefix frontend run build
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend test
npm --prefix frontend run test:browser
```

Browser checks use installed Chrome/Chromium with a new temporary profile and their
own loopback Vite server on free port 4175. Set `CHROME_BIN` if needed. Default browser
tests intercept all API requests with fixtures, test desktop/mobile overflow, hour
selection, research-mode metric selection, error/retry/reset and no browser storage
or iframe. Screenshots are ignored under `frontend/test-results/`.

Final checks passed: production build, TypeScript build-mode checking, Oxlint
without warnings, **23 unit tests**, fixture browser regressions and the real API
HTTP/browser regressions. Desktop (1440px) and mobile (390px) screenshots were
visually reviewed with no horizontal overflow. The unchanged repository Python
suite also passed **168 tests, 17 skips**: 15 opt-in DB cases and two optional local
source/schema checks. No persistence changes required live database testing.

Real HTTP and real browser integration (no mocked inference):

```bash
export DEMO_MODEL_ARTIFACT_ID=notebook09-prototype
export DEMO_MODEL_VERSION=sha256:16ac3e6deb025a82de6806ecefe5c1af4b31d2742563e600385113d4caf5dde2
npm --prefix frontend run test:api
REAL_API=1 npm --prefix frontend run test:browser
```

These public identifiers were supplied by the owner for the activated, checksum-bound
bundle. Update them only when an approved installed model version changes. The
service owns all artifact loading/verification; the tests read no model files.

Executed real result on 2026-10-28: two canonical POSTs returned **HTTP 200**, covering
24 synthetic hours and an independent five-hour prefix. Both responses passed the
bundled schema; prefix records matched exactly. Per-vital value/age parity also passed. Maximum decomposition residual was
`3.686287386450715e-18`. Artifact integrity was `checksum_bound_manifest`.

The selected ICU-hour-24 response included:

```json
{
  "static_probability": 0.03316692548290534,
  "temporal_probability": 0.026243079123216256,
  "final_probability": 0.03248301485067044,
  "temporal_weight": 0.09877611326222642,
  "predictive_entropy_nats": 0.14327015921059238,
  "emergency_fallback": false
}
```

The separate real Chrome test exercised **React → Vite `/api` proxy → FastAPI**,
received HTTP 200, validated the canonical response and displayed 24-hour predictions,
allocations and explanations. Its optional evaluation GET returned 404 and was handled
explicitly. These are engineering integration checks on synthetic observations, not
validation on real patients or proof of adaptive benefit. No DB or AWS service was
changed. The separate Explorer browser checks also contact the local Streamlit app.

The backend briefly became unavailable (`ECONNREFUSED`) during validation, then
returned before final checks. The final real HTTP rerun passed again, including
exact parity of current/last-observed vital values and measurement ages at all 24
hours. The service must remain running for subsequent live demonstrations; the
explicitly labeled fixture remains usable independently.

## Safe review and integration

Work remains on isolated `feature/react-product-integration`. Backend checkpoint
**8652c21** follows AWS checkpoint **e3b06cd** on `feature/aws-publish`; neither was
pushed or merged automatically. Every subsequent changed file is under `frontend/`.

1. Review the AWS/backend commits through the normal PR process targeting
   `feature/codex-integration`. Keep ECR publishing unexecuted until reviewed/merged.
   The authoritative port-8003 backend is independently maintained by the teammate.
2. Review `git diff 8652c21 -- frontend` and new files, run the checks, then create a
   **frontend-only commit**. Do not stage Python, artifacts or other worktrees.
3. After the reviewed backend work is integrated, open a frontend PR against the
   updated integration branch. If backend PRs were squash-merged, cherry-pick only
   the frontend commit onto a fresh branch based on the updated integration ref.
4. Resolve the known overlap with `origin/frontend/trajectory-ui` in `App.tsx`,
   `App.css` and `api/client.ts` by reviewing/composing the changes. This candidate
   already reuses its panels and raw-input generator; do not replace the frontend
   directory wholesale or use broad `--ours`/`--theirs` conflict resolutions.
5. Preserve all teammate Streamlit/Python files and the pinned Explorer environment.
   Use the configurable local URL and tested external link, preserving the backend's
   canonical API schema. No automatic merge or dashboard replacement is needed.
6. Re-run frontend checks and real synthetic-input API integration on the composed
   candidate. Preserve the existing lockfile; no dependency versions changed here.

Remaining prerequisites: available backend for live inference; installed public model
identifiers; a separately reviewed aggregate evaluation endpoint
with real model/split provenance before displaying model/cohort calibration or paired
results. Forecasting, learned hybrid serving and clinician-review validation remain
outside this integration. Nothing here authorizes deployment or clinical use.

## Private retrospective export integration

Select **Connect API / export → Open private retrospective outputs**, then choose
`retrospective-01.json` from the backend's ignored `results/dashboard/` directory.
Do not move it into this repository, `frontend/public`, or the build assets.

The file picker reads the selected file locally into React memory. It sends no API
request and stores no browser data. It accepts only `sepsis-decision-summary.v1`
with `payload_kind=sanitized_retrospective_demo`, declared checksum-bound artifact
metadata, and **null measurements and log-odds contribution vectors for every
record**. Other payload kinds, malformed histories, unredacted fields, empty files
and files over 5 MiB are rejected. The previous view remains visible on failure.
File authenticity and historical training are not independently verified in React;
provenance is displayed as declared by the chosen file.

Successful import updates the same timeline, previous/current probabilities,
controller weights, change terms, entropy, disagreement and fallback panels. The
banner says **Retrospective model outputs · local export**, not synthetic input or
a live API stream. Aggregate missingness/completeness is preserved. Raw values,
per-vital ages and logistic contribution vectors remain explicitly unavailable
when redacted. Earlier availability of explanations during backend verification
does not make the removed vectors available in this sanitized file.

Derived trajectories remain sensitive. There is no automatic filesystem loading,
export upload, browser persistence or automatic switch to synthetic values. The
separate synthetic evaluation study is labeled as unrelated to this episode/model;
a decision export is not a cohort evaluation report. Use **Use synthetic example**
only when explicitly switching back to demonstration data.

New implementation files are `src/api/retrospective.ts` (file/schema/redaction
checks), `src/lib/source.ts` (source labels and retained-data messages), and
`tests/retrospective.test.ts`. Existing workspace/panels are extended additively.

An optional local-file browser regression uses:

```bash
RETROSPECTIVE_EXPORT_PATH=/absolute/path/to/retrospective-01.json \
  npm --prefix frontend run test:browser
```

Set `RETROSPECTIVE_EXPECTED_RECORDS` for an export other than 37 records. To also
exercise the real API, supply the existing model-identifier variables and
`REAL_API=1`. Without that flag, all test API calls are synthetic mocks; the selected
local export is still loaded through the actual browser file picker. Import checks
save no private record values or screenshots, assert no additional API request or
browser storage, exercise first-observation selection and rejection of an invalid
replacement file, then explicitly reset to the synthetic view.

The owner-authorized export's aggregate structure was inspected read-only: 37 ICU
records, `checksum_bound_manifest`, all measurement/contribution vectors redacted,
maximum reported change residual `4.119968255444917e-18`. It was not copied into this
worktree. Source-level model/feature parity and 74 pre-redaction explanations remain
backend verification evidence, not newly reproduced training or clinical results.

For the current Streamlit page inventory, CSV/contract compatibility, dependency
prerequisites, safe link configuration and the multi-terminal hackathon sequence,
see [STREAMLIT_RESEARCH_EXPLORER.md](STREAMLIT_RESEARCH_EXPLORER.md). That audit pins
the inspected remote commit and distinguishes source inspection from runtime checks.

Retrospective milestone verification: build, typecheck, warning-free lint and
23 frontend tests passed; the full unchanged Python suite passed 168 tests with
17 skips (15 opt-in DB cases and two optional local files). Fixture browser checks
and the actual 37-record file-picker import passed, including failed-import retention,
no extra API request and no browser persistence. The combined real FastAPI/browser
run also passed after the owner restarted port 8003: canonical POST HTTP 200,
24 synthetic input hours, prefix/measurement parity, followed by the private local
37-hour import. No retrospective record values or screenshots were saved.
The React evaluation-summary route remains 404. Streamlit's independent Model
Evaluation page is now present at c28e2ad and verified in simulated mode. The local
Explorer URL is explicitly configured; no public URL is hardcoded into React.

## Final Research Explorer milestone (c28e2ad)

The complete three-page Streamlit application is now pinned and running in the
separate `hackathon-streamlit-demo` worktree with a hashed Explorer-only environment.
The local configurable link targets `http://127.0.0.1:8501`; it is not a public
deployment default. Run `npm --prefix frontend run explorer` for Streamlit, and
`npm --prefix frontend run dev` for React. The exact three-service commands, both
CSV contracts, source/metric caveats, setup pins and verification are in
[STREAMLIT_RESEARCH_EXPLORER.md](STREAMLIT_RESEARCH_EXPLORER.md). No existing React
design, importer or backend implementation changed in this step.

Additional checks:

```bash
../hackathon-streamlit-demo/.venv/bin/python frontend/tests/explorer_smoke.py ../hackathon-streamlit-demo
VERIFY_EXPLORER=1 npm --prefix frontend run test:browser
```

The first executes all unchanged pages in explicitly simulated mode. The second
clicks the actual configured link and visits all three pages in a separate browser
tab. Streamlit has no real CSV inputs in this demo, so calibration/bootstrap/policy
results are labeled simulated and are not evidence of trained-model improvement.
