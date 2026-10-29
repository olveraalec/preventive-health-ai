# Preventive Health Intelligence

**Interpretable, time-aware sepsis-risk research with evidence-aware model allocation**

> **Research prototype — not a medical device.** This application estimates the PhysioNet 2019 Sepsis Challenge's **shifted sepsis label** from historical ICU observations. Its scores are **not** clinically validated diagnoses, calibrated probabilities of sepsis in the next six hours, or autonomous clinical recommendations. Synthetic examples, retrospective replay, and cohort evaluation are explicitly different forms of evidence.

## Why we built this

Clinical observations arrive irregularly, are often missing, and do not always improve prediction when added to an algorithm. More complex models can also be less reliable when patient context changes. We built a system that asks three questions:

1. **What does the currently available information predict?**
2. **Why did that prediction change?**
3. **Is there enough evidence to justify adapting the inference strategy?**

Our hackathon prototype makes these questions inspectable through a live model-backed web experience, retrospective inference replay, and a separate research evaluation interface. It combines a time-aware modeling pipeline with an evidence-aware controller and an additive, auditable decision record.

## Demo experience

| Surface | What it demonstrates | Evidence status |
| --- | --- | --- |
| **React: Patient Monitoring** | Hourly model scores, static/temporal experts, allocation history, missingness and measurement freshness | Interactive simulation and model-backed inference; source clearly labeled |
| **React: Adaptive Intelligence** | Expert disagreement, controller allocation, uncertainty, fallback and exact prediction-change decomposition | Model-backed for connected inference; mathematical identity verified |
| **React: Retrospective Import** | Browser-local import of a sanitized 37-hour historical inference export | Genuine retrospective model outputs; physiological inputs omitted from export |
| **React: Model Evaluation** | Exploratory policy comparison and study interface | **Synthetic study unless separately verified real evaluation is loaded** |
| **Streamlit: Research Explorer** | Patient dashboards, adaptive decisions, calibration and patient-bootstrap comparisons | Data-source dependent; synthetic fallback when CSVs are absent |

**Source clarity:** A synthetic observation submitted to our actual trained model produces a *real model prediction from a synthetic input*, not a clinical study result. A retrospective replay demonstrates reproducibility, not prospective clinical efficacy. A synthetic policy-comparison chart is never represented as an empirical improvement.

## Architecture

```text
Historical PhysioNet 2019 hourly observations
                  │
         causal feature building
                  │
       ┌──────────┴──────────┐
       │                     │
 Static logistic expert   Temporal logistic expert
       │                     │
       └──────────┬──────────┘
                  │
       evidence/reliability-aware
          bounded allocation
                  │
       final probability + metadata
                  │
       FastAPI decision-summary API
                  │
       ┌──────────┴───────────┐
       │                      │
 React + Vite product     Streamlit research explorer
       │                      │
 Browser-local replay     local CSV-based evaluation
```

The Python project lives under `src/healthcare_ai_lab/`. Its components include preprocessing, static/temporal model loading and inference, controller research, decision contracts, trajectories, evaluation, an agent-workflow research scaffold, and database utilities. React is under `frontend/`. The teammate's Streamlit app is under `dashboard/` once its reviewed branch is integrated.

## Mathematical core

### Two predictive experts and bounded allocation

For patient episode $i$ at observation hour $t$:

- $p_{S,it}$: prediction from the **static** logistic model;
- $p_{T,it}$: prediction from the **temporal** logistic model;
- $q_{it}$: bounded allocation to the temporal expert.

The displayed blended score is

$$
p_{it} = (1-q_{it})p_{S,it}+q_{it}p_{T,it}
       = p_{S,it}+q_{it}(p_{T,it}-p_{S,it}).
$$

The controller may **abstain from changing the allocation** and use a preselected static/constant fallback when its reliability evidence is insufficient. Experimental covariance-based and anticipatory controllers are evaluated separately; they must not be conflated with the policy used for a particular live inference record. See [Research methodology](docs/RESEARCH_METHODOLOGY.md).

### Exact decision-change attribution

Writing $d_t=p_{T,t}-p_{S,t}$, the change between successive observations decomposes exactly as

$$
\underbrace{p_t-p_{t-1}}_{\text{total change}} =
\underbrace{(p_{S,t}-p_{S,t-1})}_{\text{static movement}}+
\underbrace{q_{t-1}(d_t-d_{t-1})}_{\text{expert-disagreement movement}}+
\underbrace{(q_t-q_{t-1})d_t}_{\text{controller reallocation}}.
$$

Unlike generic feature-importance labels, this is an **algebraic accounting identity** for the prediction update. It does **not** claim that a measurement *caused* a clinical outcome. Optional logistic log-odds contributions are similarly model explanations, not causal effects.

Predictive Bernoulli entropy, expert disagreement, observation freshness, fallback reasons, and model/artifact provenance accompany supported API responses.

### Loss and evaluation principles

The principal probabilistic classification loss is Bernoulli log loss:

$$
\ell(y,p)=-y\log(p)-(1-y)\log(1-p).
$$

Patient-level evaluation first aggregates hourly loss **within each patient**, then compares policies over patients. The evaluation framework includes log loss, Brier score, discrimination (AUROC and average precision), calibration/error bins, prevalence, patient-level paired deltas and bootstrap intervals, tail/CVaR summaries, and controller behavior/abstention. A separate general-purpose metric registry also retains regression metrics (RMSE, MAE, bias, Gaussian NLL, CRPS and interval coverage). Not every available metric is displayed or meaningful for every task.

See [Research methodology](docs/RESEARCH_METHODOLOGY.md) for equations, split policy and interpretation limits.

## Data and evaluation integrity

- **Dataset:** [PhysioNet / Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) hourly ICU observations, approximately 40,336 patient records in the public challenge training cohorts.
- **Target:** `SepsisLabel` is shifted relative to nominal onset by the Challenge definition (positive labels begin approximately six hours before the nominal onset). The trained score targets that **challenge label**, not an independently validated six-hour onset probability.
- **Temporal information:** Feature engineering is prefix/causality aware: no future observation should affect an earlier prediction.
- **Grouping:** Patient-grouped training/evaluation and out-of-fold procedures are essential to avoiding within-patient leakage. Different training, validation and test sets are not interchangeable.
- **Real-evaluation exports:** CSV files must include labels, split provenance, policy definitions, cohort counts and validation before any dashboard reports empirical model comparisons.
- **Research limitations:** Exploratory validation did not establish that continuous adaptation consistently outperforms a conservative policy; performance degraded under a constructed reversed-context shift. The controller's abstention/fallback behavior is therefore a research result in its own right.

### Verified local release-candidate results

| Verification | Observed result | What it establishes |
| --- | --- | --- |
| Retrospective patient replay | **37 hourly observations** | Feature and prediction reproducibility for one historical episode |
| Inference pathways | Raw-hourly, engineered, decision-summary predictions/weights matched | Consistency across supported adapters |
| Decision decomposition | Max residual **4.12 × 10⁻¹⁸** | Numerical fidelity of the exact attribution identity |
| Expert log-odds explanations | **74 / 74** available | Explanation export for the checked replay |
| Backend workspace audit | **314 passed; 15 PostgreSQL tests skipped** | Local test suite result; not a deployment or clinical validation certificate |
| React connected test | **24 synthetic observation hours**, live HTTP 200 | Actual trained model used on synthetic inputs |
| React retrospective import | **37 records**, browser-tested | Privacy-conscious local viewing of exported outputs |

These numbers are from the team's October 28, 2026 local integration reports. The underlying run records and source commits are the authoritative reproducibility artifacts. Additional optional-file/DB checks were skipped in some environments; hosted CI and live database tests are separate verification steps.

## Running the local demo

**Prerequisites:** Python compatible with the verified model manifest, `uv`, Node.js/npm, and access to the locally held **trusted** model bundle and checksum-bound manifest. The model bundle and raw patient datasets are not shipped with the public source repository.

In three separate terminals, after integrating the reviewed frontend and dashboard branches:

**1. FastAPI (port 8003)**

```bash
cd /path/to/preventive-health-ai
PYTHON_DOTENV_DISABLED=1 \
SEPSIS_MODEL_BUNDLE=artifacts/models/prototype_bundle.joblib \
uv run --no-sync uvicorn healthcare_ai_lab.api.main:app \
  --host 127.0.0.1 --port 8003 --no-access-log
```

Check `http://127.0.0.1:8003/openapi.json`; the supported interpretation route is `POST /api/v1/research/sepsis/trajectory/decision-summary` with an optional `?include_contributions=true`. Its request body follows `RawTrajectoryRequest` (`dataset`, `episode_id`, `model`, `observations`). See `docs/FRONTEND_BACKEND_HANDOFF_RC1.md` for the full API contract.

**2. React (Vite development server)**

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

Open the local URL printed by Vite (typically `http://127.0.0.1:5173`). Configure the development `/api` proxy to target FastAPI on port `8003` per `frontend/README.md`. Select **Connect API → Run model on synthetic observations** for the live, clearly labeled model-backed path. To inspect historical results, choose the private retrospective-import flow and select the sanitized JSON from your local machine; it is processed in browser memory.

**3. Streamlit (separate research interface, optional)**

```bash
python -m streamlit run dashboard/app.py --server.port 8501
```

Open `http://127.0.0.1:8501`. Set the React research explorer URL in the frontend environment per `frontend/.env.example`. **Localhost links are for local demos only.** The dashboard expects local `data/predictions.csv` for monitoring and `data/eval_predictions.csv` for policy comparison, and may show simulated examples when either file is absent. Confirm the current dependency file and Streamlit setup in the merged repository before running.

**Verified retrospective export** (sensitive derived data; local only):

```bash
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python \
  -m healthcare_ai_lab.experiments.export_decisions \
  --source retrospective \
  --output results/dashboard/retrospective-01.json
```

This exporter omits raw source identifiers, physiological measurements and contribution vectors, refuses overwrites and keeps the output Git-ignored. Do **not** publish patient-derived trajectories or upload them to public hosted demos without appropriate authorization and review.

## Source layout

```text
frontend/                  React / TypeScript / Vite application, tests and UI docs
dashboard/                 Streamlit clinical/research explorer (after integration)
src/healthcare_ai_lab/     Python package: preprocessing, models, controller,
                          trajectories, evaluation, API and supporting services
tests/                     Backend and integration tests
docs/                      API contracts, examples, engineering docs
markdowns/                 Extended research notes / architecture decisions
.github/workflows/         CI and ECR publishing workflows
artifacts/models/          Local verified model artifacts (not distributed)
data/, results/            Local, ignored datasets and generated outputs
```

The source layout is a cross-branch integration target. If reading an earlier commit, a directory may be present only in its dedicated branch.

## Engineering and deployment

The project uses automated tests and linting, Docker builds and an AWS ECR publishing workflow. A private ECR repository and GitHub OIDC publishing configuration have been prepared. **Building an image or configuring ECR is not the same as deploying a publicly accessible product**: hosted image publication, model delivery, service authentication, rate/body limits, HTTPS ingress, readiness monitoring and rollback need to be verified before production exposure. The existing research endpoint must not be advertised as a secured clinical API.

The large container image and Python-version-to-artifact compatibility are known deployment considerations. PostgreSQL integration has tests, but do not imply that every local demonstration uses a persistent database.

## Scientific and clinical limitations

- Not a medical device, clinical warning system, or autonomous diagnostic/treatment service.
- No prospective evaluation, independent external clinical validation, prospective calibration guarantee, or confirmed clinical benefit.
- No deployed/validated next-hour forecasting or clinician-review recommendations; hybrid serving beyond the existing mixture is experimental.
- Sample-level attribution is algebraic/model-based, not a causal clinical explanation.
- Continuous-control improvements were **not** established as uniformly robust; distribution shift remains a central open problem.
- A missing evaluation CSV means the Streamlit evaluation page runs on **synthetic study data**, not validated trained-model results.
- A synthetic observation passed through the fitted model is still a **synthetic input**.

## Research directions

Prospective and external validation; robust conditional reliability estimation; error-covariance and residual-dependence analysis; active information selection and measurement-cost policies; calibrated forecasting; alert burden and lead-time research; patient-subgroup and shift evaluation; clinician-facing usability studies; and secure, auditable deployment.

## Hackathon and reproducibility

Built for the **SOLIX AI Hackathon 2026** at UC San Diego. Earlier research/modeling formed part of the project's foundation; the hackathon deliverable focused on connecting mathematical decision summaries, verified inference, React monitoring, and collaborative research visualization into an end-to-end prototype.

See the [research methodology](docs/RESEARCH_METHODOLOGY.md), [presentation outline](docs/PRESENTATION_OUTLINE.md), and [repository integration/cleanup plan](docs/REPOSITORY_CLEANUP_PLAN.md).
