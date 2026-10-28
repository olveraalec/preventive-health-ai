# Transfer checklist and runbook

## Source-of-truth and scope
This package is a **snapshot** of the local repository made by the packaging script. It is not guaranteed to be fully portable until checks below pass. The original project is never modified by this script. No dataset, trained model, Jupyter notebook, secret, or virtual environment is transferred automatically.

## Before use
- [ ] Inspect `PACKAGE_MANIFEST.json` for missing/oversized files and warnings.
- [ ] Review `compose.yaml`, `Dockerfile`, `sql/`, `src/`, and `scripts/` for passwords, hostnames, cloud IDs, paths, and any sensitive strings; the scanner is heuristic, not a security guarantee.
- [ ] Copy *only* necessary configuration values into a new local `.env`. Do not commit `.env`.
- [ ] Verify `uv.lock` corresponds to `pyproject.toml` and platform target.
- [ ] Determine which components require running PostgreSQL or external services.
- [ ] Check `src` imports, build context, and tests with **actual** local dependencies.

## Setup (from starter directory)
```bash
uv sync --locked
PYTHONPATH=src uv run pytest tests/ -q
```

## Docker (if relevant files are present)
```bash
docker compose config
docker compose up --build -d
```
Only start containers *after* checking copied configuration. A successful Docker build does not establish the live agent's utility.

## Experiment results
The `results/` directory includes selected existing CSVs **if available** in the source repo. They are not regenerated during packaging. Preserve seeds, task split definitions, source dataset versions, and original experiment scripts to reproduce them.

## Minimum hackathon adaptation
- Define input and outputs, data leakage boundary and domain-specific safety requirements.
- Implement real downstream action and outcome verification.
- Keep static/top-score baselines and covariance ablation.
- Do not treat synthetic performance as evidence for the new dataset.
- Time permitting, test unseen subjects/episodes with delay/missingness/shift.

## Optional production steps (not shipped as completed)
CI, secret management, rate limits, service-auth, migrations, telemetry/observability, reproducible Docker multi-architecture build and any AWS deployment require independent review and tests. Avoid asserting these were completed merely because source files exist.
