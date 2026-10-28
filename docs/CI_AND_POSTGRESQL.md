# CI validation and PostgreSQL readiness

## Automated validation

`.github/workflows/ci.yml` runs on every push and pull request, with read-only
repository permissions. The single `validation` job uses Ubuntu, Python 3.12 and
uv 0.11.26, installs the project/dev dependencies using `uv sync --locked --group dev`,
then runs:

```bash
uv run --no-sync pytest -q -ra --junitxml=pytest-results.xml
uv run --no-sync ruff check --config ruff-ci.toml .
```

`RUN_DB_TESTS=0` explicitly disables live PostgreSQL tests; all other tests are
collected normally. `PYTHON_DOTENV_DISABLED=1` prevents loading local `.env` values.
There is no database service, model/data download step, clinical inference job,
deployment, test-failure suppression or `continue-on-error`. Failed tests fail the
job. Ruff still runs after a test failure if dependency installation succeeded.

Pytest's `-ra` lists skip reasons in the logs. An always-running summary step reads
JUnit XML and reports passed/failed/error/skipped counts and each skip reason. It
does not turn a failed test step into a success. If setup fails before a report is
produced, the summary says so instead of reporting a successful empty suite.

The local checkout currently has 136 passing non-database tests and 15 intentional
database skips when live integration is disabled. With the disposable database
enabled, all 151 tests passed. See the dated
[verification and security audit](API_SECURITY_READINESS_2026-10-28.md).
A clean checkout without optional local notebook/data files can also skip
`test_local_notebook_source_fingerprint` and `test_local_processed_schema_only`;
the committed synthetic notebook-parity fixture remains exercised. No patient
dataset or model artifact is needed by the non-database suite.

`ruff-ci.toml` defines an explicit clean baseline for the inference/feature/artifact
and trajectory modules and their new regression tests. It checks syntax, undefined
names, unused/redefined symbols and import ordering (`E4/E7/E9/F/I`). `E402` is
excluded to preserve the API's existing staged imports. Older research modules,
scripts, notebooks and legacy tests remain outside this lint scope. This is not a
claim of repository-wide Ruff cleanliness. Add new production modules and their
tests to this scope; widen it as legacy debt is addressed in separate changes.

Run the equivalent local checks without contacting a database:

```bash
RUN_DB_TESTS=0 PYTHON_DOTENV_DISABLED=1 uv run --no-sync pytest -q -ra
uv run --no-sync ruff check --config ruff-ci.toml .
uv lock --check --offline
git diff --check
```

GitHub validates the committed checkout, not untracked local files. Include the
reviewed implementation and synthetic fixtures when preparing the change; keep
patient records, model binaries, local manifests and secrets ignored. No commit,
push or remote workflow run is performed by this task.

## Regression coverage review

This milestone changes configuration and documentation only; no production Python
behavior is added or changed. Existing recent behavior has corresponding tests:

| Production behavior | Regression tests |
| --- | --- |
| Artifact checksum, feature order, manifest compatibility, missing values, emergency static fallback, real sidecar loader | `tests/test_artifact.py` |
| Notebook feature parity, missingness, gaps, prefix causality, batch/incremental equivalence, patient isolation | `tests/test_sepsis_features.py` |
| Raw-hourly adaptation, model-feature compatibility, episode/time validation, outcome-input rejection | `tests/test_raw_trajectories.py` |
| Trajectory state isolation, fallback/recovery, bounds, prefix causality, engineered endpoint compatibility | `tests/test_trajectories.py` |
| Existing inference, optimizer, bounded controller, HMM, agent/memory behavior and API routes | Existing `test_sepsis_inference`, `test_controller_*`, `test_adaptive_controller`, `test_hmm`, `test_memory*`, `test_agent_workflow`, `test_api` modules |

This is a test-to-behavior inventory, not exhaustive coverage certification.
In particular the offline artifact verification CLI has a split-overlap regression
test and separate local retrospective evidence, but not automated synthetic coverage
of every successful/failing provenance-reconstruction branch. Statistical registry
and paired-suite edge cases also need dedicated tests; `test_registry.py` currently
tests runtime observability, while `test_evaluation.py` tests the older metric helpers.

## Current database integration

The repository uses psycopg and parameterized SQL. `database_connection()` obtains
credentials from the environment, opens a PostgreSQL connection with a five-second
timeout, and uses the connection context for commit/rollback. The combined
`save_agent_run_with_selections()` function issues all inserts in one transaction.
That is an implemented transaction boundary, not yet a verified rollback test.

Existing ordered SQL files:

1. `sql/001_initial_schema.sql`: `agent_runs`, `memory_items`,
   `evaluation_results`, constraints and indexes.
2. `sql/002_memory_scoping.sql`: nullable `memory_items.episode_id`, its index,
   and `memory_selections` with run FK, unique selected IDs/ranks and positive ranks.

Compose runs PostgreSQL 16 bound to localhost (host port defaults to 5433), with
a named persistent volume and a healthcheck. Its API service depends on database
health. SQL files are **not** mounted or automatically applied, and there is no
migration-version ledger. Existing persistent volumes may therefore have different
schema states. The repository's host port default is 5432, unlike Compose's host
mapping. Set the intended test port explicitly; do not infer readiness from `/ready`.
The current API readiness route performs no database check.

The original live test, `test_persist_and_retrieve_run`, writes and retrieves one
synthetic run. The module now contains 15 passing live cases, including episode
isolation, constraints and rollback. `test_api.py` mocks database retrieval/persistence;
those API tests do not by themselves exercise actual PostgreSQL behavior.

## Run existing opt-in tests in a disposable database

The following is a **manual runbook**, not commands executed by this task. It starts
a separate disposable PostgreSQL 16 container, without the Compose persistent
volume or API service. Use a free localhost port and a unique container name.
Run in a fresh **Bash** shell from the repository root. The password prompt supplies
only a temporary development credential; it does not change any existing account.
Do not use production credentials or a production database.

```bash
set -euo pipefail
export POSTGRES_HOST=127.0.0.1
export POSTGRES_PORT=55433
export POSTGRES_DB=healthcare_ai_test
export POSTGRES_USER=healthcare_test
export PYTHON_DOTENV_DISABLED=1
read -r -s -p 'Disposable database password: ' POSTGRES_PASSWORD
printf '\n'
export POSTGRES_PASSWORD

docker run --detach --rm --name healthcare-ai-lab-dbtest \
  --publish 127.0.0.1:55433:5432 \
  --env POSTGRES_DB --env POSTGRES_USER --env POSTGRES_PASSWORD \
  postgres:16

# Register cleanup only after this new container was created successfully.
trap 'docker stop healthcare-ai-lab-dbtest >/dev/null; unset POSTGRES_PASSWORD' EXIT

for attempt in {1..30}; do
  if docker exec healthcare-ai-lab-dbtest \
    pg_isready -U healthcare_test -d healthcare_ai_test; then
    break
  fi
  sleep 1
done
docker exec healthcare-ai-lab-dbtest \
  pg_isready -U healthcare_test -d healthcare_ai_test

# Apply only the two existing SQL files, to this new disposable database.
docker exec -i healthcare-ai-lab-dbtest \
  psql -U healthcare_test -d healthcare_ai_test -v ON_ERROR_STOP=1 \
  --single-transaction < sql/001_initial_schema.sql
docker exec -i healthcare-ai-lab-dbtest \
  psql -U healthcare_test -d healthcare_ai_test -v ON_ERROR_STOP=1 \
  --single-transaction < sql/002_memory_scoping.sql

RUN_DB_TESTS=1 uv run --no-sync pytest tests/test_database_integration.py -q -ra
```

The container is removed on shell exit; no named production/development volume is
reused. Synthetic test rows remain only until disposable-container removal. The
subsequent local verification used an equivalent new, randomly named loopback
container, applied only 001/002, passed all live tests and removed that container.
Existing services and production databases were not queried or changed.

## Gaps and minimal next integration plan

1. Completed locally: existing SQL files applied, all four tables verified, and
   repository integration exercised independently of Compose health and API mocks.
2. Completed locally: memory/episode retrieval, limits/order, selected ranks, unique/
   FK/check constraints and second-insert rollback. API storage-failure unit tests
   and broader concurrency/error cases remain useful follow-up coverage.
3. Now that the live suite passes locally, propose a separate opt-in PostgreSQL 16 CI job with
   an ephemeral service. Keep the fast non-database job independent. No database
   service or schema change is added in this milestone.
4. Make later schema decisions explicitly: migrations have no ledger; old memory
   rows can retain null episodes; selections have no FK to `memory_items` or database
   constraint linking a selected memory to its run's episode. Selection scores and
   evaluation values lack the finite-value constraint used on `memory_items`.
   `evaluation_results` has no repository write interface or integration tests.
   These are review items, not approved migrations.
5. Before using stored memories in causal patient replay, define observation and
   outcome availability times and a decision-time retrieval cutoff. Current retrieval
   filters episodes but not temporal eligibility, and a separate global-candidate
   helper is unscoped. Trajectory/controller state and outcome persistence are not
   integrated. Do not treat database `created_at` as a verified clinical timestamp.

PostgreSQL repository readiness is **verified locally for the 15 synthetic cases**;
hosted database CI and production readiness remain unverified. The next integration
steps are a reviewed database CI job and local API authentication/capability-readiness
work. Mathematics, model artifacts, API routes, database source, SQL files and Compose
configuration remain unchanged.
