# Manual ECR image publication

This adds `.github/workflows/publish-ecr.yml` independently of the existing CI.
It publishes a research application image only. It does not provision resources,
modify IAM, deploy EC2/ECS, migrate a database, or establish clinical/model readiness.
The existing AWS security ADR v0.2, deployment safety ADR v0.1 and future ECS design
remain applicable; this implements only the separately requested image-publishing step.

## Review and merge safely

1. Review the diff on `feature/aws-publish`, including the Dockerfile, root
   `.dockerignore`, workflow and synthetic configuration tests. Keep other agents'
   work in the main checkout untouched. Commit/push only through your normal review
   process; this implementation does neither automatically.
2. Open a PR with **base `feature/codex-integration`** and head `feature/aws-publish`.
   Resolve conflicts through review, preserve `.github/workflows/ci.yml`, and require
   its checks to pass. After merging, repeat checks on the resulting target commit.
3. Do not dispatch from `feature/aws-publish`. The job permits exactly
   `refs/heads/feature/codex-integration`, matching the reported IAM branch trust.
   Do not add a GitHub `environment` without separately reviewing OIDC subject/trust
   changes. No IAM relationship has been inspected or changed here.
4. Only after review and merge, an authorized maintainer may manually dispatch
   `publish-ecr.yml` selecting `feature/codex-integration`. GitHub requires a manual
   workflow to be present on the default branch for dispatch availability. If the
   integration branch is not the default branch, arrange reviewed registration of
   this same guarded workflow on the default branch before dispatch; do not weaken
   the branch restriction or change the default branch for convenience.

## Existing AWS prerequisites (no automatic setup)

- GitHub repository variables: `AWS_REGION=us-west-2`, `AWS_ROLE_ARN` set to the
  approved role ARN. No static AWS keys or patient/model secrets are supplied.
- Existing private ECR repository `preventive-health-ai` in that role's account and
  region, already configured for `IMMUTABLE` tags with no exclusions. A mutable or
  missing repository blocks publication; this workflow never changes its settings.
- Existing OIDC trust must match this GitHub repository, the authorized branch and
  audience `sts.amazonaws.com`. The user reports branch authorization; live AWS
  trust and permission verification remain pending.
- Role permissions scoped to that ECR repository: `DescribeRepositories`,
  `DescribeImages`, `BatchCheckLayerAvailability`, `InitiateLayerUpload`,
  `UploadLayerPart`, `CompleteLayerUpload`, `PutImage`; `GetAuthorizationToken`
  requires its API's global resource scope. Missing permissions are blockers;
  request a separately reviewed IAM change instead of broadening this workflow.
- Approved publication budget/ownership and registry retention/security review.

## Gates and outputs

The read-only validation job installs locked Python 3.12/dev dependencies with
uv 0.11.26, runs the full non-database suite and the existing Ruff baseline plus
workflow tests. Failures block publishing, independently of the normal CI workflow.
Database integration is intentionally disabled because no persistence changes occur.

The branch-guarded publishing job validates variables, builds with
`docker buildx build --platform linux/amd64 --load`, checks the resulting OS/CPU and
non-root user, imports runtime dependencies/API, then starts the image with no
external network and probes `/health` internally. It authenticates with OIDC only
after those gates. No port is published and no database or model is contacted.

The exact locally tested image is pushed as `<repository-uri>:<full-40-character-SHA>`;
there is no `latest` tag. ECR immutability rejects overwrites, including reruns for
an already published SHA. Use an existing verified digest or a reviewed new commit;
do not delete/rewrite a tag to make a retry succeed. Concurrency serializes runs per
ref. The job summary reports the commit, tag, queried ECR digest and digest-qualified
pull reference. If digest lookup fails after push, publication may have succeeded:
check ECR before retrying. No successful digest is claimed when lookup fails.

## Docker/data boundary and deployment blockers

The new **root** `.dockerignore` denies everything except Docker/packaging inputs,
README and Python source; the old `tests/.dockerignore` did not protect root builds.
Patient datasets, `.env`, AWS credentials, tests, local virtual environments, model
binaries and artifact manifests are excluded. Review admitted Python/README content
for embedded secrets or records before merging: extension filtering is not a secret
scanner. Do not insert patient records or encoded model binaries into source files.

The Dockerfile retains its multi-stage/non-root design and production-only frozen
lock installation. It now supplies the README required by package metadata and pins
uv to the CI version. Python base image tags remain mutable; commit tags identify
source and are enforced immutable in ECR, but do not promise reproducible rebuilds.
This load/test/push path does not attach SBOM/provenance attestations. Those and
dependency/secret scanning remain separate release-hardening gates before deployment.

No approved model delivery mechanism is introduced: the application image contains
no research model bundle, sidecar or patient data. Liveness is not model readiness.
Artifact provenance/checksum validation, capability readiness, auth/TLS, private
database configuration, capacity/cost approval and rollback controls remain blockers
to a real deployment under the existing ADRs. No API/schema/math definitions change.

## Local validation

```bash
uv run --no-sync pytest tests/test_publish_workflow.py -q
uv run --no-sync pytest -q
uv run --no-sync ruff check tests/test_publish_workflow.py
uv run --no-sync ruff check --config ruff-ci.toml .
uv lock --check --offline
git diff --check
```

Configuration tests parse YAML and check the trigger, branch/permission isolation,
test/build ordering, architecture gate, immutable tag/digest contract and packaging
inputs. They do not emulate GitHub Actions or AWS. Validate actual hosted behavior
only after the reviewed merge; no dispatch or AWS access was performed here.

Worktree verification on 2026-10-28: baseline 134 passed/17 skipped; after changes
140 passed/17 skipped, including six new configuration tests. The skips were 15
database integration cases and two optional local notebook/schema checks. Scoped
Ruff, the existing Ruff baseline, offline lock verification, YAML parsing, all shell
steps' `bash -n` checks and whitespace checks passed. Initial environment setup
could not use the sandboxed default uv cache; dependencies were subsequently
installed entirely offline into this worktree's `.venv` using the existing cache.

The local Docker daemon reports `linux/aarch64`; neither `python:3.12-slim` nor the
BuildKit image is cached. An amd64 container build/smoke test was not run locally
because it would require uncached external downloads. The workflow's actual Linux
amd64 build, runtime dependency installation and smoke gates remain unverified until
the reviewed hosted run. `actionlint` is not installed; YAML parsing and shell syntax
validation do not replace GitHub's workflow validation. Live DB tests were not run
because no persistence behavior changes.
