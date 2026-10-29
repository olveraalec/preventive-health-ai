# Manual publication of the model-free research API

This adds `.github/workflows/publish-ecr.yml`; existing `ci.yml` is unchanged.
It publishes an application image, not a deployed or clinically validated service.
No workflow was dispatched and no ECR/EC2/IAM resources were created by this task.

## Execution contract

- Only `workflow_dispatch`, repository `olveraalec/preventive-health-ai`, ref
  `refs/heads/feature/codex-integration`. Other branches/repositories skip both
  jobs. The shell guard also rejects wrong scope, malformed commit SHA, missing
  role or a region other than `us-west-2`.
- `validate` checks out the exact event commit, installs the locked Python 3.12
  project/dev dependencies, runs the full non-database pytest suite and scoped
  Ruff. Test failures block `publish`; the summary reports counts and skip reasons.
- `publish` depends on successful validation of that same SHA. It builds
  `linux/amd64`, loads the candidate locally and verifies its architecture and
  non-root user. An offline, network-disabled container probe imports the API,
  checks `/health`, preserves `/ready`'s `inference_enabled=false`, and requires
  missing-model inference to return 503. It checks the application tree for
  unapproved payloads. This does not establish database or model readiness.
- Only then does `aws-actions/configure-aws-credentials@v6` assume the existing
  OIDC role from **`vars.AWS_ROLE_ARN`**, using **`vars.AWS_REGION`**. This task's
  variable name supersedes the older bootstrap guide's `AWS_ECR_ROLE_ARN` example;
  no existing variables or credentials are changed.
- The private ECR login masks its password. The verified local image is tagged
  and pushed as `<registry>/preventive-health-ai:<full Git commit SHA>`.
  `aws ecr describe-images` retrieves its remote SHA256 digest. Invalid/missing
  digests fail the job; job outputs and summary expose the immutable
  `<registry>/preventive-health-ai@sha256:...` reference. There is no `latest` tag.
- Default permissions are `contents: read`; only `publish` also grants
  `id-token: write`. No static AWS keys, deployment steps, exposed host ports,
  environment-based OIDC subjects or infrastructure provisioning are introduced.
  Concurrent publications are serialized without cancelling an active push.

## Before the first manual run

1. Commit/review the change on the authorized branch. GitHub requires the workflow
   definition on the repository's **default branch** to enable manual dispatch;
   select `feature/codex-integration` as the run ref. Adding the definition to the
   default branch does not authorize publishing that branch. See
   [GitHub manual workflow documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
2. Verify repository variables `AWS_ROLE_ARN` and `AWS_REGION=us-west-2`. The
   existing role's OIDC trust must match the actual repository/authorized ref and
   `sts.amazonaws.com` audience. YAML guards supplement IAM trust and branch
   review/protection; they do not replace them. Actual IAM trust was not inspected.
3. The private ECR repository must already exist. The role needs
   `ecr:GetAuthorizationToken` on `*`, and repository-scoped
   `ecr:BatchCheckLayerAvailability`, `ecr:InitiateLayerUpload`,
   `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, `ecr:PutImage`, and
   **`ecr:DescribeImages`** for digest verification. AWS's
   [ECR push permissions reference](https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push-iam.html)
   describes the push actions. No EC2/ECS/IAM modification permissions are needed.
4. Prefer existing ECR tag immutability and scan-on-push. This workflow does not
   alter repository settings, inspect scan findings or certify absence of
   vulnerabilities. Rerunning the same commit may fail against immutable tags;
   do not delete/overwrite an existing release to force a rerun. A digest-read
   failure after a successful push leaves the image in ECR but fails the job;
   inspect the run/repository before retrying.

After these prerequisites, a human with repository write access can run:

```bash
gh workflow run publish-ecr.yml --repo olveraalec/preventive-health-ai --ref feature/codex-integration
```

The command is documentation only and was not executed. No public API or deployment
is enabled by publishing the image.

## Image contents and build changes

The previous `.dockerignore` was under `tests/`, so it did not protect the root
Docker build context. The new root file denies everything and admits only
Dockerfile/ignore metadata, `pyproject.toml`, `uv.lock`, `README.md`, and `.py` files
under `src/`. Directory exceptions must not be added: Docker treats them as
including descendants, which can admit credentials or binaries. The rules were
exercised with a real scratch Docker build containing only synthetic patient,
credential, joblib and bytecode sentinels; those payloads were excluded and nested
Python sources retained. Any future non-Python package resources need explicit
review before extending this allowlist. Source code itself still requires normal
secret/privacy review; an extension filter cannot certify arbitrary Python text.

The Dockerfile now copies the README required by Hatch's project metadata, pins
uv to the existing CI version `0.11.26` and uses `uv sync --locked`. The existing
Python 3.12, dependencies, non-root runtime and container-internal Uvicorn binding
are preserved. No credential build arguments, dataset copies or model copies are
added. The base image and third-party action major tags remain updateable; a Git
SHA tag alone does not make later rebuilds byte-identical. Record/use the published
digest; future supply-chain work can pin base/action digests and add SBOM/scans.

## Remaining deployment blockers

- **Model delivery:** no model bundle or local manifest is embedded. A separately
  approved, access-controlled delivery mechanism must provide the trusted
  `prototype_bundle.joblib` and adjacent checksum-bound
  `prototype_bundle.manifest.json`, preferably read-only; set
  `SEPSIS_MODEL_BUNDLE` to that runtime path. Preserve feature versions and trained
  provenance. Do not bake arbitrary joblib files or patient records into the image.
- **Exact model-runtime compatibility:** current manifest validation compares all
  verified software versions, including the Python patch version. This build's
  floating `python:3.12-slim` resolved to Python 3.12.16; the approved local model
  manifest records Python 3.12.13. Merely mounting it into this image will fail
  compatibility validation. A future model-capable release must pin a compatible
  runtime or perform the existing artifact verification procedure in the selected
  runtime before approving a corresponding manifest. Do not weaken validation,
  edit provenance declarations by hand, or claim the model-free smoke verifies it.
- **Readiness/security:** `/health` and this probe do not validate inference.
  Existing generic `/ready` intentionally reports inference disabled. Capability
  readiness, authentication/authorization, request bounds, protected API docs,
  trusted proxy headers and HTTPS still need separate implementation/review before
  public exposure. Existing API contracts remain unchanged.
- **Runtime dependencies/capacity:** the Linux amd64 lock pulls PyTorch and CUDA
  packages through the existing ML dependency stack, despite this image not using
  a GPU for its API smoke test. This increases download/build time and disk/image
  requirements. CPU-only/minimal dependency packaging is a separate reviewed task;
  no dependencies or lockfile are changed here. Measure RSS/startup/storage before
  choosing EC2 capacity. No EC2 architecture was selected; a future ARM/Graviton
  target requires an explicit platform and dependency verification change.
- **Persistence:** database-backed routes require an isolated/private database,
  runtime-injected secrets, existing migrations and live connectivity tests. No
  PostgreSQL configuration/schema changes or live database tests occur here.
- **Hosted verification:** first hosted test/build, OIDC assumption, private ECR
  push and remote digest verification remain unverified until a human dispatches
  the workflow. No remote immutable digest can be reported by this local task.

Read for this task: root AGENTS and pyproject, AWS security/deployment ADR v0.2,
blue/green safety ADR v0.1, and the external bootstrap runbook. No mathematical,
inference, model, frontend or database behavior is changed.

## Local verification receipt

Baseline: 262 tests passed, 15 opt-in PostgreSQL tests skipped. Final checks:

```text
pytest -q tests/test_ecr_publish.py: 18 passed
pytest -q -rs: 280 passed, 15 skipped (Requires local PostgreSQL)
ruff check tests/test_ecr_publish.py: passed
ruff check --config ruff-ci.toml .: passed
actionlint 1.7.12, including ShellCheck: passed
git diff --check, plus new-file whitespace checks: passed
Real Docker scratch-context synthetic payload isolation: passed
Final linux/amd64 application build: passed
Exact workflow offline container smoke script: passed
```

Python checks used `PYTHON_DOTENV_DISABLED=1 RUN_DB_TESTS=0`, existing uv and the
temporary uv cache. Docker checks ran locally without AWS credentials, host port
publication, model mounts or database connectivity. The initial new-workflow guard
tests exposed failed-command handling in the local shell; explicit rejection/
failure exits were added, and all failure paths now pass. Docker sentinel tests
also established why source-directory exceptions must not be used. No existing
backend tests were weakened.

The final image is local tag `preventive-health-ai:local-publish-verified`,
`linux/amd64`, user `appuser`. Docker reports 3,307,613,055 image bytes (about 3.31 GB);
this is not a measurement of peak build disk, RSS or deployed startup latency.
The original dependency download/preparation took about 8m24s on this machine;
the final rebuild reused that cache. Verification ran under emulation on an ARM64
Mac, not on an EC2 host or GitHub runner. Local image ID:
`sha256:4164a67dd64503c3a8a7ac7ed2a47e123b0c4f03562ac1eb37821f330d16d147`.
This is **not** an ECR digest or published release identity; the build includes
the current uncommitted worktree. The workflow will test/build its checked-out SHA.

Files added: root `.dockerignore`, `publish-ecr.yml`, this guide and
`tests/test_ecr_publish.py`. Files updated: Dockerfile and the new test's
`ruff-ci.toml` scope entry. Existing CI, dependency lock/project metadata, APIs,
model artifacts and mathematical research history were preserved. Live PostgreSQL,
OIDC trust, ECR authorization/push/digest and hosted execution remain unverified.
