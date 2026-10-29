# AWS external bootstrap — healthcare research prototype

**Status:** Setup guide only, **not** deployed. Recorded 2026-10-28.

## Safety boundaries

- Treat GitHub-hosted CI as working, but ECR push, EC2 deployment and blue/green CD as **not yet verified**.
- The API audit identified unauthenticated routes, insufficient object authorization, weak readiness, unbounded requests, and incomplete logging controls. Do **not** make the research API publicly reachable until these are remedied and tested.
- Work outside the repository while Codex modifies the backend. Do not edit current workflows or application files concurrently.
- Never paste AWS credentials, authorization codes, raw patient data, or private keys into a chat, Git, Docker images or CI logs.
- Use one AWS region (`us-west-2` is consistent with earlier projects), least-privilege roles, immutable images, and resource cleanup/budget monitoring.

## Phase 0 — account and repository inspection (read-only)

From `~/code/projects/hackathon-starter` on macOS:

```bash
brew install awscli             # Only if `aws` is not installed; do not reinstall unnecessarily.
aws --version
gh --version
aws sts get-caller-identity --query '{Account:Account,Arn:Arn}' --output table
export AWS_REGION=us-west-2
export AWS_DEFAULT_REGION="$AWS_REGION"
gh repo view --json nameWithOwner --jq .nameWithOwner
git branch --show-current
gh run list --limit 5
```

If AWS login is unavailable, use your existing IAM Identity Center/SSO profile (`aws configure sso`, then `aws sso login --profile <name>`), or another established non-root, limited-permission administrator identity. Do not share access keys.

## Phase 1 — set a cost budget (console)

Open AWS Billing & Cost Management → Budgets. Create a modest monthly **cost** budget and email alerts; the budget is notification-oriented, not a hard spend limit. Monitor EBS, EC2 uptime, IPv4, and ALB especially. Resources incur charges while running even if unused.

## Phase 2 — create ECR (low cost, non-public)

```bash
export AWS_REGION=us-west-2
aws ecr describe-repositories \
  --repository-names preventive-health-ai \
  --region "$AWS_REGION" \
  --query 'repositories[0].repositoryUri' --output text
```

If `RepositoryNotFoundException`, create it **once**:

```bash
aws ecr create-repository \
  --repository-name preventive-health-ai \
  --image-tag-mutability IMMUTABLE \
  --image-scanning-configuration scanOnPush=true \
  --region "$AWS_REGION" \
  --query 'repository.{URI:repositoryUri,ARN:repositoryArn}' \
  --output table
```

Immutable tags prevent overwriting release tags. Push unique Git commit SHA tags, then deploy by image **digest**. Set a lifecycle policy later to bound storage costs.

## Phase 3 — GitHub Actions ↔ AWS OIDC (no long-lived AWS keys)

In AWS IAM → Identity providers, add OpenID Connect provider:

- Issuer: `https://token.actions.githubusercontent.com`
- Audience: `sts.amazonaws.com`

Then create a role for GitHub's OIDC provider with `sts:AssumeRoleWithWebIdentity` trust scoped to **only the intended repository/branch**. **Important 2026 GitHub change:** repositories created after July 15, 2026 use immutable owner/repository ID claims in the OIDC `sub` by default. The older name-only format may not work. Verify exact repository created date and token subject before finalizing trust. For newer repositories, the subject resembles:

```text
repo:OWNER@OWNER_NUMERIC_ID/REPO@REPO_NUMERIC_ID:ref:refs/heads/main
```

Repository metadata:

```bash
REPO_SLUG="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
gh api "repos/$REPO_SLUG" \
  --jq '{name: .full_name, owner_id: .owner.id, repo_id: .id, created: .created_at}'
```

If a workflow uses a GitHub *environment*, the `sub` changes again to environment syntax; scope the branch with GitHub Environment branch-protection rules. Don't use `repo:OWNER/*` or general wildcards.

Create an IAM permissions policy dedicated to ECR push only. Required API actions typically include:

- `ecr:GetAuthorizationToken` (resource `*`)
- `ecr:BatchCheckLayerAvailability`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, `ecr:PutImage` (resource only the **specific ECR repository ARN**)

Optionally add `ecr:DescribeImages` restricted to the repository for post-push verification. Do **not** attach `AdministratorAccess` or grant EC2 or IAM modification privileges to the ECR-only role.

Add the role ARN to **GitHub repository Actions variables** as `AWS_ECR_ROLE_ARN`, and region as `AWS_REGION=us-west-2`. Role ARN is an identifier, not a secret. Require code reviews on the release branch; prefer manual `workflow_dispatch` for the first ECR push.

When Codex has finished touching `.github/workflows`, add a separate ECR publish workflow gated on test success and `workflow_dispatch`, using `permissions: {contents: read, id-token: write}` and `aws-actions/configure-aws-credentials`. Scope AWS trust to the release branch, not arbitrary pull requests. Test first using a tiny temporary container image or a model-free API image; do not include raw data or arbitrary joblib files.
## Phase 4 — EC2 deployment capacity and networking (after security fixes)

- Begin with capacity measurement of API Docker container (`docker stats` after warmup), not a guess. Choose x86_64 Docker build if deploying to `t3.*`; your Mac is ARM64.
- If needed, a `t3.medium` is 4 GiB, `t3.large` is 8 GiB. Larger instance may make concurrent blue/green containers feasible, but a single instance remains a single failure domain.
- EC2 managed with **Session Manager** through a narrowly scoped instance role (usually `AmazonSSMManagedInstanceCore`). Prefer **no inbound SSH 22**. For private subnets use VPC endpoints or a carefully secured egress design for SSM and ECR.
- Initially allow **zero inbound** ports while testing via SSM port forwarding. Database port 5432/5433 must never be exposed publicly.
- Once authentication, input bounds, readiness, artifact attestation and secrets are verified, add HTTPS using Nginx plus TLS on a domain (or ALB+ACM). HTTPS is not a substitute for API authorization.
- Ship only the image and trusted model artifact+checksum manifest. In the uploaded starting snapshot Dockerfile copied `src/` but not `artifacts/models/`—a gap that must be fixed or a separate verified artifact mount provided before claiming a working cloud model.

## Phase 5 — deployment topology choices

**Cost-oriented demonstration:** One EC2 instance with Nginx + FastAPI image and health-checked replacement/rollback. Not true zero-downtime blue/green when host capacity is insufficient.

**Single-host blue/green exercise:** If RAM allows, run two release containers on separate internal ports, warm and health-check green, then change the Nginx upstream and reload; preserve blue for rollback. Host remains single point of failure.

**Real cloud blue/green:** ECS native blue/green with separate task sets and ALB target groups; supports staged traffic shifting, bake periods and rollback. More infrastructure and cost; implement after demo readiness.

## Phase 6 — minimal cloud acceptance tests

1. GitHub CI green; no credentials in logs.
2. ECR image digest and source commit recorded; image built for target CPU architecture.
3. Private app returns liveness and **capability-specific readiness** with verified model+manifest; fail closed on invalid artifact.
4. Authenticated inference responds to synthetic requests; unauthenticated requests are denied.
5. DB only reachable from authorized app network paths, tests pass.
6. Blue / green smoke test identifies active release and rollback restores previous digest.
7. Privacy and non-clinical messaging verified; no patient records or secrets logged.
8. Terminate unused EC2 and ALB resources and monitor charges.

Official references:
- https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws
- https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-tag-mutability.html
- https://aws.amazon.com/blogs/devops/choosing-between-amazon-ecs-blue-green-native-or-aws-codedeploy-in-aws-cdk/
- https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-prerequisites.html
