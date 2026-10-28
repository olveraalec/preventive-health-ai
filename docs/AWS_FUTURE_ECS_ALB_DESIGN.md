# Future ECS/ALB blue-green alternative — design only

Date: 2026-10-28. Implements no infrastructure and does not revise the research ADRs.
The EC2/Nginx prototype proposal is documented separately in
[API_SECURITY_READINESS_2026-10-28.md](API_SECURITY_READINESS_2026-10-28.md).

## Architecture and rollout

Propose a private ECR repository with immutable image digests; an ECS service with
controller `ECS` and strategy `BLUE_GREEN`; an HTTPS ALB listener with ACM certificate;
blue and green target groups; and an isolated test listener/rule for candidate probes.
Keep ECS tasks and PostgreSQL/RDS private. An internet-facing ALB is optional and
requires explicit audience/network approval; never make task or DB ports public.
Configure the native ECS managed traffic-shift integration and its infrastructure
role rather than combining native ECS instructions with legacy CodeDeploy task-set
configuration. CodeDeploy is a separate future alternative, not an implicit requirement.

Freeze a release manifest with git SHA, image digest, Python/uv lock, model SHA,
feature schema, metric-registry version and compatible migration inventory. Stage
green concurrently, verify real capability readiness and synthetic inference, then
shift traffic using a reviewed rule and observe during a configured bake interval.
CloudWatch alarms should cover error rates, latency, task health, resource pressure
and synthetic probe failures; rollback routing to retained healthy blue on failure.
Thresholds and bake duration require measured baseline traffic/cold-start evidence.

Blue and green need simultaneous capacity. Do not promise zero downtime without
dual-capacity, connection draining, health and rollback tests. The current `/ready`
stub is not a valid research-capability ALB health gate. Auth may use an approved
OIDC/Cognito ALB action, but private task ingress and application authorization must
remain enforced; do not trust client-supplied identity headers. Validate the chosen
identity mechanism and failure behavior before adding a listener rule.

Current SQL remains unchanged. Any later migration must be expand-compatible with
both versions before traffic moves; contracting/removing columns requires a later
review after rollback is no longer needed. Application rollback does not reverse SQL.
Use encrypted backups with tested restore procedures for any persistent database.

These choices follow the current
[ECS deployment model](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/deployment-type-blue-green.html)
and [required resources](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/blue-green-deployment-implementation.html).

## IAM and GitHub OIDC requirements

Keep the current test CI read-only. A separately approved deployment job may receive
`id-token: write` and `contents: read`, protected-environment approval and exact-ref
restrictions. Use short-lived OIDC federation, not static AWS access keys.

The role trust conditions must require `aud=sts.amazonaws.com` and an **exact**
subject for the intended repository and protected environment. Do not use a broad
`repo:ORG/REPO:*` condition. GitHub documents legacy and newer immutable owner/repo
ID subject formats; select the actual repository's format instead of guessing IDs.
Environment branch/tag protection is also required. No JWT, AWS key or role credential
should be printed, persisted in Git or copied into model/image artifacts.

Separate roles and permissions:

| Role | Proposed scope |
| --- | --- |
| Test CI | No AWS role or token permission |
| Image publishing job | Only the intended ECR repository's upload actions; `ecr:GetAuthorizationToken` where its API requires global resource scope |
| Deployment job | Describe/update only intended ECS service/task resources; narrowly reviewed task-definition registration; `iam:PassRole` only for exact approved roles with the ECS-tasks service condition |
| ECS execution role | Pull the intended image, write approved log streams, retrieve exact injected secret ARNs and permitted KMS key where needed |
| ECS task role | Only runtime API permissions actually needed; no infrastructure administration or broad Secrets Manager access |
| ECS infrastructure role | Only the managed load-balancer integration permissions required for the approved service |
| EC2 fallback instance role | Session Manager and narrowly scoped image/secret/log access; no embedded developer keys |

Provisioning permissions belong to an independently approved setup role. Do not grant
the deploy job blanket IAM, security-group, DNS or account-administration access.
For EC2 replacement, any future SSM command permission must be constrained to the
approved instance tags and command document, not arbitrary shell access to all hosts.

See [GitHub OIDC in AWS](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws).

## Secrets and network controls

Choose Secrets Manager or encrypted SSM parameters for exact runtime secrets. Separate
deployment identity, DB bootstrap/migration role and least-privilege DB runtime role.
Prefer task-agent injection or tightly scoped runtime retrieval; keep values out of
images, source, Compose defaults, command-line arguments, test reports and logs.
Define rotation/restart behavior and failure/readiness when a secret cannot load.
No current credentials are altered by this proposal.

Security groups should form explicit edges: audience -> ALB 443; ALB SG -> task
port; application SG -> database 5432. Deny direct internet access to tasks, database,
SSH, Uvicorn, Jupyter and development host ports. Do not assume a particular VPC CIDR.
Limit egress to required services; compare approved VPC endpoints with NAT cost and
availability. Restrict operational access through Session Manager. Choose/log TLS
termination boundaries explicitly; an HTTPS edge with HTTP backend is not end-to-end TLS.

Logs and monitoring must use allowlisted, patient-free fields with retention and
access policies. ALB/Nginx/native error logs can contain URLs and network identifiers;
do not automatically export them as privacy-safe research telemetry. External demos
should use synthetic payloads; real local retrospective records remain local.

## Budget and capacity gates

Before provisioning, approve a region, monthly cap, daily burn target, owner and
expiration date. Measure actual image size, peak model RSS, CPU/latency, cold-start,
database connections and two-version concurrent memory. No instance-size, dollar
estimate or free-tier assumption is certified by this audit.

Price both options with the selected region's current rate card:

- EC2: instance hours + EBS + public IPv4 if used + transfer + snapshots + logs +
  secrets, plus database/RDS if separate. Add a second host if true blue/green is required.
- ECS/ALB: vCPU/memory task hours (including green overlap) + ALB hours/LCUs +
  private database + storage/backups + ECR + logs + secrets + NAT/endpoints + transfer.

Propose actual/forecast alerts at 50/80/100 percent of the approved cap, resource tags
for owner/project/expiry, bounded replica counts, retention and cleanup ownership.
Budgets are delayed alerts, not a real-time hard spending stop; any automatic budget
action requires its own reviewed IAM/failure plan. Do not use hypothetical alerts
as permission to provision resources.
See [AWS Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html).

## Unverified prerequisites

No AWS configuration, role trust, SG/firewall, DNS, certificate, cost estimate, ECS
task capacity, ALB health, traffic switch, CloudWatch rollback or restore procedure
has been created or tested. Auth gate, capability readiness, redacted diagnostic
logging and immutable image/artifact delivery are not implemented. No cloud deployment
is authorized by this document.
