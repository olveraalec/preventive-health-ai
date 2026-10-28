# AWS API Security, HTTPS, and Deployment Decision Record — v0.2

**Date:** 2026-10-28  
**Project:** `hackathon-starter` / `healthcare_ai_lab`  
**Status:** Design approved for research planning only; no cloud provisioning or public exposure authorized.  
**Relationship:** Expands `DEPLOYMENT_SAFETY_BLUE_GREEN_ADR_v0.1.md` rather than replacing it.

## 1. Threat model and public demo boundary

The prototype is research-only and predicts the shifted PhysioNet Sepsis Challenge label, not clinically validated onset risk. Risks include: unauthorized inference/API access; disclosure of credentials or patient information; exposed PostgreSQL, SSH, Jupyter, Uvicorn or Swagger; unencrypted transport; model-artifact spoofing; dependency supply-chain changes; log leakage; and data-forward leakage from replay.

Terminology:
- `127.0.0.1` is loopback, not an internet-facing address.
- `0.0.0.0` is a wildcard *bind* address, not a destination IP. It can be correct **inside a container** when the host's published port is restricted to `127.0.0.1`.
- `172.31.0.0/16` is commonly used for private AWS VPC networking; actual ranges depend on configuration.
- EC2 public IPv4 addresses and their `ec2-...compute.amazonaws.com` hostnames can be legitimate public endpoints and are not necessarily a developer's home IP. Public addressing is not by itself a vulnerability; what is exposed on it matters.

**Existing repository snapshot evidence (Oct 28)**: `compose.yaml` binds PostgreSQL on `127.0.0.1:${POSTGRES_HOST_PORT:-5433}:5432` and FastAPI on `127.0.0.1:8002:8000`, so this configuration does not publish those ports to remote interfaces on the Docker host. Its Dockerfile's Uvicorn process binds `0.0.0.0:8000` **inside the container**, necessary for Docker port forwarding. Verify the *current* files again before deployment; the snapshot is not a live security audit.

## 2. Three deployment tiers

### A. Local development (recommended immediately)
- FastAPI: `uv run --no-sync uvicorn healthcare_ai_lab.api.main:app --host 127.0.0.1 --port 8000`.
- Existing Docker Compose host bindings to loopback only; Postgres local port 5433, container service port 5432.
- No public DNS or internet ingress; use synthetic records for tests and browser demos where possible.

### B. Budget EC2 prototype with Nginx
- Single EC2 host and Docker or Uvicorn bound to loopback; Nginx accepts traffic on 443 (80 only for HTTPS redirect/ACME if required).
- Registered DNS name, publicly trusted certificate via ACME/Certbot; automate renewal. Nginx terminates TLS and proxies only to loopback application port.
- Security group: allow 443 from intended audience; optional 80 for redirect/issuance; deny public 8000, 8002, 5432, 5433, 8888 and SSH 22. Prefer AWS Systems Manager Session Manager using an instance IAM role. Apply host/container firewall rules as appropriate.
- Nginx HTTPS is **not** a replacement for application authentication. Public inference/agent routes need access controls, least privilege and rate limiting. Consider a managed identity provider or explicitly authorized authenticated reverse-proxy gate for the demo. Limit request sizes, redact logs, disable or protect interactive `/docs`, `/redoc` and `/openapi.json` on public deployments.
- Bind reverse-proxy client headers only from trusted proxy addresses; do not implicitly trust internet-supplied `X-Forwarded-For` headers. Restrict CORS to actual frontend origins, but do not mistake CORS for authentication.
- Deploy immutable images by digest; keep prior image+manifest for health-checked rollback. This single-instance replacement does **not** guarantee zero downtime or blue/green capability.

### C. AWS-managed true blue/green (preferred future)
- GitHub Actions locked dependency verification -> image build + SBOM/provenance -> private ECR digest.
- OIDC-federated least-privilege deployment IAM role restricted to intended repository/ref/environment. Do not use static AWS keys in GitHub Secrets.
- ECS service with native blue/green deployment + Application Load Balancer, two target groups, TLS via AWS Certificate Manager, authenticated application tier, private networking and security-group boundaries.
- Alternative separate EC2 environment/CodeDeploy when appropriate; do not force two memory-heavy model containers onto an undersized host.
- Maintain blue and green healthy concurrently, test candidate through staging listener or isolated test access, switch target-group weights only after health/readiness and synthetic inference passes; monitor and roll back routing upon alarms.
- Provision budgets, cost alerts, resource limits, tagged resources and cleanup instructions before first cloud write.

## 3. Credential, identity, and data flows

- GitHub CI should have read-only permissions by default; deployment workflow grants `id-token: write` *only where needed* for OIDC. Bind trust policy to repository, protected branch or GitHub environment; minimize ECR/ECS/EC2 IAM permissions.
- EC2 instance profile or ECS task role retrieves approved secrets from AWS Secrets Manager or SSM Parameter Store, not from source or bundled image layers. Separate application secrets from deployment identity. Rotate and audit credentials.
- Use PostgreSQL on a private network. For a small local demo, Docker PostgreSQL stays accessible only on loopback. For persistent cloud research use an explicitly configured private RDS or a securely managed private database; no publicly accessible database endpoint.
- Minimize operational tables to pseudonymous episode references, decision/model versions, metrics, timestamps and status. Never place raw clinical patient records, original IDs or model binaries into logs or GitHub artifacts.
- Perform a secret scan, dependency scan, IAM/security-group review, and synthetic privacy-safe probe before any external demonstration.

## 4. Reliability and mathematical deployment contract

- `/health`: liveness, no private operational details.
- `/ready`: process prerequisites; do not claim research model available if checksum, feature version or bundle load fails. Consider separate `/ready/research-sepsis` readiness to retain compatibility with existing API.
- Static-only emergency fallback when secondary inference fails; verify exact behavior by synthetic fault injection.
- Validate the loaded artifact's SHA-256, feature order, preprocessing version, training provenance limitations, and research-only label semantics on startup.
- Persist release manifest: git SHA, image digest, Python/uv lock, model bundle SHA, feature schema, metric-registry version, backward-compatible database migration.
- Blue/green shares a DB, so expand → compatible deploy → switch → later contract is mandatory; never assume application rollback reverses destructive SQL.
- Monitor uptime, synthetic probe results, API latency, errors, resource use, fallback rates and controller-weight distribution; never expose patient data in telemetry.

## 5. Acceptance criteria and staged execution

1. Repair repository-root `AGENTS.md` so it contains full instructions, not a broken pointer to `markdowns/AGENTS.md`. Verify with `test -f AGENTS.md`, and inspect its actual content.
2. Confirm GitHub Actions workflow executes on the hosted runner, with 136+ non-database tests and accurate skips, and checks fail on a deliberate synthetic test failure (in a safe test branch if needed).
3. Spin up disposable PostgreSQL locally and apply 001/002 SQL migrations explicitly; run opt-in DB tests including isolation, constraints, transaction rollback. Do not alter schema without review.
4. Confirm source-only security check: host ports, network listeners, secret patterns, authentication/readiness for all externally reachable endpoints, and absence of raw data in fixtures.
5. Build and run container locally with an immutable manifest; synthetic `/health`, `/ready`, and inference smoke probes must pass.
6. Document and test HTTPS + access-control reverse proxy configuration locally/staging; do not provision cloud infrastructure without specific approval.
7. Estimate cost/memory for EC2-only and ECS/ALB blue-green alternatives; record actual model peak RSS and cold-start time. Approve a budget before AWS writes.
8. When explicitly authorized, create AWS resources with OIDC and least-privilege roles, implement TLS, then add staged deployment and reversible traffic switch; database remains private.

## 6. Prohibited shortcuts

- No committing access keys, passwords, `.env`, local certificates or patient data.
- No opening `0.0.0.0/0` access to Postgres, raw Uvicorn or SSH for convenience.
- No assuming HTTPS provides login or that CORS is authorization.
- No secret values echoed in shell commands, CI summaries or screenshots.
- No labeling a rolling single-container deployment as zero-downtime blue/green.
- No claiming healthcare compliance, clinical safety, or model generalization without supporting controls and validation.

## Sources for later AWS implementation checks
- GitHub OIDC: https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws
- AWS EC2 public IPv4: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/working-with-ip-addresses.html
- AWS Systems Manager Session Manager: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html
- AWS ECS blue/green: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/migrate-codedeploy-to-ecs-bluegreen.html
