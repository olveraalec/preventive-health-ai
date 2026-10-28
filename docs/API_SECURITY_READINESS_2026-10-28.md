# Local database verification and API security readiness

Date: 2026-10-28. Research prototype only. This implementation-facing assessment
follows root `AGENTS.md` and the AWS/deployment ADRs in `markdowns/`; it does not
change those decisions or authorize public exposure. No cloud resources, DNS,
credentials, firewall rules, production databases or application routes were changed.

## Verification boundary

Docker context metadata established a local Unix socket before daemon access.
A newly named PostgreSQL 16 container used a random database name, an ephemeral
password kept in the verification process's memory, an automatically allocated
loopback-only host port, and no shared named volume. Neither existing PostgreSQL
instance nor the running API was restarted or queried. Existing SQL 001 and 002
were applied only to the new disposable database; no migration was created.

The initial opt-in test passed before changes. The expanded suite has 15 synthetic
live tests covering the existing round trip plus:

- Four-table inventory and episode column from both existing SQL files.
- Episode separation, empty retrieval, stable score/ID ordering and retrieval limits.
- Quoted/SQL-like episode identifiers handled as bound query parameters.
- Run-plus-selection persistence, order and rank round trips.
- Duplicate-selection and invalid-pathway failures after a successful first insert:
  the new run and all its selections roll back, while preexisting memories survive.
- Selection-only failure rolls back its inserts while preserving the existing run.
- Missing-run FK, duplicate ranks, positive rank and finite/nonnegative memory-score
  constraints; unknown-run retrieval returns no result.

The opt-in module now refuses implicit defaults or remote hosts: tests require
`POSTGRES_HOST=127.0.0.1`, `POSTGRES_USER=healthcare_test`, and `POSTGRES_DB` equal to
`healthcare_ai_test` or prefixed `healthcare_ai_test_`. This is a test safety guard,
not a new database/application interface. Existing runbook configuration satisfies it.

Final checks: targeted live tests **15 passed**, full suite with the disposable
database **151 passed, zero skipped**, and full suite with live tests disabled
**136 passed, 15 skipped** (all opt-in database cases). Scoped Ruff and
`git diff --check` passed. The disposable container was stopped and removed after
verification; its synthetic rows and temporary credential were not retained.

## Local network observations

Source and runtime observations agree: Compose publishes the API on
`127.0.0.1:8002` and PostgreSQL on `127.0.0.1:5433`. Another existing PostgreSQL
container publishes `127.0.0.1:5432`. Visible host listeners on audited ports
8000/8002/5432/5433/8888 were all loopback-bound; 8888 was loopback-only too.
The new test database also had an exclusively loopback Docker port binding.

Dockerfile's `0.0.0.0:8000` binds **inside** the container and does not override
Compose's host restriction. `EXPOSE 8000` does not itself publish a host port.
Do not run that image with an unrestricted `-p 8000:8000`, host networking, or
wildcard Uvicorn binding on the host. Local containers sharing a Docker bridge and
trusted local processes are still within the development trust boundary.

This is not an internet reachability scan or a cloud security-group audit. Docker
and the inspected listener list bound the observation; no public firewall or DNS
configuration was queried or changed.

## Security findings

| Area | Observed behavior and consequence | Required before external use |
| --- | --- | --- |
| Authentication | No authentication dependencies or application middleware on any API route. Controller, agent, stored-run and research inference routes accept unauthenticated calls. | Approved identity gate with negative tests; application authorization for stored-run/episode access before exposing persistence routes. UUIDs and episode names are not authorization. |
| Documentation | `/docs`, `/redoc`, `/openapi.json` return 200 without authentication. | Protect or hide at the exposure boundary without changing current development routes. |
| Host validation | A synthetic arbitrary Host header is accepted; no TrustedHost middleware. | Exact proxy host allowlist, then a separately reviewed application host policy. |
| Validation | Numeric/bounds/schema checks and strict trajectory contracts exist, but history/candidate lengths and many string sizes are unbounded; legacy engineered input is `dict[str, Any]`. | Measured request/body/concurrency limits; retain mathematical and NaN semantics when reviewing DTO bounds. |
| Validation privacy | Default 422 responses echo rejected input; verified with a synthetic invalid step. | Review response redaction before real data can enter an externally reachable endpoint; never log request/response bodies. |
| Logging | No application redaction policy or privacy-safe structured audit log. Uvicorn access logging defaults on; URL paths can contain run IDs. No runtime logs were inspected. | Disable raw access/body logging and implement allowlisted aggregate telemetry; review error logging and retention. |
| CORS | No CORS middleware; no allow-origin header for an untrusted synthetic Origin. This restricts browser reads but does not authenticate non-browser clients. | Prefer same-origin demo access; if needed, allow only exact approved origins/methods/headers, not wildcard credentialed CORS. |
| Proxy trust | Installed Uvicorn defaults enable proxy headers and default trusted peers to `127.0.0.1,::1` unless configured by environment. Container gateway peers may differ. | Explicit peer trust; overwrite forwarded headers at the ingress. Never use `FORWARDED_ALLOW_IPS=*` as a shortcut. |
| TLS | Current Compose/Uvicorn provide local plain HTTP; no TLS proxy is configured. | Approved trusted certificate, tested HTTPS and renewal; HTTPS does not provide authentication. |
| Readiness | `/health` is liveness. `/ready` always returns ready with `inference_enabled:false`; it checks no DB or model. | Add a separately reviewed capability-specific readiness gate, preserving the existing route contract. |
| Artifacts/image | Runtime validates activated local sidecars. The Dockerfile copies source/venv, not the verified bundle/manifest. Its base tags and uv installer are mutable. | Define immutable release/image+artifact delivery and startup verification. Do not assume the current image can serve model inference. |
| CI | Read-only permissions, locked install, scoped Ruff and explicit skips exist; GitHub execution remains unverified. No image/security scan or live DB CI job. | Hosted CI evidence, dependency/image scans and a separate ephemeral DB job after review. |

No application hardening was silently applied during this audit. Newly added
database tests and their Ruff scope do not change API behavior or inference.

## Database findings still requiring later review

The tested transaction boundary behaves correctly. Existing schema gaps remain:
nullable old memory episodes, no selection-to-memory/episode FK, no decision-time
availability filter, no migration ledger, and incomplete finite-value constraints
on selection/evaluation values. SQL initialization is manual in Compose. There is
no trajectory/outcome persistence or evaluation-results repository interface.
Database owner/bootstrap privileges must be separated from a future runtime role.
Any changes require separate schema approval; none were made here.

## Proposed EC2 boundary (design only)

See [the concrete Nginx example](examples/nginx/research-ec2.conf.example). It is
not an active configuration or an instruction to open ports. It proposes:

- TLS 1.2/1.3 for one approved hostname, default-host rejection and short initial
  HSTS only after certificate/renewal verification.
- One POST research-inference route; docs, agent, DB and controller routes stay
  blocked at this prospective public proxy while still existing locally.
- A required localhost bearer-token authorization sidecar using `auth_request`.
  It must validate signature/algorithm, issuer, audience, expiry and the constant
  `research:inference` permission. No such sidecar exists yet. Missing/error auth
  responses must deny service; never replace it with an always-200 stub.
- A 2 MiB body limit, 2 requests/second per peer with a burst of four, two concurrent
  connections per peer and explicit timeouts. These are proposed limits, not measured
  capacity or calibrated clinical thresholds; test shared-NAT behavior and valid histories.
- No raw URL, query string, address, authorization token, cookie or body in the
  proposed access log. Native Nginx errors are disabled because they can contain
  identifiers; a reviewed redacted diagnostic sink is still required.
- Overwritten forwarded/identity headers, no trusted incoming X-Forwarded-For chain,
  and local-only health/readiness access. No broad CORS policy is added.

The proposed backend is **native host Uvicorn**, not the existing Docker bridge:

```bash
# Design only; not launched by this task.
uv run --no-sync uvicorn healthcare_ai_lab.api.main:app \
  --host 127.0.0.1 --port 8000 --proxy-headers \
  --forwarded-allow-ips=127.0.0.1 --no-access-log
```

A containerized alternative must separately identify its actual trusted proxy peer;
do not copy the host-native trust setting blindly. The Nginx build must include
`http_ssl_module` and `http_auth_request_module` and support `ssl_reject_handshake`.
No Nginx executable is installed locally: `nginx -t`, TLS/auth integration, certificate
renewal, rate-limit/load tests and rollback behavior have **not** been verified.
The example uses reserved `.invalid` names and unavailable certificate paths; it
cannot be treated as a ready deployment. Authentication is a required missing component.

EC2 proposal: allow 443 only from the approved demo audience; 80 is optional solely
for approved redirect/issuance needs. Keep 22/8000/8002/5432/5433/8888 closed externally.
Prefer Session Manager over SSH. Define host/container firewall behavior alongside
security groups; database access is private/loopback. Use immutable releases with
the previous image/artifact retained. A single-instance replacement has a downtime
window and is not true blue/green.

## Next implementation milestone

Implement and test an opt-in **local-only** identity/access-control boundary plus
capability-specific readiness, preserving current development routes. Use synthetic
negative authentication/authorization tests, missing/corrupt-artifact readiness and
secondary-model fallback probes, and a local HTTPS proxy dry run. Also run the
existing CI workflow on GitHub when an authorized push is made. Public deployment
remains gated on auth, readiness, privacy-safe logging, immutable artifact delivery,
measured capacity, budget approval and explicit cloud authorization.

Official references checked for this design:
[Nginx auth_request](https://nginx.org/en/docs/http/ngx_http_auth_request_module.html),
[Nginx TLS](https://nginx.org/en/docs/http/ngx_http_ssl_module.html).
Future AWS design is separate in [AWS_FUTURE_ECS_ALB_DESIGN.md](AWS_FUTURE_ECS_ALB_DESIGN.md).
