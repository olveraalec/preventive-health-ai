"""Offline checks for the manual image-publication security contract."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow():
    # BaseLoader retains GitHub's `on` key instead of YAML 1.1 boolean coercion.
    return yaml.load(
        (ROOT / ".github/workflows/publish-ecr.yml").read_text(),
        Loader=yaml.BaseLoader,
    )


def test_only_manual_dispatch_and_authorized_branch_can_publish():
    config = workflow()
    assert set(config["on"]) == {"workflow_dispatch"}
    publish = config["jobs"]["publish"]
    assert publish["needs"] == "validation"
    assert publish["if"] == (
        "github.event_name == 'workflow_dispatch' && "
        "github.ref == 'refs/heads/feature/codex-integration'"
    )
    assert "environment" not in publish  # Keep branch-based OIDC subject.


def test_oidc_permission_is_scoped_to_publish_job():
    config = workflow()
    assert config["permissions"] == {"contents": "read"}
    assert "permissions" not in config["jobs"]["validation"]
    assert config["jobs"]["publish"]["permissions"] == {
        "contents": "read",
        "id-token": "write",
    }
    for job in config["jobs"].values():
        for step in job["steps"]:
            assert "continue-on-error" not in step
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] == "false"


def test_full_tests_and_lint_gate_publication():
    steps = workflow()["jobs"]["validation"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "uv sync --locked --group dev" in commands
    assert "uv run --no-sync pytest -q -ra" in commands
    assert "ruff check --config ruff-ci.toml ." in commands
    assert workflow()["env"]["RUN_DB_TESTS"] == "0"


def test_build_and_smoke_gates_precede_credentials_and_push():
    steps = workflow()["jobs"]["publish"]["steps"]
    build = next(i for i, s in enumerate(steps) if "buildx build" in s.get("run", ""))
    smoke = next(i for i, s in enumerate(steps) if "docker image inspect" in s.get("run", ""))
    auth = next(i for i, s in enumerate(steps) if s.get("uses", "").startswith("aws-actions/configure-aws-credentials@"))
    push = next(i for i, s in enumerate(steps) if "docker push" in s.get("run", ""))
    assert build < smoke < auth < push
    assert "--platform linux/amd64 --load" in steps[build]["run"]
    assert "linux/amd64" in steps[smoke]["run"]
    assert "--network none" in steps[smoke]["run"]
    assert "/health" in steps[smoke]["run"]


def test_immutable_sha_tag_digest_and_configuration_contract():
    config = workflow()
    publish = config["jobs"]["publish"]
    assert config["env"]["ECR_REPOSITORY"] == "preventive-health-ai"
    assert publish["env"]["IMAGE_TAG"] == "${{ github.sha }}"
    assert publish["env"]["AWS_REGION"] == "${{ vars.AWS_REGION }}"
    assert publish["env"]["AWS_ROLE_ARN"] == "${{ vars.AWS_ROLE_ARN }}"
    commands = "\n".join(s.get("run", "") for s in publish["steps"])
    assert '"$AWS_REGION" == us-west-2' in commands
    assert '"$mutability" == IMMUTABLE' in commands
    assert "imageDetails[0].imageDigest" in commands
    assert "$REPOSITORY_URI@$digest" in commands
    assert ":latest" not in commands


def test_build_context_is_source_only_and_packaging_readme_is_available():
    rules = [
        line for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert rules[0] == "**"
    assert {rule for rule in rules if rule.startswith("!")} == {
        "!Dockerfile", "!.dockerignore", "!pyproject.toml", "!uv.lock",
        "!README.md", "!src/", "!src/**/", "!src/**/*.py",
    }
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile
    assert "UV_NO_DEV=1" in dockerfile
    assert "uv sync --frozen" in dockerfile
    assert "COPY . " not in dockerfile
