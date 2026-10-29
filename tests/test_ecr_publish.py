"""Offline release-boundary regressions; never invoke real AWS or Docker."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workflow():
    # BaseLoader preserves the YAML `on` key rather than treating it as a boolean.
    return yaml.load(
        (ROOT / ".github/workflows/publish-ecr.yml").read_text(), Loader=yaml.BaseLoader
    )


def test_manual_scope_permissions_and_failed_validation_block_publish(workflow):
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    for job in jobs.values():
        condition = job["if"]
        assert "github.event_name == 'workflow_dispatch'" in condition
        assert "github.repository == 'olveraalec/preventive-health-ai'" in condition
        assert "github.ref == 'refs/heads/feature/codex-integration'" in condition
        assert "continue-on-error" not in job
        assert all("continue-on-error" not in step for step in job["steps"])
        checkout = next(
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["with"] == {
            "ref": "${{ github.sha }}",
            "persist-credentials": "false",
        }
    assert jobs["publish"]["needs"] == "validate"
    assert "needs.validate.result == 'success'" in jobs["publish"]["if"]
    assert jobs["publish"]["permissions"] == {"contents": "read", "id-token": "write"}
    validation = jobs["validate"]["steps"]
    assert any(
        "pytest -q -ra --junitxml=" in step.get("run", "") for step in validation
    )
    assert any(
        "ruff check --config ruff-ci.toml ." in step.get("run", "")
        for step in validation
    )


@pytest.mark.parametrize(
    "override,allowed",
    [
        ({}, True),
        ({"GITHUB_REF": "refs/heads/main"}, False),
        ({"GITHUB_REPOSITORY": "someone/other"}, False),
        ({"GITHUB_EVENT_NAME": "pull_request"}, False),
        ({"AWS_REGION": "us-east-1"}, False),
        ({"AWS_ROLE_ARN": ""}, False),
        ({"AWS_ROLE_ARN": "not-a-role"}, False),
        ({"GITHUB_SHA": "branch-name"}, False),
    ],
)
def test_scope_and_variable_guard_executes_without_cloud_access(
    workflow, override, allowed
):
    env = {
        "PATH": os.environ["PATH"],
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": "olveraalec/preventive-health-ai",
        "GITHUB_REF": "refs/heads/feature/codex-integration",
        "GITHUB_SHA": "a" * 40,
        "AWS_REGION": "us-west-2",
        "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/synthetic-ecr",
    }
    command = workflow["jobs"]["validate"]["steps"][0]["run"]
    result = subprocess.run(
        ["bash", "-c", command],
        env={**env, **override},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is allowed
    assert env["AWS_ROLE_ARN"] not in result.stdout + result.stderr


def test_build_and_offline_verification_precede_aws_identity(workflow):
    steps = workflow["jobs"]["publish"]["steps"]
    build = next(
        i
        for i, step in enumerate(steps)
        if "docker buildx build" in step.get("run", "")
    )
    smoke = next(
        i for i, step in enumerate(steps) if "--network none" in step.get("run", "")
    )
    identity = next(
        i
        for i, step in enumerate(steps)
        if step.get("uses", "").startswith("aws-actions/configure-aws-credentials@")
    )
    login = next(
        i
        for i, step in enumerate(steps)
        if step.get("uses", "").startswith("aws-actions/amazon-ecr-login@")
    )
    push = next(i for i, step in enumerate(steps) if step.get("id") == "push")
    assert build < smoke < identity < login < push
    assert "--platform linux/amd64 --load" in steps[build]["run"]
    assert steps[identity]["with"]["role-to-assume"] == "${{ vars.AWS_ROLE_ARN }}"
    assert steps[identity]["with"]["aws-region"] == "${{ vars.AWS_REGION }}"
    assert steps[login]["with"] == {"registry-type": "private", "mask-password": "true"}
    assert workflow["env"]["LOCAL_IMAGE"] == "preventive-health-ai:${{ github.sha }}"


def test_embedded_shell_and_python_parse(workflow):
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            command = step.get("run", "")
            subprocess.run(["bash", "-n"], input=command, text=True, check=True)
            if "<<'PY'\n" in command:
                python = command.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
                compile(python, "workflow-heredoc", "exec")


@pytest.mark.parametrize(
    "digest,success",
    [("sha256:" + "b" * 64, True), ("None", False), ("sha256:bad", False)],
)
def test_push_reports_only_a_valid_remote_digest(workflow, tmp_path, digest, success):
    # Substitute executable stubs, so the actual workflow shell is exercised
    # without granting credentials, contacting ECR or publishing an image.
    for name, body in (
        ("docker", 'printf "%s\\n" "$*" >> "$CALL_LOG"'),
        ("aws", 'printf "%s\\n" "$FAKE_DIGEST"'),
    ):
        executable = tmp_path / name
        executable.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
        executable.chmod(0o700)
    output, summary, calls = (
        tmp_path / name for name in ("output", "summary", "calls")
    )
    env = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALL_LOG": str(calls),
        "FAKE_DIGEST": digest,
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
        "GITHUB_SHA": "a" * 40,
        "ECR_REGISTRY": "123456789012.dkr.ecr.us-west-2.amazonaws.com",
        "ECR_REPOSITORY": "preventive-health-ai",
        "LOCAL_IMAGE": "synthetic:local",
        "AWS_REGION": "us-west-2",
    }
    command = next(
        step["run"]
        for step in workflow["jobs"]["publish"]["steps"]
        if step.get("id") == "push"
    )
    result = subprocess.run(
        ["bash", "-c", command], env=env, capture_output=True, text=True, check=False
    )
    assert (result.returncode == 0) is success
    image = f"{env['ECR_REGISTRY']}/preventive-health-ai:{env['GITHUB_SHA']}"
    assert calls.read_text().splitlines() == [
        f"tag synthetic:local {image}",
        f"push {image}",
    ]
    if success:
        assert output.read_text() == f"image_uri={image}\ndigest={digest}\n"
        assert f"preventive-health-ai@{digest}" in summary.read_text()
    else:
        assert not output.exists() and not summary.exists()


def test_docker_context_is_default_deny_and_build_supplies_project_readme():
    lines = [
        line
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert lines[0] == "**"
    # Any broadened context requires review: no dataset/model/environment trees.
    file_rules = {
        "!Dockerfile",
        "!.dockerignore",
        "!pyproject.toml",
        "!uv.lock",
        "!README.md",
        "!src/**/*.py",
    }
    # File-only exceptions were verified with real Docker and synthetic binary,
    # patient, credential and bytecode sentinels. Directory exceptions admit
    # their descendants and must not be added to this security boundary.
    assert set(lines[1:]) == file_rules
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile
    assert "uv==0.11.26" in dockerfile and "--frozen" not in dockerfile


@pytest.mark.parametrize("failure", ["tag", "push", "describe"])
def test_publish_command_failures_never_emit_success(workflow, tmp_path, failure):
    for name, body in (
        ("docker", 'if [[ "$1" == "$FAIL_AT" ]]; then exit 9; fi'),
        (
            "aws",
            'if [[ "$FAIL_AT" == "describe" ]]; then exit 9; fi\nprintf "sha256:%064d\\n" 0',
        ),
    ):
        executable = tmp_path / name
        executable.write_text("#!/bin/bash\n" + body + "\n")
        executable.chmod(0o700)
    output, summary = tmp_path / "output", tmp_path / "summary"
    command = next(
        step["run"]
        for step in workflow["jobs"]["publish"]["steps"]
        if step.get("id") == "push"
    )
    result = subprocess.run(
        ["bash", "-c", command],
        env={
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "FAIL_AT": failure,
            "GITHUB_SHA": "a" * 40,
            "LOCAL_IMAGE": "synthetic:local",
            "ECR_REGISTRY": "synthetic.invalid",
            "ECR_REPOSITORY": "preventive-health-ai",
            "AWS_REGION": "us-west-2",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not output.exists() and not summary.exists()
