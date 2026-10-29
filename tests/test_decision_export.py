"""Synthetic exports and privacy guards; no patient files or model binaries."""

import json
import os
import subprocess
import sys

import pytest

from healthcare_ai_lab.experiments import export_decisions as exports
from healthcare_ai_lab.trajectories.decision_contracts import DecisionSummaryResponse


def example():
    return DecisionSummaryResponse.model_validate(
        exports.synthetic_example()["response"]
    )


def test_retrospective_adapter_selects_validation_replay_and_redacts_before_export(
    tmp_path, monkeypatch
):
    import pandas as pd

    from healthcare_ai_lab.experiments import verify_sepsis_bundle
    from healthcare_ai_lab.models.artifact import sha256_file

    toy = exports.synthetic_example()
    raw = pd.DataFrame(
        [
            {
                "patient_id": "synthetic-private-source",
                "ICULOS": row["icu_hour"],
                **row["measurements"],
            }
            for row in toy["request"]["observations"]
        ]
    )
    raw_path = tmp_path / "data/processed/sepsis_development.parquet"
    raw_path.parent.mkdir(parents=True)
    raw.to_parquet(raw_path, index=False)
    temporal_path = raw_path.with_name("sepsis_temporal_features.parquet")
    pd.DataFrame(
        {
            "patient_id": ["synthetic-private-source", "synthetic-reserved-test"],
            "split": ["validation", "test"],
        }
    ).to_parquet(temporal_path, index=False)
    artifact = tmp_path / "artifacts/models/prototype_bundle.joblib"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("synthetic placeholder; never deserialize")
    artifact.with_suffix(".manifest.json").write_text("{}")
    bundle = {
        "trajectory_metadata": {"model": toy["request"]["model"]},
        "artifact_manifest": {
            "source_hashes": {
                "raw": sha256_file(raw_path),
                "temporal": sha256_file(temporal_path),
            }
        },
    }
    monkeypatch.setattr(exports, "load_bundle", lambda p: bundle)
    calls = []

    def parity(*args):
        calls.append("parity")
        return {"passed": True}

    monkeypatch.setattr(verify_sepsis_bundle, "smoke_test", parity)

    class OfflineAPI:
        def __init__(self, app):
            pass

        def post(self, route, json):
            assert calls == ["parity"]
            assert len(json["observations"]) == 3
            assert "synthetic-private-source" not in str(json)
            assert "synthetic-reserved-test" not in str(json)
            assert all(
                row["episode_id"] == "retrospective-demo"
                for row in json["observations"]
            )

            class Response:
                status_code = 200

                def json(self):
                    return toy["response"]

            return Response()

    monkeypatch.setattr(exports, "TestClient", OfflineAPI)
    result, evidence = exports.retrospective_demo(tmp_path)
    assert evidence["matching_hours"] == 3
    assert result.payload_kind == "sanitized_retrospective_demo"
    assert all(r.measurements is None for r in result.records)
    assert "synthetic-private-source" not in result.model_dump_json()


def test_redaction_removes_identifiers_measurements_and_contributions():
    source = example()
    source.episode_id = "synthetic-private-id"
    before = source.model_dump()
    sanitized = exports.sanitize_demo(source)
    assert source.model_dump() == before
    assert sanitized.episode_id == "retrospective-demo"
    assert sanitized.payload_kind == "sanitized_retrospective_demo"
    assert all(
        r.measurements is None and r.log_odds_contributions is None
        for r in sanitized.records
    )
    assert [r.current for r in source.records] == [r.current for r in sanitized.records]
    text = sanitized.model_dump_json()
    assert (
        "synthetic-private-id" not in text
        and '"Age"' not in text
        and "current_value" not in text
    )
    assert "not anonymous" in text


def test_synthetic_example_and_schema_cli(tmp_path, monkeypatch, capsys):
    assert exports.synthetic_example() == exports.synthetic_example()
    for source in ("synthetic", "schema"):
        output = tmp_path / f"{source}.json"
        monkeypatch.setattr(
            "sys.argv", ["export", "--source", source, "--output", str(output)]
        )
        exports.main()
        result = json.loads(output.read_text())
        assert result
        assert os.stat(output).st_mode & 0o777 == 0o600
    assert "synthetic-demo" not in capsys.readouterr().out
    with pytest.raises(SystemExit, match="FileExistsError"):
        exports.main()


@pytest.mark.parametrize(
    "outside", ["docs/patient.json", "frontend/patient.json", "fixture.txt"]
)
def test_retrospective_output_must_stay_ignored(tmp_path, outside):
    with pytest.raises(exports.ExportError):
        exports.write_export(tmp_path / outside, {}, root=tmp_path, retrospective=True)
    assert not (tmp_path / outside).exists()


def test_output_exclusive_finite_and_symlink_escape(tmp_path):
    path = tmp_path / "results/dashboard/demo.json"
    exports.write_export(
        path, {"research_only": True}, root=tmp_path, retrospective=True
    )
    with pytest.raises(FileExistsError):
        exports.write_export(
            path, {"different": True}, root=tmp_path, retrospective=True
        )
    assert json.loads(path.read_text()) == {"research_only": True}
    with pytest.raises(ValueError):
        exports.write_export(
            tmp_path / "results/invalid.json",
            {"invalid": float("nan")},
            root=tmp_path,
            retrospective=True,
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "results/link").symlink_to(outside)
    with pytest.raises(exports.ExportError):
        exports.write_export(
            tmp_path / "results/link/demo.json", {}, root=tmp_path, retrospective=True
        )


def test_missing_and_mismatched_local_evidence_fail_before_replay(
    tmp_path, monkeypatch
):
    with pytest.raises(exports.ExportError, match="required"):
        exports.retrospective_demo(tmp_path)
    for name in (
        "artifacts/models/prototype_bundle.joblib",
        "artifacts/models/prototype_bundle.manifest.json",
        "data/processed/sepsis_development.parquet",
        "data/processed/sepsis_temporal_features.parquet",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic placeholder; never deserialize")
    monkeypatch.setattr(
        exports, "load_bundle", lambda p: {"artifact_manifest": {"source_hashes": {}}}
    )
    with pytest.raises(exports.ExportError, match="differs"):
        exports.retrospective_demo(tmp_path)


def test_retrospective_cli_uses_sanitized_summary_and_logs_only_aggregates(
    tmp_path, monkeypatch, capsys
):
    summary = exports.sanitize_demo(example())
    monkeypatch.setattr(
        exports, "retrospective_demo", lambda root: (summary, {"matching_hours": 3})
    )
    output = tmp_path / "results/demo.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export",
            "--source",
            "retrospective",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    exports.main()
    assert json.loads(output.read_text())["episode_id"] == "retrospective-demo"
    logged = json.loads(capsys.readouterr().out)
    assert logged["matching_hours"] == 3 and "episode_id" not in logged


def test_numerical_adapter_imports_without_agent_providers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import healthcare_ai_lab.trajectories.decisions; "
                "assert not any(n in sys.modules for n in ('langgraph','openai','anthropic'))"
            ),
        ],
        check=True,
        env={**os.environ, "PYTHON_DOTENV_DISABLED": "1"},
    )
