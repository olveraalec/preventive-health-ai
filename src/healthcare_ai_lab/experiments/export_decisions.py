"""Local-only dashboard fixtures. Never export source IDs or physiological values."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from healthcare_ai_lab.models.artifact import manifest_path, sha256_file
from healthcare_ai_lab.models.sepsis_inference import load_bundle
from healthcare_ai_lab.preprocessing.sepsis import RAW_REQUIRED
from healthcare_ai_lab.trajectories.decision_contracts import DecisionSummaryResponse
from healthcare_ai_lab.trajectories.decisions import summarize_raw_trajectory
from healthcare_ai_lab.trajectories.raw import RawTrajectoryRequest, runtime_provenance


class ExportError(ValueError):
    """Static diagnostics safe for console output."""


class _SyntheticExpert:
    def __init__(self, temporal=False):
        self.temporal = temporal

    def predict_proba(self, frame):
        # Fixed demonstration arithmetic, never fitted or used as a Sepsis model.
        hour = frame["ICULOS"].to_numpy()
        probability = (0.4 + 0.02 * hour) if self.temporal else (0.1 + 0.01 * hour)
        return np.c_[1 - probability, probability]


class _SyntheticGain:
    def predict(self, frame):
        return np.zeros(len(frame))


def synthetic_example():
    model = {"artifact_id": "synthetic-example", "model_version": "synthetic-v1"}
    measurements = dict.fromkeys(RAW_REQUIRED, None)
    measurements.update(
        Age=55, Gender=1, HospAdmTime=-12, Unit1=1, HR=80, MAP=75, Resp=18
    )
    request = RawTrajectoryRequest.model_validate(
        {
            "dataset": "sepsis2019",
            "episode_id": "synthetic-demo",
            "model": model,
            "observations": [
                {
                    "episode_id": "synthetic-demo",
                    "icu_hour": hour,
                    "measurements": {**measurements, "HR": hr},
                }
                for hour, hr in ((1, 80), (2, 90), (4, None))
            ],
        }
    )
    bundle = {
        "static": {"features": ["ICULOS"], "model": _SyntheticExpert()},
        "temporal": {"features": ["ICULOS"], "model": _SyntheticExpert(temporal=True)},
        "controller": _SyntheticGain(),
        "meta_features": ["ICULOS", "entropy_static"],
        "trajectory_metadata": {
            "dataset": "sepsis2019",
            "model": model,
            "provenance": runtime_provenance().model_dump(),
        },
    }
    response = summarize_raw_trajectory(bundle, request)
    response.payload_kind = "synthetic_example"
    return {"request": request.model_dump(), "response": response.model_dump()}


def sanitize_demo(response: DecisionSummaryResponse) -> DecisionSummaryResponse:
    # A typed, field-based projection. No source ID, raw value or invertible
    # log-odds feature contribution belongs in a dashboard export.
    data = response.model_dump()
    data["episode_id"] = "retrospective-demo"
    data["payload_kind"] = "sanitized_retrospective_demo"
    for record in data["records"]:
        record["measurements"] = None
        record["log_odds_contributions"] = None
    data["limitations"] = [
        *data["limitations"],
        "Export omits measurements and feature contributions; derived patient trajectories remain sensitive, not anonymous.",
    ]
    return DecisionSummaryResponse.model_validate(data)


def retrospective_demo(root: Path) -> tuple[DecisionSummaryResponse, dict]:
    """Reverify the prior validation replay, then export only derived summaries."""
    from healthcare_ai_lab.api.main import app
    from healthcare_ai_lab.experiments.verify_sepsis_bundle import smoke_test

    artifact = root / "artifacts/models/prototype_bundle.joblib"
    raw_path = root / "data/processed/sepsis_development.parquet"
    temporal_path = root / "data/processed/sepsis_temporal_features.parquet"
    if not all(
        p.is_file()
        for p in (artifact, manifest_path(artifact), raw_path, temporal_path)
    ):
        raise ExportError(
            "Trusted model/manifest and verified raw/engineered evidence are required"
        )
    bundle = load_bundle(artifact)
    manifest = bundle.get("artifact_manifest")
    if (
        manifest is None
        or manifest["source_hashes"].get("raw") != sha256_file(raw_path)
        or manifest["source_hashes"].get("temporal") != sha256_file(temporal_path)
    ):
        raise ExportError("Local evidence differs from the verified manifest")
    parity = smoke_test(bundle, raw_path, temporal_path, artifact)
    # Same deterministic validation-only choice as the already-verified smoke.
    cohort = pd.read_parquet(
        temporal_path, columns=["patient_id"], filters=[("split", "==", "validation")]
    )
    if cohort.empty:
        raise ExportError("No verified validation replay available")
    patient = min(cohort.patient_id.unique())
    raw = pd.read_parquet(
        raw_path,
        columns=["patient_id", "ICULOS", *RAW_REQUIRED],
        filters=[("patient_id", "==", patient)],
    )
    raw = raw.sort_values("ICULOS")
    payload = {
        "dataset": "sepsis2019",
        "episode_id": "retrospective-demo",
        "model": bundle["trajectory_metadata"]["model"],
        "observations": [
            {
                "episode_id": "retrospective-demo",
                "icu_hour": int(row["ICULOS"]),
                "measurements": {
                    name: None if pd.isna(row[name]) else row[name]
                    for name in RAW_REQUIRED
                },
            }
            for row in raw.to_dict("records")
        ],
    }
    with patch("healthcare_ai_lab.api.main.get_research_bundle", return_value=bundle):
        response = TestClient(app).post(
            "/api/v1/research/sepsis/trajectory/decision-summary?include_contributions=true",
            json=payload,
        )
    if response.status_code != 200:
        raise ExportError("Decision-summary API replay failed")
    full = DecisionSummaryResponse.model_validate(response.json())
    attributions = [
        a
        for record in full.records
        for a in (record.log_odds_contributions or {}).values()
    ]
    summary = sanitize_demo(full)
    residuals = [
        abs(r.change.identity_residual)
        for r in summary.records
        if r.change.identity_residual is not None
    ]
    return summary, {
        "matching_hours": len(summary.records),
        "prior_replay_parity": parity,
        "max_abs_decomposition_residual": max(residuals, default=0.0),
        "available_log_odds_explanations": sum(
            a.status == "available" for a in attributions
        ),
    }


def write_export(path: Path, data: dict, *, root: Path, retrospective=False):
    destination = path.resolve()
    if retrospective and not destination.is_relative_to((root / "results").resolve()):
        raise ExportError(
            "Retrospective exports must remain under the Git-ignored results directory"
        )
    if destination.suffix != ".json":
        raise ExportError("Export requires a new JSON file")
    serialized = json.dumps(data, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("synthetic", "retrospective", "schema"),
        default="synthetic",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        evidence = {}
        if args.source == "retrospective":
            summary, evidence = retrospective_demo(root)
            data = summary.model_dump()
        elif args.source == "schema":
            data = DecisionSummaryResponse.model_json_schema()
        else:
            data = synthetic_example()
        output = args.output if args.output.is_absolute() else root / args.output
        write_export(
            output, data, root=root, retrospective=args.source == "retrospective"
        )
    except ExportError as exc:
        raise SystemExit(f"Export refused: {exc}") from None
    except Exception as exc:  # noqa: BLE001 -- never print patient-bearing tracebacks
        raise SystemExit(
            f"Export failed ({type(exc).__name__}); no patient details displayed"
        ) from None
    print(
        json.dumps(
            {
                "exported": True,
                "source": args.source,
                "research_only": True,
                **evidence,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
