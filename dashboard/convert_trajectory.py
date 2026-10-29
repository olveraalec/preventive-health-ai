"""Convert a saved trajectory JSON into data/predictions.csv for dashboard/app.py.

    python dashboard/convert_trajectory.py saved.json
    python dashboard/convert_trajectory.py saved.json --labels labels.csv --append

saved.json: {"request": <RawTrajectoryRequest>, "response": <RawTrajectoryResponse>}
labels.csv (optional): columns  hour,SepsisLabel  - retrospective labels for the
same episode, used only for evaluation overlays. Without it the dashboard shows
no onset markers and no detection metrics.
Unobserved measurements stay empty; nothing is imputed.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

VITALS = ["HR", "MAP", "Resp", "Temp"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--labels")
    ap.add_argument("--out", default="data/predictions.csv")
    ap.add_argument("--append", action="store_true", help="add this episode to an existing output file")
    a = ap.parse_args()

    j = json.loads(Path(a.json_file).read_text(encoding="utf-8"))
    req, resp = j["request"], j["response"]
    pred = pd.DataFrame(resp["predictions"]).rename(
        columns={"icu_hour": "hour", "final_probability": "risk"})
    obs = pd.DataFrame([
        {"hour": o["icu_hour"], **{v: o["measurements"].get(v) for v in VITALS}}
        for o in req["observations"]])
    df = pred.merge(obs, on="hour", how="left")
    df.insert(0, "patient_id", resp["episode_id"])
    if a.labels:
        lab = pd.read_csv(a.labels)[["hour", "SepsisLabel"]]
        df = df.merge(lab, on="hour", how="left")
    keep = ["patient_id", "hour", "risk", "static_probability", "temporal_probability",
            "temporal_weight", "status", *VITALS] + (["SepsisLabel"] if a.labels else [])
    df = df[keep]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.append and out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows ({df['patient_id'].nunique()} episodes) to {out}")


if __name__ == "__main__":
    main()
