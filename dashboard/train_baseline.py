"""Sepsis early-warning baseline: LightGBM on rolling-window features.

Run from the repo root:
    python src/train.py --data path/to/physionet.parquet            # full run
    python src/train.py --data path/to/physionet.csv --max-patients 3000   # quick test

Input: one row per patient-hour (PhysioNet 2019 columns: HR, O2Sat, Temp, SBP, MAP,
DBP, Resp, EtCO2, lab columns, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS,
SepsisLabel) plus a patient id column (patient_id / pid / id ...). CSV or Parquet.

Output: data/predictions.csv (test patients only) for dashboard/app.py, and printed
metrics: AUROC, AUPRC, and alert quality at several thresholds.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
DELTA_LABS = ["Lactate", "Creatinine", "WBC", "Platelets", "Bilirubin_total"]
DASH_VITALS = ["HR", "MAP", "Resp", "Temp"]  # what the dashboard plots
ID_CANDIDATES = ["patient_id", "patientid", "pid", "patient", "id", "file", "filename"]


# ------------------------------------------------------------------ loading
def load(path: str, max_patients: int | None, seed: int) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, sep="|" if p.suffix.lower() == ".psv" else ",")

    canon = {c.lower(): c for c in VITALS + LABS + STATIC + ["ICULOS", "SepsisLabel"]}
    df.columns = [canon.get(str(c).lower(), c) for c in df.columns]

    id_col = next((c for c in df.columns if str(c).lower() in ID_CANDIDATES), None)
    if id_col is None:
        raise SystemExit(f"No patient id column found. Columns: {list(df.columns)[:15]}...\n"
                         f"Rename yours to one of {ID_CANDIDATES}.")
    df = df.rename(columns={id_col: "patient_id"})
    if "SepsisLabel" not in df.columns:
        raise SystemExit("No SepsisLabel column found.")

    if max_patients:
        ids = df["patient_id"].drop_duplicates()
        if len(ids) > max_patients:
            keep = np.random.default_rng(seed).choice(ids.to_numpy(), max_patients, replace=False)
            df = df[df["patient_id"].isin(keep)]

    if "ICULOS" in df.columns:
        df["hour"] = df["ICULOS"]
    else:
        df["hour"] = df.groupby("patient_id").cumcount() + 1
    return df.sort_values(["patient_id", "hour"]).reset_index(drop=True)


# ----------------------------------------------------------------- features
def build_features(df: pd.DataFrame):
    """Returns (X, filled) where filled holds forward-filled vitals for plotting."""
    pid = df["patient_id"]
    hour = df["hour"].astype("float32")
    vitals = [c for c in VITALS if c in df]
    labs = [c for c in LABS if c in df]
    feats: dict[str, pd.Series] = {}

    for c in vitals + labs:
        feats[c] = df[c].groupby(pid).ffill()                     # last known value
        last_obs = hour.where(df[c].notna()).groupby(pid).ffill()
        feats[f"{c}_since"] = hour - last_obs                      # hours since measured

    for c in vitals:                                               # rolling means + trend
        x = feats[c]
        cs = x.fillna(0.0).groupby(pid).cumsum()
        cn = x.notna().astype("float32").groupby(pid).cumsum()
        for w in (6, 12):
            s = cs - cs.groupby(pid).shift(w).fillna(0.0)
            k = cn - cn.groupby(pid).shift(w).fillna(0.0)
            feats[f"{c}_mean{w}"] = s / k.replace(0, np.nan)
        feats[f"{c}_delta6"] = x - x.groupby(pid).shift(6)

    for c in DELTA_LABS:
        if c in feats:
            feats[f"{c}_delta6"] = feats[c] - feats[c].groupby(pid).shift(6)

    feats["ICULOS"] = hour
    for c in STATIC:
        if c in df:
            feats[c] = df[c]
    X = pd.DataFrame(feats).astype("float32")
    filled = pd.DataFrame({c: feats[c] for c in vitals})
    return X, filled


# --------------------------------------------------------------- evaluation
def alert_table(t: pd.DataFrame, thresholds) -> pd.DataFrame:
    """Patient-level alert quality, same logic as the dashboard."""
    first_lab = t[t["SepsisLabel"] == 1].groupby("patient_id")["hour"].min() + 6
    pats = pd.DataFrame({"onset": first_lab.reindex(t["patient_id"].unique())})
    septic = pats["onset"].notna()
    rows = []
    for thr in thresholds:
        fa = t[t["risk"] >= thr].groupby("patient_id")["hour"].min().reindex(pats.index)
        caught = septic & fa.notna() & (fa <= pats["onset"])
        lead = (pats["onset"] - fa)[caught]
        rows.append({"threshold": thr,
                     "caught_early": caught.sum() / max(septic.sum(), 1),
                     "median_lead_h": lead.median() if len(lead) else np.nan,
                     "false_alarm_rate": (~septic & fa.notna()).sum() / max((~septic).sum(), 1)})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV or Parquet, one row per patient-hour")
    ap.add_argument("--out", default="data/predictions.csv")
    ap.add_argument("--max-patients", type=int, default=None, help="subsample for quick tests")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading...")
    df = load(args.data, args.max_patients, args.seed)
    print(f"  {df['patient_id'].nunique():,} patients, {len(df):,} rows, "
          f"{df.groupby('patient_id')['SepsisLabel'].max().mean():.1%} develop sepsis")

    print("Building features...")
    X, filled = build_features(df)
    y = df["SepsisLabel"].astype(int)

    # split BY PATIENT so no patient appears in both train and test
    ptab = df.groupby("patient_id")["SepsisLabel"].max()
    ids, lab = ptab.index.to_numpy(), ptab.to_numpy()  # plain numpy: works with any id type
    train_ids, test_ids, lab_tr, _ = train_test_split(ids, lab, test_size=0.2, stratify=lab,
                                                      random_state=args.seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, stratify=lab_tr,
                                          random_state=args.seed)
    tr, va, te = (df["patient_id"].isin(s).to_numpy() for s in (train_ids, val_ids, test_ids))

    pos_weight = 10  # keeps risk scores in a usable 0-1 range for threshold sliders
    print(f"Training LightGBM on {tr.sum():,} rows...")
    model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               scale_pos_weight=pos_weight, n_jobs=-1, random_state=args.seed,
                               verbose=-1)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(40, verbose=False)])

    risk = model.predict_proba(X[te])[:, 1]
    t = df.loc[te, ["patient_id", "hour", "SepsisLabel"]].copy()
    t["risk"] = risk

    print(f"\nTest set: {len(test_ids):,} patients (held out by patient)")
    print(f"  AUROC (row-level): {roc_auc_score(y[te], risk):.3f}")
    print(f"  AUPRC (row-level): {average_precision_score(y[te], risk):.3f}")
    print("\nAlert quality by threshold (patient-level):")
    print(alert_table(t, [0.2, 0.3, 0.4, 0.5, 0.6]).to_string(
        index=False, formatters={"caught_early": "{:.0%}".format, "false_alarm_rate": "{:.0%}".format,
                                 "median_lead_h": "{:.1f}".format, "threshold": "{:.1f}".format}))

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop features:", ", ".join(imp.head(12).index))

    out = t[["patient_id", "hour", "risk", "SepsisLabel"]].copy()
    for c in DASH_VITALS:
        out[c] = filled.loc[te, c].to_numpy() if c in filled else np.nan
    out = out[["patient_id", "hour", "risk", *DASH_VITALS, "SepsisLabel"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out):,} rows to {args.out}  ->  python -m streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
