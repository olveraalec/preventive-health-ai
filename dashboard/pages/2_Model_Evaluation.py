"""Model evaluation & reliability dashboard (Streamlit + Plotly).

Put in  dashboard/pages/  next to 1_Adaptive_Decisions.py.
Reads data/eval_predictions.csv: one row per patient-hour with
    patient_id, hour, SepsisLabel, and one probability column per policy:
    static_probability, temporal_probability, hard_switch_probability (optional),
    continuous_probability (or `risk`)
Export HELD-OUT patients only (patient-disjoint). Without the file, labeled SIMULATED data is shown.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

EVAL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval_predictions.csv"
CANDIDATES = {"Static": ["static_probability"], "Temporal": ["temporal_probability"],
              "Hard-switching": ["hard_switch_probability"], "Continuous": ["continuous_probability", "risk"]}
COLORS = {"Static": "#6b7280", "Temporal": "#2563eb", "Hard-switching": "#7c3aed", "Continuous": "#d97706"}
LABEL, MIN_PATIENTS, EPS = "SepsisLabel", 20, 1e-12

st.set_page_config(page_title="Model evaluation", layout="wide")


def simulate(n: int = 300, seed: int = 11) -> pd.DataFrame:
    """Labeled synthetic patients. Continuous averages two independent noisy forecasts, so it can look
    better by construction. These numbers are not evidence about any real model."""
    rng, sig, frames = np.random.default_rng(seed), (lambda x: 1 / (1 + np.exp(-x))), []
    for k in range(n):
        T = int(rng.integers(24, 60)); h = np.arange(1, T + 1)
        pos = rng.random() < 0.15
        onset = int(rng.integers(12, T)) if pos else None
        base = 0.04 + (0.35 * sig((h - (onset - 8)) / 3) if pos else 0)
        s = np.clip(base + rng.normal(0, 0.05, T), 0.001, 0.999)
        t = np.clip(base + rng.normal(0, 0.05, T), 0.001, 0.999)
        frames.append(pd.DataFrame({
            "patient_id": f"SIM-{k}", "hour": h, LABEL: (h >= onset - 6).astype(int) if pos else 0,
            "static_probability": s, "temporal_probability": t,
            "hard_switch_probability": np.where(np.abs(t - s) > 0.05, t, s),
            "continuous_probability": s + 0.3 * (t - s)}))
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load(stamp: float = 0):  # stamp = file modified time, refreshes the cache
    if not EVAL_PATH.exists():
        return simulate(), "SIMULATED (data/eval_predictions.csv not found)", None
    df = pd.read_csv(EVAL_PATH)
    need = {"patient_id", LABEL} - set(df.columns)
    return (df, f"Experimental file ({EVAL_PATH.name})", f"Missing columns: {sorted(need)}" if need else None)


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ap(p, y):  # average precision; ties are broken by order, so it is approximate
    if y.sum() in (0, len(y)):
        return np.nan
    yy = y[np.argsort(-p, kind="mergesort")]
    return float((np.cumsum(yy) / np.arange(1, len(yy) + 1))[yy == 1].mean())


f = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.4f}"

df, src, err = load(EVAL_PATH.stat().st_mtime if EVAL_PATH.exists() else 0)
st.title("Model evaluation and reliability")
st.caption(f"Data: {src}. Retrospective evaluation against the shifted PhysioNet Challenge label. Research use only.")
if err:
    st.error(err); st.stop()
sim = src.startswith("SIMULATED")
if sim:
    st.warning("SIMULATED DATA. These numbers are not experimental results. Blending two independent noisy forecasts improves loss by construction here, so a Continuous gain in this demo means nothing.")

cols = {n: next((c for c in cs if c in df.columns), None) for n, cs in CANDIDATES.items()}
cols = {n: c for n, c in cols.items() if c}
missing_pol = [n for n in CANDIDATES if n not in cols]
if len(cols) < 2:
    st.error("Need at least two policy probability columns."); st.stop()
if missing_pol:
    st.info(f"Not available in this file: {', '.join(missing_pol)}.")

base = st.sidebar.selectbox("Baseline policy", list(cols), index=0)
B = st.sidebar.slider("Bootstrap resamples", 200, 4000, 2000, 200)

use = df.dropna(subset=[*cols.values(), LABEL])
y = use[LABEL].to_numpy(int)
pm = pd.DataFrame({n: logloss(use[c].to_numpy(), y) for n, c in cols.items()}).groupby(use["patient_id"].to_numpy()).mean()
n_pat = len(pm)
pos_pat = use.groupby("patient_id")[LABEL].max().reindex(pm.index).to_numpy() == 1
enough = n_pat >= MIN_PATIENTS
st.caption(f"{n_pat} patients, {len(use):,} patient-hours, {int(y.sum()):,} positive hours "
           f"({int(pos_pat.sum())} positive patients). {len(df) - len(use):,} rows dropped for a missing policy output.")
if not enough:
    st.info(f"Confidence intervals are Not available: need at least {MIN_PATIENTS} patients.")

rng = np.random.default_rng(0)
Bn = max(200, min(B, 20_000_000 // max(n_pat, 1)))
idx = rng.integers(0, n_pat, size=(Bn, n_pat), dtype=np.int32) if enough else None
ci = lambda v: tuple(np.percentile(v[idx].mean(axis=1), [2.5, 97.5])) if enough else (np.nan, np.nan)

rows = []
for n, c in cols.items():
    p, v = use[c].to_numpy(), pm[n].to_numpy()
    lo, hi = ci(v)
    k = max(1, int(np.ceil(0.1 * len(v))))
    rows.append({"Policy": n, "Patient-mean log loss": f(v.mean()),
                 "95% CI (patient bootstrap)": f"[{f(lo)}, {f(hi)}]" if enough else "Not available",
                 "Hourly log loss": f(logloss(p, y).mean()), "Brier": f(((p - y) ** 2).mean()), "AUPRC": f(ap(p, y)),
                 "Worst patient": f(v.max()), "CVaR90 (worst 10% mean)": f(np.sort(v)[::-1][:k].mean())})
st.subheader("Policy metrics (lower is better, except AUPRC)")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.subheader(f"Patient-level change vs {base} (negative = lower log loss)")
rows, fig = [], go.Figure()
for n in cols:
    if n == base:
        continue
    d = pm[n].to_numpy() - pm[base].to_numpy()
    lo, hi = ci(d)
    verdict = "Not available" if not enough else ("Interval below 0" if hi < 0 else "Interval above 0" if lo > 0 else "Interval includes 0")
    rows.append({"Policy": n, "Mean change": f(d.mean()), "95% CI": f"[{f(lo)}, {f(hi)}]" if enough else "Not available",
                 "Interval vs 0": verdict, "Patients improved": f"{(d < 0).mean():.0%}",
                 "Worst-patient change": f(d.max()),
                 "Mean change, positive patients": f(d[pos_pat].mean()) if pos_pat.any() else "n/a"})
    fig.add_trace(go.Box(y=d, name=n, marker_color=COLORS[n], boxpoints="outliers"))
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
fig.add_hline(y=0, line_color="gray", line_dash="dash")
fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title="Patient log-loss change")
st.plotly_chart(fig, width="stretch")

st.subheader("Calibration (equal-count bins)")
cal = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dash"), name="Perfect"))
for n, c in cols.items():
    g = use.assign(bin=pd.qcut(use[c], 10, duplicates="drop")).groupby("bin", observed=True).agg(p=(c, "mean"), o=(LABEL, "mean"), n=(LABEL, "size"))
    cal.add_trace(go.Scatter(x=g["p"], y=g["o"], mode="lines+markers", name=n, line=dict(color=COLORS[n]),
                             customdata=g["n"], hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<br>n=%{customdata}"))
cal.update_layout(height=380, margin=dict(t=10, b=10), xaxis_title="Mean predicted probability", yaxis_title="Observed label rate")
st.plotly_chart(cal, width="stretch")

with st.expander("Definitions and limitations"):
    st.markdown(
        "- **Patient-mean log loss** averages each patient's hourly Bernoulli log loss, then averages patients. Intervals are percentile bootstrap over patients.\n"
        "- **CVaR90** is the mean patient loss in the worst 10% of patients. **Worst patient** is the single highest.\n"
        "- **Positive patients** are those with at least one label of 1.\n"
        "- AUPRC uses hourly rows and breaks ties by order, so it is approximate.\n"
        "- Rows missing any policy output are dropped for all policies, so comparisons use identical rows.\n"
        "- These numbers support a conclusion only if the file contains held-out, patient-disjoint patients and the comparison was fixed before looking. "
        "Null or negative results are shown as they are. No multiple-comparison adjustment is applied.\n"
        "- Retrospective agreement with the shifted label is not clinical impact.")