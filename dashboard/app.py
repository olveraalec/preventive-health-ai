"""Sepsis early-warning dashboard (Streamlit + Plotly).

Place this file at  dashboard/app.py  in the repo and run from the repo root:
    python -m streamlit run dashboard/app.py

It reads  data/predictions.csv  if that file exists; otherwise it shows
simulated patients so the layout works before the model is ready.

predictions.csv columns (one row per patient-hour):
    patient_id, hour, risk, HR, MAP, Resp, Temp, SepsisLabel
  - risk:        model's predicted sepsis risk (0 to 1)
  - SepsisLabel: PhysioNet label (1 from 6h before onset). Optional if you
                 supply an  onset_hour  column instead (same value per patient).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PRED_PATH = Path(__file__).resolve().parent.parent / "data" / "predictions.csv"
VITALS = {
    "HR": "Heart rate (bpm)",
    "MAP": "Mean arterial pressure (mmHg)",
    "Resp": "Respiratory rate (/min)",
    "Temp": "Temperature (°C)",
}

st.set_page_config(page_title="Sepsis Early Warning", layout="wide")


# ----------------------------------------------------------------- data
def simulate(n_patients: int = 40, seed: int = 7) -> pd.DataFrame:
    """Fake ICU patients: ~40% develop sepsis, vitals drift and risk rises before onset."""
    rng = np.random.default_rng(seed)
    sig = lambda x: 1 / (1 + np.exp(-x))
    frames = []
    for pid in range(1, n_patients + 1):
        n = int(rng.integers(48, 96))
        h = np.arange(n)
        septic = rng.random() < 0.4
        onset = int(rng.integers(24, n - 6)) if septic else None
        d = sig((h - (onset - 10)) / 4) if septic else np.zeros(n)
        r = sig((h - (onset - 8)) / 3) if septic else np.zeros(n)
        risk = 0.06 + 0.8 * r + rng.normal(0, 0.03, n)
        if not septic:  # occasional harmless bumps -> possible false alarms
            risk += 0.25 * sig((h - rng.integers(10, n)) / 2) * (rng.random() < 0.3)
        risk = pd.Series(risk).rolling(3, min_periods=1).mean().clip(0, 1)
        frames.append(pd.DataFrame({
            "patient_id": pid,
            "hour": h,
            "risk": risk,
            "HR": 76 + rng.normal(0, 4) + 35 * d + rng.normal(0, 3, n),
            "MAP": 86 + rng.normal(0, 4) - 25 * d + rng.normal(0, 3, n),
            "Resp": 16 + rng.normal(0, 1) + 10 * d + rng.normal(0, 1.2, n),
            "Temp": 36.8 + 1.3 * d + rng.normal(0, 0.15, n),
            "SepsisLabel": (h >= onset - 6).astype(int) if septic else 0,
        }))
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_data():
    if PRED_PATH.exists():
        df, source = pd.read_csv(PRED_PATH), f"Real predictions ({PRED_PATH.name})"
    else:
        df, source = simulate(), "SIMULATED patients (data/predictions.csv not found)"
    df = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    has_labels = "SepsisLabel" in df.columns or "onset_hour" in df.columns
    if "onset_hour" not in df.columns:
        if "SepsisLabel" in df.columns:
            first = df[df["SepsisLabel"] == 1].groupby("patient_id")["hour"].min() + 6
            df["onset_hour"] = df["patient_id"].map(first)  # NaN if never septic
        else:
            df["onset_hour"] = np.nan
    return df, source, has_labels


def patient_table(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    """One row per patient: onset hour, first alert hour at this threshold, lead time."""
    onset = df.groupby("patient_id")["onset_hour"].first()
    first_alert = df[df["risk"] >= thr].groupby("patient_id")["hour"].min()
    t = pd.DataFrame({"onset": onset})
    t["first_alert"] = first_alert.reindex(t.index)
    t["septic"] = t["onset"].notna()
    t["caught"] = t["septic"] & t["first_alert"].notna() & (t["first_alert"] <= t["onset"])
    t["lead"] = (t["onset"] - t["first_alert"]).where(t["caught"])
    t["false_alarm"] = ~t["septic"] & t["first_alert"].notna()
    return t


# ----------------------------------------------------------------- UI
df, source, has_labels = load_data()

st.title("ICU Retrospective Research Dashboard")
st.caption(f"Data: {source}. Research use only; not for clinical use.")
sim = source.startswith("SIMULATED")
sfx = " (simulated)" if sim else ""
if sim:
    st.warning("SIMULATED DATA. These patients have a built-in signal, so the detection and false-alarm rates below say nothing about the model.")

with st.sidebar:
    st.header("Controls")
    thr = st.slider("Alert threshold (risk score)", 0.05, 0.95, 0.40, 0.05,
                    help="An alert fires the first hour risk is at or above this value.")
    st.caption("Lower = earlier alerts and more cases caught, but more false alarms.")

table = patient_table(df, thr)
septic, nonsep = table[table["septic"]], table[~table["septic"]]
if not has_labels:
    septic = nonsep = table.iloc[0:0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", len(table))
c2.metric(f"Labeled cases alerted before onset{sfx}",
          f"{septic['caught'].mean():.0%}" if len(septic) else "n/a",
          help="Share of septic patients alerted before (or at) onset.")
c3.metric(f"Median lead time{sfx}",
          f"{septic['lead'].median():.0f} h" if septic["caught"].any() else "n/a",
          help="Hours between the first alert and sepsis onset, for caught cases.")
c4.metric(f"False-alarm rate{sfx}",
          f"{nonsep['false_alarm'].mean():.0%}" if len(nonsep) else "n/a",
          help="Share of non-septic patients who triggered at least one alert.")

if sim:
    st.caption("Simulated metrics are not model performance.")
elif has_labels:
    st.caption(f"Computed on {len(table)} loaded retrospective episode(s) at this threshold. Not a performance estimate.")
if not has_labels:
    st.info("No retrospective labels in this file, so detection metrics and onset markers are not available.")
st.divider()

# patient picker: septic patients first
order = table.sort_values("septic", ascending=False).index.tolist()
pid = st.selectbox(
    "Select patient", order,
    format_func=lambda p: f"Patient {p}  " + (("(developed sepsis)" if table.loc[p, "septic"] else "(no sepsis)") if has_labels else ""),
)
g = df[df["patient_id"] == pid]
row = table.loc[pid]

# headline for this patient
if not has_labels:
    if pd.notna(row["first_alert"]):
        st.warning(f"Estimate first reached the threshold at hour {row['first_alert']:.0f}. No retrospective labels loaded.")
    else:
        st.info("Estimate stayed below the threshold. No retrospective labels loaded.")
elif row["septic"] and row["caught"]:
    st.success(f"Alert fired at hour {row['first_alert']:.0f}, **{row['lead']:.0f} hours before** "
               f"sepsis onset (hour {row['onset']:.0f}).")
elif row["septic"] and pd.notna(row["first_alert"]):
    st.warning(f"Alert fired at hour {row['first_alert']:.0f}, after onset (hour {row['onset']:.0f}). "
               "Try lowering the threshold.")
elif row["septic"]:
    st.error(f"No alert at this threshold; sepsis onset was hour {row['onset']:.0f}. Try lowering the threshold.")
elif pd.notna(row["first_alert"]):
    st.warning(f"False alarm: alert fired at hour {row['first_alert']:.0f} but the patient never developed sepsis.")
else:
    st.info("No alerts: patient stayed below the threshold and did not develop sepsis.")

# chart: risk on top, vitals underneath, shared time axis
fig = make_subplots(rows=1 + len(VITALS), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                    row_heights=[0.38] + [0.155] * len(VITALS),
                    subplot_titles=["Estimated probability of shifted challenge label"] + list(VITALS.values()))
fig.add_trace(go.Scatter(x=g["hour"], y=g["risk"], mode="lines", name="Final",
                         line=dict(color="#d62728", width=3)), row=1, col=1)
fig.add_hline(y=thr, line_dash="dash", line_color="gray", row=1, col=1,
              annotation_text=f"Threshold {thr:.2f}", annotation_position="top left")
for col, nm, color in [("static_probability", "Static", "#6b7280"), ("temporal_probability", "Temporal", "#2563eb")]:
    if col in g.columns:
        fig.add_trace(go.Scatter(x=g["hour"], y=g[col], mode="lines", name=nm,
                                 line=dict(color=color, width=2, dash="dot")), row=1, col=1)
if pd.notna(row["first_alert"]):
    fig.add_trace(go.Scatter(x=[row["first_alert"]], y=[g.loc[g["hour"] == row["first_alert"], "risk"].iloc[0]],
                             mode="markers", name="Alert fired",
                             marker=dict(size=14, color="orange", symbol="triangle-up",
                                         line=dict(width=1, color="black"))), row=1, col=1)
for i, col in enumerate(VITALS, start=2):
    fig.add_trace(go.Scatter(x=g["hour"], y=g[col], mode="lines", name=col, showlegend=False,
                             line=dict(width=2)), row=i, col=1)
if row["septic"]:
    for i in range(1, 2 + len(VITALS)):
        extra = {"annotation_text": "Sepsis onset"} if i == 1 else {}  # no blank annotations
        fig.add_vline(x=row["onset"], line_dash="dot", line_color="black", row=i, col=1, **extra)
fig.update_yaxes(range=[0, 1], row=1, col=1)
fig.update_xaxes(title_text="Hours since ICU admission", row=1 + len(VITALS), col=1)
fig.update_layout(height=760, margin=dict(t=40, b=20), legend=dict(orientation="h", y=1.06))
st.plotly_chart(fig, width="stretch")

st.caption("Research prototype built on retrospective PhysioNet 2019 data. Not clinically validated; "
           "intended to support, not replace, clinician judgment.")
