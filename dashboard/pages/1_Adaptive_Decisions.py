"""Adaptive AI decision dashboard: static vs temporal, controller allocation,
fallback, model disagreement and predictive entropy.

Put this file in  dashboard/pages/  next to dashboard/app.py. It reads data/predictions.csv
written by dashboard/convert_trajectory.py (or shows labeled SIMULATED episodes).
Required columns: patient_id, hour, risk (final), static_probability,
temporal_probability, temporal_weight, status.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PRED_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "predictions.csv"
REQUIRED = ["patient_id", "hour", "risk", "static_probability", "temporal_probability",
            "temporal_weight", "status"]
COLORS = {"Stable": "#16a34a", "Adapting": "#d97706", "Fallback": "#dc2626"}

st.set_page_config(page_title="Adaptive AI decisions", layout="wide")


def simulate() -> pd.DataFrame:
    """Hand-made episodes to show the layout. Not model behavior."""
    rng = np.random.default_rng(3)
    sig = lambda x: 1 / (1 + np.exp(-x))
    rows = []
    for k in range(1, 4):
        for h in range(1, 49):
            s = float(np.clip(0.05 + 0.25 * sig((h - 28 - 3 * k) / 5) + rng.normal(0, 0.008), 0.001, 0.999))
            fb = 34 <= h <= 38
            t = None if fb else float(np.clip(s + 0.06 * np.sin(h / 4 + k) + rng.normal(0, 0.01), 0.001, 0.999))
            q = 0.0 if fb else float(np.clip(0.1 + 0.4 * sig((h - 18) / 2), 0, 0.6))
            rows.append({"patient_id": f"SIM-{k}", "hour": h, "static_probability": round(s, 4),
                         "temporal_probability": None if t is None else round(t, 4),
                         "temporal_weight": round(q, 4),
                         "status": "static_only_fallback" if fb else "bounded_anticipatory",
                         "risk": round(s if t is None else s + q * (t - s), 6)})
    return pd.DataFrame(rows)


@st.cache_data
def load():
    if PRED_PATH.exists():
        df, src = pd.read_csv(PRED_PATH), f"Real predictions ({PRED_PATH.name})"
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            return None, src, missing
    else:
        df, src = simulate(), "SIMULATED (data/predictions.csv not found)"
    return df.sort_values(["patient_id", "hour"]).reset_index(drop=True), src, []


def entropy(p: pd.Series) -> pd.Series:
    p = p.clip(1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


df, src, missing = load()
st.title("Adaptive AI decisions")
st.caption(f"Data: {src}. Research use only; not for clinical use.")
if missing:
    st.error(f"data/predictions.csv lacks columns {missing}. Run convert_trajectory.py on a saved trajectory JSON.")
    st.stop()
if src.startswith("SIMULATED"):
    st.warning("SIMULATED DATA. The allocation pattern is hand-made to show the layout. It is not model behavior.")

eps = st.sidebar.slider("Allocation-change tolerance ε", 0.0, 0.2, 0.02, 0.005,
                        help="Display rule: an hour counts as Adapting when |q change| exceeds ε.")
pid = st.selectbox("Episode", df["patient_id"].unique().tolist())
g = df[df["patient_id"] == pid].reset_index(drop=True)
dq = g["temporal_weight"].diff().abs().fillna(0)
g["phase"] = np.where(g["status"] == "static_only_fallback", "Fallback", np.where(dq > eps, "Adapting", "Stable"))
g["disagreement"] = g["temporal_probability"] - g["static_probability"]
backend_ent = "entropy_static" in g.columns
if not backend_ent:
    g["entropy_static"] = entropy(g["static_probability"])
g["entropy_final"] = entropy(g["risk"])

cols = st.columns(3)
for c, ph in zip(cols, COLORS):
    n = int((g["phase"] == ph).sum())
    c.metric(f"{ph} hours", f"{n} of {len(g)}", f"{n / len(g):.0%}", delta_color="off")

fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    row_heights=[0.34, 0.22, 0.22, 0.22],
                    subplot_titles=["Estimated probability of shifted challenge label", "Temporal allocation q",
                                    "Model disagreement d = T − S", "Predictive entropy (nats)"])
for col, nm, color, w in [("static_probability", "Static S", "#6b7280", 2),
                          ("temporal_probability", "Temporal T", "#2563eb", 2), ("risk", "Final", "#d97706", 3)]:
    fig.add_trace(go.Scatter(x=g["hour"], y=g[col], mode="lines", name=nm, line=dict(color=color, width=w)), row=1, col=1)
fig.add_trace(go.Scatter(x=g["hour"], y=g["temporal_weight"], mode="lines", name="q",
                         line=dict(color="#6b7280", width=1), showlegend=False), row=2, col=1)
for ph, color in COLORS.items():
    m = g[g["phase"] == ph]
    fig.add_trace(go.Scatter(x=m["hour"], y=m["temporal_weight"], mode="markers", name=ph,
                             marker=dict(color=color, size=9)), row=2, col=1)
fig.add_trace(go.Scatter(x=g["hour"], y=g["disagreement"], mode="lines+markers", name="d",
                         line=dict(color="#2563eb"), showlegend=False), row=3, col=1)
fig.add_hline(y=0, line_color="gray", line_width=1, row=3, col=1)
fig.add_trace(go.Scatter(x=g["hour"], y=g["entropy_final"], mode="lines", name="Final entropy",
                         line=dict(color="#d97706")), row=4, col=1)
fig.add_trace(go.Scatter(x=g["hour"], y=g["entropy_static"], mode="lines", name="Static entropy",
                         line=dict(color="#6b7280", dash="dot")), row=4, col=1)
run = None  # shade contiguous fallback runs on every row
for h, ph in list(zip(g["hour"], g["phase"])) + [(None, None)]:
    if ph == "Fallback" and run is None:
        run = [h, h]
    elif ph == "Fallback" and run is not None:
        run[1] = h
    elif run is not None:
        for i in range(1, 5):
            fig.add_vrect(x0=run[0] - 0.5, x1=run[1] + 0.5, fillcolor=COLORS["Fallback"], opacity=0.1,
                          line_width=0, row=i, col=1)
        run = None
fig.update_yaxes(range=[-0.02, 1.02], row=2, col=1)
fig.update_xaxes(title_text="Hours since ICU admission", row=4, col=1)
fig.update_layout(height=820, margin=dict(t=40, b=20), legend=dict(orientation="h", y=1.07))
st.plotly_chart(fig, width="stretch")

with st.expander("How to read this page"):
    st.markdown(
        "- **Fallback** (red bands): the backend reported `static_only_fallback` for that hour.\n"
        "- **Adapting / Stable**: display rule applied by this page. Adapting means q moved by more than ε since the previous hour.\n"
        "- **Disagreement** is T − S as reported. It is empty where there is no temporal estimate.\n"
        f"- **Entropy**: {'backend entropy_static column and' if backend_ent else 'binary entropy computed here, in nats, from the reported probabilities; the backend entropy_static is not in the file, and'} "
        "the final-probability entropy computed here. It is not a confidence interval and not a calibrated uncertainty.\n"
        "- Contract requests: richer backend fallback and abstention reasons, and the backend decision-summary fields.")
