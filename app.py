"""app.py — Space Weather Analytics · Home & KPI Dashboard"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import load_main_data, load_kp_data, load_dst_data, load_f107_data, load_flares_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar, render_date_filter
from utils.theme import apply_theme, layout, axis

st.set_page_config(
    page_title="Space Weather Analytics",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
auto_refresh_check()
render_refresh_sidebar()

st.title("☀️ Space Weather Analytics")
st.markdown("**Real-time & historical solar / geomagnetic data intelligence**  \n"
            "Data sources: SILSO · NOAA NCEI/SWPC · GFZ · NASA OMNI · WDC Kyoto")

df_all = load_main_data()

if df_all.empty:
    st.warning("No data found. Click **⚡ Fetch Latest Data** in the sidebar to download the dataset.")
    st.stop()

df = render_date_filter(df_all)

# Load individual datasets for KPIs (fall back to main df columns if not available)
kp_df  = load_kp_data()
dst_df = load_dst_data()
f107_df = load_f107_data()

# Stale-data banner
missing_cols = [c for c in ["ssn","kp_daily_max","dst_daily_min","f107"]
                if c not in df_all.columns or df_all[c].isna().all()]
if missing_cols:
    st.warning(
        "Incomplete data detected. Columns " + ", ".join(missing_cols) +
        " are missing or empty. Your data was built with an older pipeline "
        "version. Click the Fetch Latest Data button in the sidebar to rebuild."
    )

st.markdown(f"**Records in view:** {len(df):,}  |  "
            f"**Span:** {df.index.min().date()} → {df.index.max().date()}")
st.markdown("---")

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def _last_val(source_df, col, fallback_df=None, fallback_col=None):
    for d, c in [(source_df, col), (fallback_df, fallback_col or col)]:
        if d is not None and not d.empty and c in d.columns and d[c].notna().any():
            return float(d[c].dropna().iloc[-1])
    return float("nan")

with k1:
    val  = _last_val(df, "ssn")
    prev = float(df["ssn"].iloc[-8]) if "ssn" in df.columns and len(df) > 7 else val
    st.metric("Latest SSN", f"{val:.0f}" if not np.isnan(val) else "N/A",
              f"{val-prev:+.0f} vs 7d ago" if not np.isnan(val) else "")

with k2:
    val = _last_val(f107_df, "f107", df, "f107")
    st.metric("F10.7 Flux (sfu)", f"{val:.1f}" if not np.isnan(val) else "N/A")

with k3:
    val = _last_val(kp_df, "kp_daily_max", df, "kp_daily_max")
    color = "🟢" if val < 5 else ("🟡" if val < 7 else "🔴")
    st.metric("Latest Kp Max", f"{color} {val:.1f}" if not np.isnan(val) else "N/A")

with k4:
    val = _last_val(dst_df, "dst_daily_min", df, "dst_daily_min")
    st.metric("Dst Min (nT)", f"{val:.0f}" if not np.isnan(val) else "N/A")

with k5:
    kp_col_df = kp_df if not kp_df.empty and "kp_daily_max" in kp_df.columns else df
    if "kp_daily_max" in kp_col_df.columns:
        filt = kp_col_df.loc[df.index.min():df.index.max()]
        n_storms = (filt["kp_daily_max"] >= 5).sum()
        st.metric("G1+ Storms", f"{n_storms:,}", "in selected window")
    else:
        st.metric("G1+ Storms", "N/A")

st.markdown("---")

# ── Mini overview charts ──────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Sunspot Number")
    if "ssn" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=df.index, y=df["ssn"],
                                   line=dict(color="#fbbf24", width=1), opacity=0.5, name="Daily"))
        rm = df["ssn"].rolling(27, center=True, min_periods=5).mean()
        fig.add_trace(go.Scattergl(x=rm.index, y=rm,
                                   line=dict(color="#f59e0b", width=2), name="27d Mean"))
        fig.update_layout(**layout(350, xaxis=axis("Date"), yaxis=axis("SSN")))
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Geomagnetic Kp Index")
    if "kp_daily_max" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=df.index, y=df["kp_daily_max"],
                                   line=dict(color="#22c55e", width=1),
                                   fill="tozeroy", fillcolor="rgba(34,197,94,0.15)", name="Kp Max"))
        fig.add_hline(y=5, line_dash="dash", line_color="#fbbf24", annotation_text="G1")
        fig.add_hline(y=7, line_dash="dash", line_color="#ef4444", annotation_text="G3")
        fig.update_layout(**layout(350, yaxis_range=[0, 9.5],
                                   xaxis=axis("Date"), yaxis=axis("Kp (0–9)")))
        st.plotly_chart(fig, use_container_width=True)

st.info("👈 Use the **sidebar** to navigate between analysis pages.")
