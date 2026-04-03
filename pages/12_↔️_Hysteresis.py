"""pages/12_↔️_Hysteresis.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_main_data, load_kp_data, load_sunspots_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar, render_date_filter
from utils.theme import apply_theme, layout, axis
from analysis import analyze_hysteresis

st.set_page_config(page_title="Hysteresis", page_icon="↔️", layout="wide")
apply_theme(); auto_refresh_check(); render_refresh_sidebar()

st.title("↔️ Hysteresis Analysis: Solar Cycle Phase Effects")

with st.expander("ℹ️ What We Are Analyzing: Hysteresis", expanded=False):
    st.markdown("""
    - **What we are doing:** Plotting Solar Activity (SSN) against Geomagnetic Activity (Kp) and colorizing the points based on whether the cycle is actively *rising* to a peak or *falling* away from it.
    - **Goal:** To expose an asymmetry in space weather: the Earth is battered by more severe storms *after* the solar maximum has passed. During the declining phase, coronal holes migrate towards the solar equator, spewing recurrent high-speed solar wind streams. The sun is calming down, but the storms are getting worse!
    """)


st.info("💡 **Historical Trivia:** The hysteresis effect implies delayed consequences. It explains why satellite operators experience worse atmospheric drag anomalies years after the sunspot peak has passed, catching unprepared orbital trajectory calculations off guard.")

df_main = render_date_filter(load_main_data())
kp_df   = load_kp_data()
ssn_df  = load_sunspots_data()

if df_main.empty:
    st.warning("No data."); st.stop()

df = df_main.copy()
for src in [kp_df, ssn_df]:
    if not src.empty:
        for col in src.columns:
            if col not in df.columns:
                df[col] = src[col].reindex(df.index)

ssn_cols = [c for c in df.columns
            if any(k in c.lower() for k in ("ssn","sn")) and df[c].notna().any()]
kp_cols  = [c for c in df.columns if "kp" in c.lower() and df[c].notna().any()]

if not ssn_cols:
    st.warning("No SSN/sunspot data. Click **⚡ Fetch Latest Data**."); st.stop()
if not kp_cols:
    st.warning("No Kp data. Click **⚡ Fetch Latest Data**."); st.stop()

st.info("Green = rising phase, red = falling phase. Asymmetry proves solar cycle 'memory'.")

c_cfg, c_viz = st.columns([1, 2])
with c_cfg:
    sn_col = st.selectbox("Sunspot proxy", ssn_cols)
    kp_col = st.selectbox("Geomagnetic index", kp_cols)

with c_viz:
    try:
        hdf = analyze_hysteresis(df, sn_col=sn_col, kp_col=kp_col)
        rising  = hdf[hdf["phase_type"] == "Rising"]
        falling = hdf[hdf["phase_type"] == "Falling"]
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=rising[sn_col], y=rising[kp_col], mode="markers",
                                   marker=dict(size=4, color="#22c55e", opacity=0.4),
                                   name="Rising Phase"))
        fig.add_trace(go.Scattergl(x=falling[sn_col], y=falling[kp_col], mode="markers",
                                   marker=dict(size=4, color="#ef4444", opacity=0.4),
                                   name="Falling Phase"))
        fig.update_layout(**layout(500,
            xaxis=axis(f"{sn_col} (Solar Activity)"),
            yaxis=axis(f"{kp_col} (0–9)")))
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Rising: **{len(rising):,}** days  |  Falling: **{len(falling):,}** days")
    except Exception as e:
        st.error(f"Hysteresis error: {e}")
