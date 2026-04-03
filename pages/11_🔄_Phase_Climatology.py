"""pages/11_🔄_Phase_Climatology.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_main_data, load_kp_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar, render_date_filter
from utils.theme import apply_theme, layout, axis
from analysis import analyze_phase_locked_climatology

st.set_page_config(page_title="Phase Climatology", page_icon="🔄", layout="wide")
apply_theme(); auto_refresh_check(); render_refresh_sidebar()

st.title("🔄 Phase-Locked Climatology: Solar Cycle Risk Zones")

with st.expander("ℹ️ What We Are Analyzing: Phase Climatology", expanded=False):
    st.markdown("""
    - **What we are doing:** Normalizing time into "fractions of a solar cycle" (where $0$ is the start of a cycle and $1.0$ is the end) and binning storm occurrences into these fractional buckets.
    - **Goal:** To map the "danger zone". It answers exactly *where* in the 11-year calendar geomagnetic storms are most likely. It turns out storms do not always happen exactly at Solar Maximum!
    """)

st.info("💡 **Historical Trivia:** Some of the most devastating storms occur *after* Solar Maximum. The infamous Halloween Storms of 2003 happened 3.5 years *after* the peak of Solar Cycle 23, deep into the declining phase.")

df_main = render_date_filter(load_main_data())
kp_df   = load_kp_data()

if df_main.empty:
    st.warning("No data."); st.stop()

df = df_main.copy()
if not kp_df.empty:
    for col in kp_df.columns:
        if col not in df.columns:
            df[col] = kp_df[col].reindex(df.index)

kp_cols = [c for c in df.columns if "kp" in c.lower() and df[c].notna().any()]
if not kp_cols:
    st.warning("No Kp data found. Click **⚡ Fetch Latest Data** to rebuild.")
    st.stop()

st.info("Phase bins divide the 11-year cycle (0=min, 1=max). "
        "Shows which cycle phases produce the most storms.")

c_cfg, c_viz = st.columns([1, 2])
with c_cfg:
    bins      = st.slider("Phase bins", 10, 40, 20, 2)
    kp_col    = st.selectbox("Kp column", kp_cols)
    threshold = st.number_input("Storm threshold (Kp)", 0.0, 9.0, 6.0, 0.5)

with c_viz:
    try:
        clim = analyze_phase_locked_climatology(df, kp_col=kp_col,
                                                storm_threshold=threshold, bins=bins)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=clim.index.astype(float), y=clim["kp_mean"],
            mode="lines+markers", line=dict(color="#22c55e", width=2),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.15)", name="Mean Kp",
        ))
        fig.add_trace(go.Scattergl(
            x=clim.index.astype(float), y=clim["storm_prob_pct"],
            mode="lines+markers", line=dict(color="#ef4444", width=2),
            name="Storm Prob (%)", yaxis="y2",
        ))
        fig.update_layout(**layout(420,
            xaxis=axis("Solar Cycle Phase (0=min, 1=max)"),
            yaxis=axis("Mean Kp"),
            yaxis2=dict(title="Storm Probability (%)", overlaying="y", side="right"),
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(clim.astype(float).round(3), use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Phase climatology error: {e}")
