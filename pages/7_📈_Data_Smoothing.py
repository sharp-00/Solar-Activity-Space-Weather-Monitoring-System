"""pages/7_📈_Data_Smoothing.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter

from utils.data_loader import load_main_data, load_kp_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar, render_date_filter
from utils.theme import apply_theme, layout, axis

st.set_page_config(page_title="Data Smoothing", page_icon="📈", layout="wide")
apply_theme(); auto_refresh_check(); render_refresh_sidebar()

st.title("📈 Data Smoothing & Noise Filtration")

with st.expander("ℹ️ What We Are Analyzing: Data Smoothing", expanded=False):
    st.markdown("""
    - **What we are doing:** Applying different mathematical filters (e.g., Simple Moving Averages, Savitzky-Golay) to raw telemetry signals.
    - **Goal:** Real-world data is inherently noisy due to ground sensor fluctuations, instrument switching, and the Sun's 27-day axial rotation. We attempt to discover the "true baseline" by filtering out this high-frequency noise without losing the signal of authentic anomalies.
    """)


st.info("💡 **Historical Trivia:** Observational noise in early historical SSN data often came from atmospheric cloud cover over observatories or variations in the telescope optics of the 18th and 19th centuries, necessitating algorithmic smoothing to find true cyclic trends.")

try:
    df_main = render_date_filter(load_main_data())
    kp_df   = load_kp_data()
    if df_main.empty:
        st.warning("No data."); st.stop()

    df = df_main.copy()
    if not kp_df.empty:
        for col in kp_df.columns:
            if col not in df.columns:
                df[col] = kp_df[col].reindex(df.index)

    st.info("**SMA** flattens peaks.  \n"
            "**Savitzky-Golay** preserves peak heights while removing noise.")

    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if df[c].notna().sum() > 50]
    if not num_cols:
        st.warning("No numeric columns with enough data."); st.stop()

    c_cfg, c_viz = st.columns([1, 2])
    with c_cfg:
        var   = st.selectbox("Variable", num_cols)
        sma_w = st.slider("SMA Window (days)", 3, 365, 27, 2)
        sg_w  = st.slider("SavGol Window (odd)", 5, 365, 51, 2)
        sg_p  = st.slider("SavGol Polynomial", 1, 5, 3)
        valid = sg_w > sg_p
        if not valid:
            st.error("Window must exceed polynomial order.")

    with c_viz:
        if var in df.columns and valid:
            s   = df[var].dropna()
            sma = s.rolling(sma_w, center=True, min_periods=1).mean()
            sg  = savgol_filter(s, window_length=sg_w, polyorder=sg_p)
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=s.index, y=s, line=dict(color="#d1d5db", width=1),
                                       opacity=0.5, name="Raw"))
            fig.add_trace(go.Scattergl(x=sma.index, y=sma, line=dict(color="#22c55e", width=2),
                                       name=f"SMA ({sma_w}d)"))
            fig.add_trace(go.Scattergl(x=s.index, y=sg, line=dict(color="#f97316", width=2),
                                       name=f"SavGol (w={sg_w},p={sg_p})"))
            fig.update_layout(**layout(520, xaxis=axis("Date"), yaxis=axis(var)))
            st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Page error: {e}")
