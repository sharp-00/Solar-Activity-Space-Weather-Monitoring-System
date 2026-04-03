"""pages/9_⏱️_Lag_Analysis.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_main_data, load_kp_data, load_stats_file
from utils.refresh import auto_refresh_check, render_refresh_sidebar, render_date_filter
from utils.theme import apply_theme, layout, axis
from analysis import cross_correlation

st.set_page_config(page_title="Lag Analysis", page_icon="⏱️", layout="wide")
apply_theme(); auto_refresh_check(); render_refresh_sidebar()

st.title("⏱️ Lag-Time & Superposed Epoch Analysis")
df_main = render_date_filter(load_main_data())
kp_df   = load_kp_data()

if df_main.empty:
    st.warning("No data."); st.stop()

# Enrich with dedicated Kp data
df = df_main.copy()
if not kp_df.empty:
    for col in kp_df.columns:
        if col not in df.columns:
            df[col] = kp_df[col].reindex(df.index)

st.info("X-rays hit Earth in 8 min; CME plasma arrives 2–4 days later.")

c_left, c_right = st.columns(2)

with c_left:
    st.subheader("Superposed Epoch Analysis")
    trigger_candidates = [c for c in ["flare_X","flare_M","ssn","f107","kp_daily_max"]
                          if c in df.columns and df[c].notna().any()]
    if not trigger_candidates:
        st.info("No suitable trigger columns found.")
    else:
        trigger_col = st.selectbox("Trigger variable", trigger_candidates)
        op  = st.selectbox("Operator", [">=","<=","=="])
        thr = st.number_input("Threshold",
                              value=1.0 if "flare" in trigger_col else 100.0)

        ops = {">=": lambda s,t: s>=t, "<=": lambda s,t: s<=t, "==": lambda s,t: s==t}
        triggers = df[ops[op](df[trigger_col], thr)].index
        st.markdown(f"**Trigger events:** {len(triggers)}")

        dst_col = next((c for c in ["dst_daily_mean","dst_daily_min"] if c in df.columns), None)
        if len(triggers) > 0 and dst_col:
            try:
                W = 10
                df_r = df.reset_index()
                epochs = []
                for t in triggers:
                    matches = df_r.index[df_r["date"] == t].tolist()
                    if not matches: continue
                    i = matches[0]
                    if W <= i < len(df_r) - W:
                        epochs.append(df_r[dst_col].iloc[i-W:i+W+1].values)
                if epochs:
                    mean_e = np.nanmean(epochs, axis=0)
                    fig = px.line(x=np.arange(-W, W+1), y=mean_e,
                                  labels={"x":"Days from Trigger","y":f"Mean {dst_col} (nT)"})
                    fig.add_vline(x=0, line_dash="dash", line_color="#f59e0b",
                                  annotation_text="Trigger")
                    fig.update_layout(**layout(320))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Epoch analysis error: {e}")

with c_right:
    st.subheader("Cross-Correlation Explorer")
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if df[c].notna().sum() > 50]
    if len(num_cols) < 2:
        st.info("Not enough numeric columns for correlation.")
    else:
        default_x = "ssn" if "ssn" in num_cols else num_cols[0]
        default_y = "dst_daily_mean" if "dst_daily_mean" in num_cols else num_cols[1]
        cx = st.selectbox("Driver",   num_cols, index=num_cols.index(default_x))
        cy = st.selectbox("Response", num_cols, index=num_cols.index(default_y))
        max_lag = st.slider("Max lag (days)", 7, 90, 30)

        if cx != cy:
            try:
                aligned = df[[cx,cy]].dropna()
                # Reset index to ensure equal-length integer-indexed Series for cross_correlation
                aligned = aligned.reset_index(drop=True)
                if len(aligned) > 50:
                    cc = cross_correlation(aligned[cx], aligned[cy], max_lag)
                    fig2 = px.line(cc, x="lag_days", y="pearson_r",
                                   labels={"lag_days":"Lag (days)","pearson_r":"Pearson r"})
                    fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
                    fig2.update_layout(**layout(320))
                    st.plotly_chart(fig2, use_container_width=True)
                    best = cc.loc[cc["pearson_r"].abs().idxmax()]
                    st.info(f"Peak: **r={best['pearson_r']:.3f}** at lag **{int(best['lag_days'])}d**")
            except Exception as e:
                st.error(f"Cross-correlation error: {e}")
