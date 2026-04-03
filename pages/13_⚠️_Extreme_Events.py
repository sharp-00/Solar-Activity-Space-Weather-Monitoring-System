"""pages/13_⚠️_Extreme_Events.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_main_data, load_kp_data, load_dst_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar, render_date_filter
from utils.theme import apply_theme, layout, axis
from analysis import find_extreme_events

st.set_page_config(page_title="Extreme Events", page_icon="⚠️", layout="wide")
apply_theme(); auto_refresh_check(); render_refresh_sidebar()

st.title("⚠️ Extreme Events Detection")

df_main = render_date_filter(load_main_data())
kp_df   = load_kp_data()
dst_df  = load_dst_data()

if df_main.empty:
    st.warning("No data."); st.stop()

df = df_main.copy()
for src in [kp_df, dst_df]:
    if not src.empty:
        for col in src.columns:
            if col not in df.columns:
                df[col] = src[col].reindex(df.index)

candidates = [c for c in ["dst_daily_min","kp_daily_max","ssn","f107","flare_xray_total"]
              if c in df.columns and df[c].notna().any()]
if not candidates:
    st.warning("No suitable columns for extreme event detection.")
    st.stop()

st.info("Days beyond the Z-score threshold are flagged as extreme events.")

c_cfg, c_viz = st.columns([1, 2])
with c_cfg:
    sigma = st.slider("Z-Score threshold (σ)", 1.5, 4.0, 2.5, 0.1)
    var   = st.selectbox("Variable", candidates)

with c_viz:
    try:
        events = find_extreme_events(df, var, sigma)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=df.index, y=df[var], mode="lines",
                                   line=dict(color="#e5e7eb", width=1),
                                   opacity=0.4, name="Normal"))
        if not events.empty:
            fig.add_trace(go.Scattergl(
                x=events.index, y=events[var], mode="markers",
                marker=dict(size=8, color=events["z_score"],
                            colorscale="RdYlGn_r", showscale=True,
                            colorbar=dict(title="Z-Score")),
                name="Extreme"))
        fig.update_layout(**layout(420, xaxis=axis("Date"), yaxis=axis(var)))
        st.plotly_chart(fig, use_container_width=True)

        if not events.empty:
            st.subheader(f"Top Extreme Events ({len(events)} total)")
            st.dataframe(events[[var,"z_score"]].head(15).round(3),
                         use_container_width=True, height=300)
        else:
            st.info("No extreme events found at this threshold.")
    except Exception as e:
        st.error(f"Extreme events error: {e}")
