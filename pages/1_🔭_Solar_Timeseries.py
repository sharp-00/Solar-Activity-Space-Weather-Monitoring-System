"""pages/1_🔭_Solar_Timeseries.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_sunspots_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar
from utils.theme import apply_theme, layout, axis

st.set_page_config(page_title="Solar Timeseries", page_icon="🔭", layout="wide")
apply_theme(); auto_refresh_check(); render_refresh_sidebar()

st.title("🔭 Solar Number Time Series & Historical Analysis")

with st.expander("ℹ️ What We Are Analyzing: Solar Timeseries", expanded=False):
    st.markdown("""
    - **What we are doing:** We plot decades of historical sunspot, radio flux, and flare data on a continuous timeline. 
    - **Goal:** To physically visualize the ~11-year solar cycle (Schwabe cycle) across multiple generations. By comparing the crests and troughs (Solar Maximums and Minimums), we identify historical patterns in solar volatility.
    """)


st.info("💡 **Historical Trivia:** The **Maunder Minimum** (1645–1715) was a period when sunspots became exceedingly rare. This era coincided with the 'Little Ice Age' in Europe and North America. Conversely, the highest explicitly recorded sunspot number occurred during Solar Cycle 19 in 1957 (SSN peaking over 350).")

try:
    df = load_sunspots_data()
    if df.empty:
        st.warning("Sunspot data unavailable. Click **⚡ Fetch Latest Data** in the sidebar.")
        st.stop()

    # Support both 'sn' (dedicated file) and 'ssn' (main merged file)
    sn_col = "sn" if "sn" in df.columns else ("ssn" if "ssn" in df.columns else None)
    if sn_col is None:
        st.warning("No sunspot number column found. Click **⚡ Fetch Latest Data**.")
        st.stop()

    df = df[df[sn_col].notna()].copy()

    st.info("**11-year solar cycles** visible in the 365-day mean.  \n"
            "**Current cycle:** Solar Cycle 25 (began 2019, max ~2024–2025).")

    # ── Time series ───────────────────────────────────────────────────────────
    st.subheader("Historical Sunspot Number")
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=df.index, y=df[sn_col],
                               line=dict(color="#f59e0b", width=1), opacity=0.5, name="Daily SSN"))
    rm = df[sn_col].rolling(365, center=True, min_periods=100).mean()
    fig.add_trace(go.Scattergl(x=rm.index, y=rm,
                               line=dict(color="#d946ef", width=3), name="365-Day Mean"))
    fig.update_layout(**layout(450, xaxis=dict(title="Date", rangeslider=dict(visible=True)),
                               yaxis=axis("Sunspot Number")))
    st.plotly_chart(fig, use_container_width=True)

    # ── Monthly heatmap ───────────────────────────────────────────────────────
    st.subheader("Monthly Mean Activity Heatmap")
    tmp = df.copy()
    tmp["year"] = tmp.index.year
    tmp["month"] = tmp.index.month
    pivot = (tmp.groupby(["year","month"])[sn_col].mean().reset_index()
               .pivot(index="year", columns="month", values=sn_col))
    fig2 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        y=pivot.index, colorscale="YlOrRd",
        colorbar=dict(title="SSN"), hoverongaps=False,
    ))
    fig2.update_layout(**layout(600, xaxis=axis("Month"), yaxis=axis("Year")))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Mean SSN", f"{df[sn_col].mean():.1f}")
    c3.metric("Max SSN", f"{df[sn_col].max():.0f}")
    c4.metric("Date Range", f"{df.index.min().year}–{df.index.max().year}")

except Exception as e:
    st.error(f"Page error: {e}")
