"""pages/6_📅_Monthly_Stats.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_main_data, load_kp_data, load_dst_data, load_f107_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar, render_date_filter
from utils.theme import apply_theme, layout, axis
from analysis import compute_monthly_stats

st.set_page_config(page_title="Monthly Stats", page_icon="📅", layout="wide")
apply_theme(); auto_refresh_check(); render_refresh_sidebar()

st.title("📅 Monthly Statistics & Temporal Aggregation")

try:
    df_main = render_date_filter(load_main_data())
    kp_df   = load_kp_data()
    dst_df  = load_dst_data()
    f107_df = load_f107_data()

    if df_main.empty:
        st.warning("No data. Click **⚡ Fetch Latest Data** in the sidebar.")
        st.stop()

    # Enrich with dedicated loaders
    df = df_main.copy()
    for src in [kp_df, dst_df, f107_df]:
        if not src.empty:
            for col in src.columns:
                if col not in df.columns or df[col].isna().all():
                    df[col] = src[col].reindex(df.index)

    st.info("Monthly aggregation smooths daily noise to reveal medium-term trends.")

    monthly = compute_monthly_stats(df.copy())
    if monthly.empty:
        st.warning("Not enough data columns to compute monthly stats.")
        st.stop()

    # ── Chart — draw all available monthly series ─────────────────────────────
    # Define which columns to try, their colors, names, and which y-axis
    SERIES = [
        ("ssn_mean",      "#f59e0b", "SSN Mean",    "y"),
        ("ssn_max",       "#fbbf24", "SSN Max",     "y"),
        ("kp_daily_max",  "#22c55e", "Kp Max",      "y2"),
        ("kp_daily_mean", "#4ade80", "Kp Mean",     "y2"),
        ("dst_daily_min", "#60a5fa", "Dst Min",     "y3"),
        ("f107_mean",     "#fb923c", "F10.7 Mean",  "y4"),
    ]

    fig = go.Figure()
    used_axes = set()
    for col, color, name, yaxis in SERIES:
        if col in monthly.columns and monthly[col].notna().any():
            fig.add_trace(go.Scattergl(
                x=monthly.index, y=monthly[col],
                line=dict(color=color, width=2), name=name,
                yaxis=yaxis,
            ))
            used_axes.add(yaxis)

    layout_kwargs = dict(height=430,
                         xaxis=axis("Date"),
                         yaxis=axis("SSN / Kp"),
                         hovermode="x unified")

    # Add secondary axes only if used
    if "y2" in used_axes:
        layout_kwargs["yaxis2"] = dict(title="Kp (0–9)", overlaying="y", side="right")
    if "y3" in used_axes:
        layout_kwargs["yaxis3"] = dict(title="Dst (nT)", overlaying="y",
                                       side="right", position=0.85)
    if "y4" in used_axes:
        layout_kwargs["yaxis4"] = dict(title="F10.7 (sfu)", overlaying="y",
                                       side="left", position=0.05)

    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Summary Table")
    # Show dates without time component for cleanliness
    display = monthly.copy()
    display.index = display.index.strftime("%Y-%m-%d")
    st.dataframe(display.round(2).head(36), use_container_width=True, height=400)

except Exception as e:
    st.error(f"Monthly stats error: {e}")
