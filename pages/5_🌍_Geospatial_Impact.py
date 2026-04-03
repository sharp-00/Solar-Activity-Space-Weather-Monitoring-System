"""pages/5_🌍_Geospatial_Impact.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_main_data, load_kp_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar
from utils.theme import apply_theme, layout

st.set_page_config(page_title="Geospatial Impact", page_icon="🌍", layout="wide")
apply_theme(); auto_refresh_check(); render_refresh_sidebar()

st.title("🌍 Geospatial Impact: Auroral Extent Mapping")

with st.expander("ℹ️ What We Are Analyzing: Geospatial Impact", expanded=False):
    st.markdown("""
    - **What we are doing:** We calculate and project the "auroral oval" onto an Earth map using historical / simulated Kp numbers.
    - **Goal:** To visually answer the question "How far south (or north) will the aurora be visible?". Severe geomagnetic storms widen the auroral footprint, drastically expanding the latitude range of visible Northern/Southern lights and mapping geomagnetically induced current (GIC) risk.
    """)


st.info("💡 **Historical Trivia:** During the 1859 Carrington Event, the auroral oval was pushed so far toward the equator that people in the Caribbean (Cuba, Jamaica) and Hawaii reported seeing the Northern Lights. Gold miners in the Rocky Mountains woke up and began making breakfast, thinking it was morning!")

try:
    df_main = load_main_data()
    kp_df   = load_kp_data()

    if df_main.empty:
        st.warning("No data."); st.stop()

    st.info("Quiet days (Kp 0–3): aurora above ±65°.  "
            "Extreme storms (Kp 8–9): aurora reaches ±35° — visible from Florida.")

    CITY_BANDS = {80:"N. Canada / Greenland", 70:"Alaska / N. Canada",
                  60:"Mid-Canada / Scotland", 55:"N. US / N. UK",
                  50:"N. US / Central Europe", 45:"Oregon / France",
                  40:"New York / S. Europe", 35:"Mid-US / Mediterranean"}

    def visible(lat):
        for t in sorted(CITY_BANDS, reverse=True):
            if lat >= t: return CITY_BANDS[t]
        return "Equatorial regions"

    # Build a unified Kp series: prefer dedicated kp_df, fall back to main data
    kp_col_name = None
    kp_source   = None
    for src in [kp_df, df_main]:
        if not src.empty:
            col = next((c for c in ["kp_daily_max","kp_daily_mean"] if c in src.columns and src[c].notna().any()), None)
            if col:
                kp_col_name = col
                kp_source   = src
                break

    dates = np.unique(df_main.index.date)
    dates.sort()
    if len(dates) == 0:
        st.warning("No dates found."); st.stop()

    sel = st.select_slider("Select date", options=dates, value=dates[len(dates) // 2])

    # Look up Kp for the selected date
    kp = 0.0
    if kp_source is not None and kp_col_name is not None:
        ts = pd.Timestamp(sel)
        if ts in kp_source.index:
            v = kp_source.loc[ts, kp_col_name]
            kp = float(v) if pd.notna(v) else 0.0
        elif not kp_source.empty:
            # Find nearest date within ±3 days
            idx = kp_source.index.get_indexer([ts], method="nearest")[0]
            if idx >= 0:
                near_date = kp_source.index[idx]
                if abs((near_date - ts).days) <= 3:
                    v = kp_source.iloc[idx][kp_col_name]
                    kp = float(v) if pd.notna(v) else 0.0

    boundary = max(35.0, 68.0 - 3.6 * kp)

    c1, c2 = st.columns(2)
    c1.metric("Kp Max", f"{kp:.2f}/9.00")
    c2.metric("Aurora Boundary", f"±{boundary:.1f}°")
    st.markdown(f"**Visible in:** {visible(boundary)}")

    if kp_source is None or kp_col_name is None:
        st.info("Kp data not available — aurora shown at quiet-day default (±68°). "
                "Click **⚡ Fetch Latest Data** for live Kp values.")

    # Build aurora grid
    lat_g = np.arange(-90, 91, 6)
    lon_g = np.arange(-180, 181, 8)
    lon_m, lat_m = np.meshgrid(lon_g, lat_g)
    mask  = (lat_m.flatten() >= boundary) | (lat_m.flatten() <= -boundary)
    lats, lons = lat_m.flatten()[mask], lon_m.flatten()[mask]
    color = "#22c55e" if kp <= 4 else ("#fbbf24" if kp <= 6 else ("#ef4444" if kp <= 8 else "#dc2626"))

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, mode="markers",
        marker=dict(size=4, color=color, opacity=0.5, line=dict(width=0)),
        hoverinfo="none"))
    fig.update_geos(projection_type="orthographic",
                    showcoastlines=True, coastlinecolor="#374151",
                    showland=True, landcolor="#1f2937",
                    showocean=True, oceancolor="#0f172a")
    fig.update_layout(**layout(620))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Page error: {e}")
