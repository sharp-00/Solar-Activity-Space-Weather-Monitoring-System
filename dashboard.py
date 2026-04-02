"""dashboard.py — Professional Interactive Dashboard for Space Weather Analytics
---------------------------------------------------------------------------
Visualizes the output of the solar weather pipeline (data/clean/solar_weather_daily.csv)
and the statistical results from data/analysis/stats/.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from pathlib import Path
from streamlit_option_menu import option_menu
import sys
sys.path.insert(0, str(Path(__file__).parent))
from analysis import (
    cross_correlation, find_extreme_events, solar_cycle_phase,
    compute_monthly_stats, analyze_periodicity, predictive_dominance,
    analyze_phase_locked_climatology, analyze_hysteresis
)

# ---------------------------------------------------------------------------
# Config & Professional UI Setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Space Weather Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS Theme (Fonts, Hiding Sidebar Arrow, Responsive blocks)
st.markdown("""
<style>
    /* Uniform professional font */
    html, body, [class*=\"css\"]  {
        font-family: 'Inter', 'Roboto', 'Helvetica Neue', sans-serif !important;
    }
    
    /* Hide the sidebar collapse arrow to keep menu static */
    [data-testid=\"collapsedControl\"] { display: none !important; }
    
    /* Main body styling */
    .main .block-container { padding-top: 2rem; max-width: 95%; }
    
    /* Headers */
    h1 { color: var(--text-color); font-weight: 700; }
    h2 { color: var(--text-color); font-weight: 600; font-size: 1.5rem; opacity: 0.9; }
    h3 { color: var(--text-color); font-weight: 500; font-size: 1.25rem; opacity: 0.8; }
    
    /* Expander styling */
    .streamlit-expanderHeader { font-weight: 600; color: #3b82f6; }
    
    /* Sidebar */
    [data-testid=\"stSidebar\"] { border-right: 1px solid rgba(128, 128, 128, 0.2); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading (Cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900)
def load_data():
    csv_path = Path("data/clean/solar_weather_daily.csv")
    if not csv_path.exists():
        st.error(f"Data file not found at {csv_path}. Please run the pipeline first.")
        st.stop()
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    return df

@st.cache_data(ttl=900)
def load_stats_file(filename):
    path = Path(f"data/analysis/stats/{filename}")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


df = load_data()

# Global default fallback columns
if "year" not in df.columns:
    df["year"] = df.index.year

# ---------------------------------------------------------------------------
# Glowing Sidebar Navigation Menu
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Space Weather Analytics")
    st.markdown("Data Intelligence & Forecasting")
    
    page = option_menu(
        menu_title=None,
        options=["System Overview", "Data Smoothing", "Correlation Matrix", "Lag-Time Analysis", "Extreme Events", "Periodicity Analysis", "Phase-Locked Climatology", "Hysteresis Analysis", "Monthly Statistics", "Geospatial Impact", "Physics Primer", "Data Sources", "Project Details"],
        icons=["activity", "bar-chart-line", "diagram-3", "clock-history", "alert-triangle", "wave", "circle-half", "arrows-expand", "calendar", "globe-americas", "lightning", "server", "info-square"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#64748b", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px", "text-align": "left", "margin":"5px", 
                "--hover-color": "rgba(128,128,128,0.2)", "border-radius": "8px"
            },
            "nav-link-selected": {
                "background-color": "#2563eb", "color": "white", "font-weight": "600",
                "box-shadow": "0 4px 6px -1px rgba(37, 99, 235, 0.4)"
            },
        }
    )

    st.markdown("---")
    
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    st.subheader("Global Data Parameters")
    date_range = st.slider(
        "Observation Window",
        min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD"
    )
    
    start_date, end_date = date_range
    df_filtered = df.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    
    st.markdown("---")
    st.caption(f"Valid Records: {len(df_filtered):,}")
    st.caption("Architecture: NOAA NCEI, SWPC, WDC Kyoto, SILSO.")


# ---------------------------------------------------------------------------
# PAGE 1: System Overview
# ---------------------------------------------------------------------------

def page_overview():
    st.title("System Overview: Time-Series Diagnostics")
    st.markdown("Macro-level inspection of solar drivers and terrestrial responses.")

    with st.expander("📖 Interpreting the System Overview", expanded=False):
        st.info(
            "**How to read these charts:**\n\n"
            "This tab directly contrasts the **cause** (Solar emissions shown on top) against the **effect** (Terrestrial disruptions on the bottom). \n"
            "- **The 11-Year Cycle:** The 27-day rolling mean lines on the SSN and F10.7 charts make the 11-year solar maximums and minimums obvious, filtering out daily rotational noise.\n"
            "- **Storm Thresholds:** Look at the Geomagnetic Kp and Dst indices. During Solar Maximums (when SSN peaks), notice how the Dst plot frequently plummets past the red dotted line (-100 nT), indicating severe auroral storms disrupting Earth's magnetosphere."
        )
        st.markdown(
            "**Definitions:** Sunspot Number (SSN), F10.7 Solar Radio Flux at 10.7 cm, Kp Index (Quasi-logarithmic planetary index), Dst Index (Disturbance Storm Time)."
        )

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sunspot Number (SSN)")
        fig_ssn = go.Figure()
        if "ssn" in df_filtered.columns:
            fig_ssn.add_trace(go.Scattergl(x=df_filtered.index, y=df_filtered["ssn"], 
                                         line=dict(color='#d97706', width=1), opacity=0.5, name="Daily SSN"))
            ssn_27d = df_filtered["ssn"].rolling(27, center=True, min_periods=5).mean()
            fig_ssn.add_trace(go.Scattergl(x=ssn_27d.index, y=ssn_27d, line=dict(color='#991b1b', width=2), name="27d Mean"))
        fig_ssn.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
        st.plotly_chart(fig_ssn, width='stretch')

    with col2:
        st.subheader("F10.7 Solar Flux")
        fig_f107 = go.Figure()
        if "f107" in df_filtered.columns:
            fig_f107.add_trace(go.Scattergl(x=df_filtered.index, y=df_filtered["f107"], 
                                          line=dict(color='#dc2626', width=1), opacity=0.5, name="Daily F10.7"))
            f107_27d = df_filtered["f107"].rolling(27, center=True, min_periods=5).mean()
            fig_f107.add_trace(go.Scattergl(x=f107_27d.index, y=f107_27d, line=dict(color='#7f1d1d', width=2), name="27d Mean"))
        fig_f107.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
        st.plotly_chart(fig_f107, width='stretch')

    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Major X-ray Flares (M & X class)")
        fig_flare = go.Figure()
        if "flare_M" in df_filtered.columns:
            fig_flare.add_trace(go.Bar(x=df_filtered.index, y=df_filtered.get("flare_M", 0), name="M-Class", marker_color='#f59e0b'))
        if "flare_X" in df_filtered.columns:
            fig_flare.add_trace(go.Bar(x=df_filtered.index, y=df_filtered.get("flare_X", 0), name="X-Class", marker_color='#b91c1c'))
        fig_flare.update_layout(barmode='stack', height=350, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
        st.plotly_chart(fig_flare, width='stretch')

    with col4:
        st.subheader("Geomagnetic Kp Index (Max)")
        fig_kp = go.Figure()
        if "kp_daily_max" in df_filtered.columns:
            fig_kp.add_trace(go.Scattergl(x=df_filtered.index, y=df_filtered["kp_daily_max"], 
                                        mode='lines', line=dict(color='#6d28d9', width=1),
                                        fill='tozeroy', fillcolor='rgba(109, 40, 217, 0.2)', name="Kp Max"))
            fig_kp.add_hline(y=5, line_dash="dash", line_color="#ea580c", annotation_text="G1 Storm")
            fig_kp.add_hline(y=7, line_dash="dash", line_color="#b91c1c", annotation_text="G3 Storm")
        fig_kp.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", yaxis_range=[0, 9.5])
        st.plotly_chart(fig_kp, width='stretch')

    st.markdown("---")
    st.subheader("Geomagnetic Dst Index (Equatorial Disturbance)")
    if "dst_daily_mean" in df_filtered.columns and "dst_daily_max" in df_filtered.columns and "dst_daily_min" in df_filtered.columns:
        fig_dst = go.Figure()
        fig_dst.add_trace(go.Scattergl(
            x=df_filtered.index.tolist() + df_filtered.index[::-1].tolist(),
            y=df_filtered["dst_daily_max"].tolist() + df_filtered["dst_daily_min"][::-1].tolist(),
            fill='toself', fillcolor='rgba(29, 78, 216, 0.2)', line=dict(color='rgba(255,255,255,0)'),
            name="Daily Range", hoverinfo="skip"
        ))
        fig_dst.add_trace(go.Scattergl(x=df_filtered.index, y=df_filtered["dst_daily_mean"], line=dict(color='#1e3a8a', width=1.5), name="Daily Mean"))
        fig_dst.add_trace(go.Scattergl(x=df_filtered.index, y=df_filtered["dst_daily_min"], mode='markers', marker=dict(size=3, color='#991b1b'), name="Daily Min", opacity=0.5))
        fig_dst.add_hline(y=-50, line_dash="dash", line_color="#ea580c", annotation_text="Moderate")
        fig_dst.add_hline(y=-100, line_dash="dash", line_color="#dc2626", annotation_text="Intense")
        fig_dst.add_hline(y=-200, line_dash="dash", line_color="#7f1d1d", annotation_text="Severe")
        fig_dst.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
        fig_dst.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_dst, width='stretch')


# ---------------------------------------------------------------------------
# PAGE 2: Data Smoothing
# ---------------------------------------------------------------------------

def page_smoothing():
    st.title("Data Smoothing & Noise Filtration")
    st.markdown("Apply digital filters to isolate macro trends from raw solar indices.")
    
    with st.expander("📖 Interpreting these Mathematical Filters", expanded=False):
        st.info(
            "**Why do we smooth this data?**\n"
            "Raw space weather telemetry is incredibly noisy. Sunspots appear and disappear daily as the sun rotates every 27 days, introducing rapid spikes that hide the true 11-year underlying cycle.\n\n"
            "**SMA vs Savitzky-Golay:**\n"
            "- **Simple Moving Average (SMA):** Great for general trends, but it computationally *flattens* the massive peaks of violent solar flares.\n"
            "- **Savitzky-Golay:** A superior polynomial filter used in astrophysics to eliminate noise while mathematically preserving the height and timing of absolute peaks. Notice how the orange line perfectly tracks extreme flare spikes that the blue SMA line completely misses."
        )
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.subheader("Filter Parameters")
        smooth_var = st.selectbox("Variable Target", [c for c in df_filtered.select_dtypes(include=np.number).columns], index=0)
        if smooth_var in df_filtered.columns:
            st.markdown("**(1) Rolling Mean (SMA)**")
            sma_window = st.slider("Window Length (Days)", 3, 365, 27, 2)
            st.markdown("**(2) Savitzky-Golay Filter**")
            savgol_window = st.slider("Kernel Window (Odd)", 5, 365, 51, 2)
            savgol_poly = st.slider("Polynomial Order", 1, 5, 3, 1)
            valid_params = savgol_window > savgol_poly
            if not valid_params: st.error("Kernel Window must strictly exceed Polynomial Order.")
        else:
            st.warning("Variable unavailable.")
    
    with col_s2:
        if smooth_var in df_filtered.columns and valid_params:
            valid_series = df_filtered[smooth_var].dropna()
            sma_series = valid_series.rolling(sma_window, center=True, min_periods=1).mean()
            savgol_series = savgol_filter(valid_series, window_length=savgol_window, polyorder=savgol_poly)
            
            fig_smooth = go.Figure()
            fig_smooth.add_trace(go.Scattergl(x=valid_series.index, y=valid_series, mode='lines', line=dict(color='#d1d5db', width=1), name=f"Raw Data", opacity=0.6))
            fig_smooth.add_trace(go.Scattergl(x=sma_series.index, y=sma_series, mode='lines', line=dict(color='#2563eb', width=2), name=f"SMA ({sma_window}d)"))
            fig_smooth.add_trace(go.Scattergl(x=valid_series.index, y=savgol_series, mode='lines', line=dict(color='#ea580c', width=2), name=f"SavGol (w={savgol_window}, p={savgol_poly})"))
            fig_smooth.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_smooth, width='stretch')


# ---------------------------------------------------------------------------
# PAGE 3: Correlation Matrix
# ---------------------------------------------------------------------------

def page_correlations():
    st.title("Correlation & Relationship Analytics")
    
    with st.expander("📖 Interpreting the Statistical Metrics", expanded=False):
        st.info(
            "**How to read this Matrix:**\n"
            "- A value near **+1.00** shows strong positive correlation. For example, SSN and F10.7 are almost perfectly correlated (+0.95+), proving they are reliable proxies for the exact same underlying solar mechanism.\n"
            "- A value near **-1.00** shows strong negative correlation. For example, look at the matrix value between Flares and the Dst Minimum.\n"
            "- **Why it matters:** A strong negative correlation between Solar Flares (cause) and Dst Index (effect) statistically proves that massive solar explosions directly cause Earth's equatorial magnetic field to crash into the negatives."
        )
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Multivariate Heatmap")
        corr_type = st.radio("Statistical Method", ["Pearson (Linear)", "Spearman (Rank)"])
        heatmap_vars = st.multiselect("Feature Selection", ["ssn", "f107", "sunspot_area", "flare_xray_total", "flare_M", "flare_X", "kp_daily_max", "dst_daily_mean", "dst_daily_min"], default=[c for c in ["ssn", "f107", "flare_xray_total", "kp_daily_max", "dst_daily_min"] if c in df_filtered.columns])
        if len(heatmap_vars) > 1:
            method = "pearson" if "Pearson" in corr_type else "spearman"
            corr_matrix = df_filtered[heatmap_vars].corr(method=method)
            fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig_heat.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_heat, width='stretch')
            
    with col_b:
        st.subheader("Bivariate Scatter Configuration")
        numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
        x_var = st.selectbox("Regressor (X)", numeric_cols, index=numeric_cols.index("f107") if "f107" in numeric_cols else 0)
        y_var = st.selectbox("Response (Y)", numeric_cols, index=numeric_cols.index("kp_daily_max") if "kp_daily_max" in numeric_cols else min(1, len(numeric_cols)-1))
        if x_var and y_var:
            fig_scat = px.scatter(df_filtered.reset_index(), x=x_var, y=y_var, color="year", hover_data=["date"], trendline="ols", trendline_color_override="#dc2626", opacity=0.6)
            fig_scat.update_layout(height=550)
            st.plotly_chart(fig_scat, width='stretch')
            try:
                r_sq = px.get_trendline_results(fig_scat).iloc[0]["px_fit_results"].rsquared
                st.info(f"**Calculated OLS $R^2$:** {r_sq:.4f}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# PAGE 4: Lag-Time Analysis
# ---------------------------------------------------------------------------

def page_lag_analysis():
    st.title("Lag-Time & Superposed Epoch Analysis")
    st.markdown("Chronological delay isolating solar causality from terrestrial consequence.")
    
    with st.expander("📖 Interpreting Superposed Epoch Analysis (SEA)", expanded=False):
        st.info(
            "**What is this mathematical model showing?**\n"
            "Superposed Epoch Analysis is a mathematical technique used to align multiple independent events (like 100 different solar flares) at a common '**Day 0**' to find their average delayed impact.\n\n"
            "- **The Transit Delay:** X-Ray flares hit Earth at the speed of light (8 minutes), but the physical plasma clouds (CMEs) that cause geomagnetic storms travel much slower. \n"
            "- Look at the graph: you will see that the Geomagnetic disturbance (the thick black line for Kp or Dst) doesn't completely peak/crash until roughly 2 to 4 days *after* Day 0. This visual delay statistically proves the speed of the physical solar wind."
        )
    
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.subheader("Iterative Delay Optimization")
        lag_df = load_stats_file("lag_time_detailed.csv")
        if not lag_df.empty:
            daily_lag = lag_df[lag_df["resolution"] == "Daily"]
            st.dataframe(daily_lag[["driver_label", "response_label", "optimal_lag", "correlation_at_peak"]], hide_index=True, height=300)
        
        st.subheader("Dynamic Superposed Epoch Analysis")
        trigger_metric = st.selectbox("Trigger Variable", ["flare_X", "flare_M", "ssn", "f107", "kp_daily_max", "dst_daily_mean"], index=2 if "ssn" in df.columns else 0)
        trigger_op = st.selectbox("Operator", [">=", "<=", "=="])
        trigger_val = st.number_input("Threshold", value=1.0 if "flare" in trigger_metric else 100.0)
        
        if trigger_metric in df.columns:
            if trigger_op == ">=": trigger_idx = df[df[trigger_metric] >= trigger_val].index
            elif trigger_op == "<=": trigger_idx = df[df[trigger_metric] <= trigger_val].index
            else: trigger_idx = df[df[trigger_metric] == trigger_val].index
            
            st.markdown(f"**Trigger Events Found:** {len(trigger_idx)}")
            
            if len(trigger_idx) > 0 and "dst_daily_mean" in df.columns:
                window = 10
                epochs = []
                for t in trigger_idx:
                    pos = df.index.get_loc(t)
                    if pos >= window and pos < len(df) - window:
                        epochs.append(df["dst_daily_mean"].iloc[pos-window:pos+window+1].values)
                
                if epochs:
                    mean_epoch = np.nanmean(epochs, axis=0)
                    epoch_days = np.arange(-window, window+1)
                    fig_sea = px.line(x=epoch_days, y=mean_epoch, labels={"x": "Days from Trigger (Epoch 0)", "y": "Mean Dst (nT)"}, title=f"Geomagnetic Response (Dst) to {trigger_metric} {trigger_op} {trigger_val}")
                    fig_sea.add_vline(x=0, line_dash="dash", line_color="orange", annotation_text="Trigger Event")
                    fig_sea.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_sea, width='stretch')
                else:
                    st.warning("Trigger events are too close to dataset edges to plot window.")
        else:
            st.warning(f"Metric {trigger_metric} not found in dataset.")

    with col_l2:
        st.subheader("Event Transit Tracking: Flare to Storm")
        flare_events = load_stats_file("flare_storm_lag_events.csv")
        if not flare_events.empty and "dst_lag_days" in flare_events.columns:
            storms = flare_events[flare_events.get("storm_triggered", False)]
            fig_hist = px.histogram(storms, x="dst_lag_days", nbins=8, title=f"Transit Delay (n={len(storms)} storms)", labels={"dst_lag_days": "Lag (days)"}, color_discrete_sequence=["#2563eb"])
            mean_lag = storms["dst_lag_days"].mean()
            fig_hist.add_vline(x=mean_lag, line_dash="dash", line_color="#b91c1c", annotation_text=f"Mean: {mean_lag:.1f}d")
            fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_hist, width='stretch')

            storms_df = load_stats_file("extreme_storms_dst.csv")
            if not storms_df.empty and set(["date", "dst_daily_min", "kp_daily_max", "f107"]).issubset(storms_df.columns):
                st.subheader("Worst Global Disruptions (Dst ≤ -100)")
                st.dataframe(storms_df[["date", "dst_daily_min", "kp_daily_max", "f107"]].head(10), hide_index=True, height=250)


# ---------------------------------------------------------------------------
# PAGE 5: Geospatial Impact (Aurora Map)
# ---------------------------------------------------------------------------

def page_geospatial_impact():
    st.title("Geospatial Impact: Auroral Extent Mapping")
    st.markdown("Slide through time to observe how exact planetary disruption pushes auroral boundaries southwards.")
    
    with st.expander("📖 Interpreting Geospatial Boundaries", expanded=False):
        st.info(
            "**Why does the mapped boundary expand?**\n"
            "During a quiet solar day (Kp 0-3), the Northern and Southern Lights are confined to the extreme poles (>65° Latitude). \n\n"
            "When a massive CME hits Earth (pushing Kp to 8 or 9), it drastically compresses the magnetosphere on the day-side and stretches the night-side tail, forcing the auroral oval dramatically down towards the equator. This interactive globe maps exactly how far south these dangerous atmospheric disruptions (and their accompanying satellite drag effects) penetrated on any historical date we possess."
        )

    dates = np.unique(df.index.date)
    dates.sort()
    default_idx = int(len(dates)/2)
    selected_date = st.select_slider("Drag to Shift Time", options=dates, value=dates[default_idx])
    row_data = df.loc[pd.Timestamp(selected_date)] if pd.Timestamp(selected_date) in df.index else pd.Series()
    kp_val = float(row_data.get("kp_daily_max", 0)) if not row_data.empty else 0
    aurora_boundary = max(35, 68 - (3.6 * kp_val))

    st.metric(label="Selected Baseline Kp", value=f"{kp_val:.2f}/9.00")
    st.markdown(f"**Aurora Visible Threshold:** ~{aurora_boundary:.1f}° Lat")

    lat_grid = np.arange(-90, 91, 4)
    lon_grid = np.arange(-180, 181, 5)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    lats_flat = lat_mesh.flatten()
    lons_flat = lon_mesh.flatten()
    mask = (lats_flat >= aurora_boundary) | (lats_flat <= -aurora_boundary)
    map_lats = lats_flat[mask]
    map_lons = lons_flat[mask]

    if kp_val <= 4:
        c, a = "#10b981", 0.4
    elif kp_val <= 6:
        c, a = "#f59e0b", 0.6
    elif kp_val <= 8:
        c, a = "#ef4444", 0.7
    else:
        c, a = "#d946ef", 0.8

    fig_map = go.Figure(go.Scattergeo(
        lat=map_lats, lon=map_lons, mode='markers',
        marker=dict(size=5, color=c, opacity=a, symbol='circle', line=dict(width=0)),
        hoverinfo='none',
    ))
    fig_map.update_geos(
        projection_type="orthographic",
        showcoastlines=True, coastlinecolor="#374151",
        showland=True, landcolor="#1f2937",
        showocean=True, oceancolor="#0f172a",
    )
    fig_map.update_layout(height=650, margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_map, width='stretch')


# ---------------------------------------------------------------------------
# PAGE 6: Physics Primer
# ---------------------------------------------------------------------------

def page_physics_primer():
    st.title("Physics Primer & Storm Simulator")
    st.markdown("Interactive exploration of how arbitrary scale metrics map to planetary consequences.")
    
    st.markdown("### 🎚️ NOAA G-Scale Storm Simulator")
    storm_cat = st.select_slider(
        "Slide to simulate storm intensity (Kp correlation):", 
        options=['Quiet (Kp 0-3)', 'Active (Kp 4)', 'G1 Minor (Kp 5)', 'G2 Moderate (Kp 6)', 'G3 Strong (Kp 7)', 'G4 Severe (Kp 8)', 'G5 Extreme (Kp 9)'],
        value='Quiet (Kp 0-3)'
    )

    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        if 'Quiet' in storm_cat or 'Active' in storm_cat:
            st.success("**Visual Impact:** Aurora confined to extreme northern latitudes (Alaska, Scandinavia).")
            st.markdown("**Infrastructure:** Nominal operations across power grids and orbital spacecraft.")
            st.markdown("**Corresponding Solar Cause:** General background solar wind, absence of large CMEs.")
        elif 'G1' in storm_cat or 'G2' in storm_cat:
            st.warning("**Visual Impact:** Aurora visible down to mid-high latitudes (Michigan, Scotland).")
            st.markdown("**Infrastructure:** Weak power grid fluctuations. High-latitude orbits face increased atmospheric drag.")
            st.markdown("**Corresponding Solar Cause:** Glancing blow CME from an M-class flare, or Coronal Hole High Speed Stream (CHHSS).")
        elif 'G3' in storm_cat or 'G4' in storm_cat:
            st.error("**Visual Impact:** Aurora pushed down to mid latitudes (Oregon, Illinois).")
            st.markdown("**Infrastructure:** Voltage corrections required in power grids. GPS degradation common. Radio blackouts likely at poles.")
            st.markdown("**Corresponding Solar Cause:** Direct Earth-directed CME impact from an X-Class flare.")
        else:
            st.error("**🚨 EXTREME DISRUPTION (G5)**")
            st.markdown("**Visual Impact:** Aurora observed natively in low latitudes (Florida, Spain).")
            st.markdown("**Infrastructure:** Widespread voltage control problems; complete high-frequency (HF) radio blackout. Massive orbit corrections required for thousands of active satellites.")
            st.markdown("**Corresponding Solar Cause:** Catastrophic multi-CME impact or Carrington-level X-class eruption.")
            st.markdown("**Historic Example in Dataset:** The legendary **Halloween Storms of 2003** (Dst -383 nT).")

    with col_p2:
        val_map = {'Quiet (Kp 0-3)': 2, 'Active (Kp 4)': 4, 'G1 Minor (Kp 5)': 5, 'G2 Moderate (Kp 6)': 6, 'G3 Strong (Kp 7)': 7, 'G4 Severe (Kp 8)': 8, 'G5 Extreme (Kp 9)': 9}
        v = val_map[storm_cat]
        fig_sim = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = v, domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Kp Index Equivalent"},
            gauge = {
                'axis': {'range': [0, 9]},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [0, 4], 'color': "#10b981"},
                    {'range': [4, 6], 'color': "#f59e0b"},
                    {'range': [6, 8], 'color': "#ef4444"},
                    {'range': [8, 9], 'color': "#d946ef"}
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': v}
            }
        ))
        fig_sim.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sim, width='stretch')

    st.markdown("---")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.subheader("What are Solar Flares?")
        st.markdown(
            "An intense burst of X-ray radiation coming from the release of magnetic energy. "
            "They are our solar system's largest explosive events. "
        )
        st.markdown(
            "- **A & B Class:** Undetectable without orbital apparatus.\n"
            "- **C Class:** Minor flares with few consequences.\n"
            "- **M Class:** Medium-sized flares. Can cause brief radio blackouts.\n"
            "- **X Class:** *Major* events triggering significant radio blackouts."
        )

    with col_f2:
        st.subheader("Coronal Mass Ejections (CMEs)")
        st.markdown(
            "Flares are often accompanied by **CMEs**. While a flare is the flash of a cannon, "
            "a CME is the cannonball. CMEs hurl massive clouds of plasma outwards. "
            "If Earth-directed, they arrive in 1 to 4 days."
        )

    st.markdown("---")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("The 11-Year Solar Cycle")
        st.markdown(
            "The Sun goes through rhythmic periods of high and low activity roughly every 11 years, driven by its internal dynamo. "
            "At **Solar Minimum**, the Sun is quiet and spotless. At **Solar Maximum**, its magnetic field becomes highly twisted and tangled, "
            "leading to a dramatic surge in sunspots, flares, and CMEs. Our data pipeline explicitly tracks these cycles using the **Sunspot Number (SSN)** "
            "and the **F10.7 Radio Flux** as proxy measurements for this underlying magnetic torsion."
        )

    with col_s2:
        st.subheader("Geomagnetic Shield (The Magnetosphere)")
        st.markdown(
            "Earth is protected by a magnetic bubble (the magnetosphere). When the solar wind or a CME strikes this bubble, it compresses and distorts. "
            "This interaction transfers massive amounts of energy into our atmosphere, causing beautifully glowing **Auroras** but also dangerous **Geomagnetic Storms**. "
            "We measure this turbulence using the **Kp Index** (global perturbation) and the **Dst Index** (an indicator of the equatorial ring current strength, which drops predictably during major storms)."
        )

    st.markdown("---")
    
    st.subheader("Other Critical Telemetry Data Points")
    st.markdown("While Sunspot Numbers track optical surface activity, we rely on three other primary measurements to quantify the true physical intensity of Space Weather events impacting Earth.")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown("#### F10.7 Solar Flux")
        st.markdown(
            "The **F10.7 cm Radio Flux** is a measurement of the solar radio emission at a wavelength of 10.7 centimeters (2800 MHz). "
            "Unlike sunspots which require clear optical viewing, radio emissions can be reliably measured in any terrestrial weather condition."
        )
        st.markdown(
            "It is highly correlated with the sunspot number but serves as a much better proxy for the **Extreme Ultraviolet (EUV)** output of the Sun, "
            "which directly controls the heating and expansion of the Earth's upper atmosphere (the thermosphere). Higher flux values mean increased satellite drag."
        )
    with col_d2:
        st.markdown("#### The Kp Index")
        st.markdown(
            "The **Planetary K-index (Kp)** is the most widely recognized indicator of global geomagnetic storm magnitude. "
            "It quantifies disturbances in the Earth's magnetic field with a 3-hour resolution."
        )
        st.markdown(
            "It is a quasi-logarithmic integer scale ranging from **0 (quiet)** to **9 (extreme storm)**. "
            "It is derived from the maximum fluctuations of horizontal magnetic components measured by a global network of ground-based magnetometers. "
            "The NOAA G-Scale (G1 to G5) maps directly mathematically to Kp values of 5 through 9."
        )
    with col_d3:
        st.markdown("#### The Dst Index")
        st.markdown(
            "The **Disturbance Storm Time (Dst)** index is the definitive scientific measurement of geomagnetic storm intensity. "
            "Unlike Kp which is an abstracted integer, Dst is measured in raw nanoteslas (nT)."
        )
        st.markdown(
            "During a storm, the solar wind injects massive amounts of plasma into Earth's **equatorial ring current**, which then induces a magnetic field that opposes Earth's own field. "
            "Consequently, the Dst value sharply **plummets into the negatives**. "
            "A normal quiet day is ~0 nT, a moderate storm is ≤-50 nT, and a severe superstorm (such as the Halloween 2003 storms or Carrington Event) involves Dst crashing below -300 nT."
        )



# ---------------------------------------------------------------------------
# PAGE 5.5: Extreme Events Analysis
# ---------------------------------------------------------------------------

def page_extreme_events():
    st.title("Extreme Events Detection & Analysis")
    st.markdown("Identify and analyze outlier days with abnormal geomagnetic and solar activity.")
    
    with st.expander("📖 Interpreting Extreme Events", expanded=False):
        st.info(
            "**What constitutes an extreme event?**\n"
            "Using statistical z-scores (standard deviations from the mean), we identify days with unusual activity levels. "
            "Events beyond 2.5σ are considered extreme. This helps discover:\n"
            "- **Geomagnetic Storms:** Dst plummeting to record lows\n"
            "- **Solar Eruptions:** Massive X-ray flares or unusually high sunspot activity\n"
            "- **Solar System Disruptions:** Extreme F10.7 radio flux spikes"
        )
    
    col_e1, col_e2 = st.columns([1, 2])
    
    with col_e1:
        st.subheader("Threshold Configuration")
        threshold_sigma = st.slider("Z-Score Threshold (σ)", 1.5, 4.0, 2.5, 0.1)
        event_var = st.selectbox("Variable", ["dst_daily_min", "kp_daily_max", "ssn", "f107", "flare_xray_total"])
    
    with col_e2:
        if event_var in df_filtered.columns:
            extreme_df = find_extreme_events(df_filtered, event_var, threshold_sigma)
            
            if len(extreme_df) > 0:
                fig_ext = go.Figure()
                fig_ext.add_trace(go.Scattergl(
                    x=df_filtered.index, y=df_filtered[event_var],
                    mode='lines', line=dict(color='#e5e7eb', width=1),
                    name="Normal Days", opacity=0.5
                ))
                fig_ext.add_trace(go.Scattergl(
                    x=extreme_df.index, y=extreme_df[event_var],
                    mode='markers', marker=dict(size=8, color=extreme_df["z_score"], 
                    colorscale="Reds", showscale=True, colorbar=dict(title="Z-Score")),
                    name="Extreme Events"
                ))
                fig_ext.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
                st.plotly_chart(fig_ext, width='stretch')
                
                st.subheader("Top Extreme Events")
                st.dataframe(extreme_df[[event_var, "z_score"]].head(10), height=250)
            else:
                st.warning("No extreme events found with current threshold.")
        else:
            st.error(f"Column {event_var} not found in data.")


# ---------------------------------------------------------------------------
# PAGE 6: Periodicity & Cycle Analysis
# ---------------------------------------------------------------------------

def page_periodicity():
    st.title("Periodicity & Solar Cycle Analysis")
    st.markdown("Discover dominant cycles and periodic patterns in solar/geomagnetic activity.")
    
    with st.expander("📖 Understanding Spectral Analysis", expanded=False):
        st.info(
            "**FFT (Fast Fourier Transform):** Global frequency analysis to identify dominant cycles across the entire dataset.\n"
            "**Wavelet (CWT):** Time-localized spectral power showing how cycles evolve through time.\n"
            "The 11-year solar cycle should dominate. Secondary peaks may reveal the 27-day solar rotation (~monthly signature) or other harmonic patterns."
        )
    
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        st.subheader("Analysis Configuration")
        period_var = st.selectbox("Variable", ["ssn", "f107", "kp_daily_max", "dst_daily_min"], index=0)
        period_method = st.radio("Method", ["FFT (Global)", "Wavelet (Temporal)"])
    
    with col_p2:
        if period_var in df_filtered.columns:
            try:
                periodicity_results = analyze_periodicity(df_filtered[period_var].dropna(), 
                                                         method='fft' if 'FFT' in period_method else 'wv')
                
                if 'FFT' in period_method:
                    st.subheader("Dominant Periods (FFT)")
                    top_periods = periodicity_results.nlargest(10, "amplitude")
                    fig_fft = px.bar(top_periods.head(10), x="period_years", y="amplitude",
                                    labels={"period_years": "Period (Years)", "amplitude": "Amplitude"},
                                    title=f"Top 10 Periods in {period_var}")
                    fig_fft.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_fft, width='stretch')
                    
                    st.dataframe(top_periods[["period_years", "amplitude"]].head(10), hide_index=True, height=250)
                else:
                    st.subheader("Temporal Periodicity (Wavelet)")
                    st.line_chart(periodicity_results["dominant_period_yrs"], height=400)
                    
            except Exception as e:
                st.error(f"Periodicity analysis failed: {e}")
        else:
            st.error(f"Column {period_var} not found.")


# ---------------------------------------------------------------------------
# PAGE 7: Phase-Locked Climatology
# ---------------------------------------------------------------------------

def page_phase_climatology():
    st.title("Phase-Locked Climatology: Solar Cycle Risk Zones")
    st.markdown("Identify 'danger zones' within the 11-year solar cycle where geomagnetic storms are most likely.")
    
    with st.expander("📖 Understanding Phase-Locked Analysis", expanded=False):
        st.info(
            "**What is phase-locking?**\n"
            "By dividing the 11-year solar cycle into 20 phase bins (5% intervals), we bin all Kp observations by their cycle phase and compute statistics. "
            "This reveals if certain phases of the cycle are inherently 'stormier' than others, independent of absolute solar activity magnitude."
        )
    
    col_pl1, col_pl2 = st.columns([1, 2])
    
    with col_pl1:
        st.subheader("Configuration")
        n_bins = st.slider("Phase Bins", 10, 40, 20, 2)
        kp_col_sel = st.selectbox("Kp Column", [c for c in df_filtered.columns if 'kp' in c.lower()], index=0)
        storm_threshold_pl = st.number_input("Storm Threshold (Kp)", 0.0, 9.0, 6.0, 0.5)
    
    with col_pl2:
        if kp_col_sel in df_filtered.columns:
            try:
                climatology = analyze_phase_locked_climatology(
                    df_filtered, kp_col=kp_col_sel, storm_threshold=storm_threshold_pl, bins=n_bins
                )
                
                fig_clim = go.Figure()
                fig_clim.add_trace(go.Scattergl(
                    x=climatology.index.astype(float), y=climatology["kp_mean"],
                    mode='lines+markers', line=dict(color='#2563eb', width=2),
                    marker=dict(size=6), name="Mean Kp", fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.2)'
                ))
                fig_clim.add_trace(go.Scattergl(
                    x=climatology.index.astype(float), y=climatology["storm_prob_pct"],
                    mode='lines+markers', line=dict(color='#b91c1c', width=2),
                    marker=dict(size=6), name="Storm Prob (%)", yaxis='y2'
                ))
                fig_clim.update_layout(
                    height=400, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(title="Solar Cycle Phase (0=min, 1=max)"),
                    yaxis=dict(title="Mean Kp"),
                    yaxis2=dict(title="Storm Probability (%)", overlaying='y', side='right'),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_clim, width='stretch')
                
                st.subheader("Climatology Statistics Table")
                st.dataframe(climatology.astype(float).round(3), height=400)
            except Exception as e:
                st.error(f"Phase climatology analysis failed: {e}")
        else:
            st.error(f"Column {kp_col_sel} not found.")


# ---------------------------------------------------------------------------
# PAGE 8: Hysteresis Analysis
# ---------------------------------------------------------------------------

def page_hysteresis():
    st.title("Hysteresis Analysis: Solar Cycle Phase Effects")
    st.markdown("Compare geomagnetic response (Kp) to sunspot activity across rising vs. falling cycle phases.")
    
    with st.expander("📖 Understanding Hysteresis", expanded=False):
        st.info(
            "**What is hysteresis?**\n"
            "Hysteresis occurs when a system's output depends not just on current input, but also on its history. "
            "In solar physics, this reveals whether the magnetosphere responds differently to the same sunspot number depending on whether the cycle is rising (building) or falling (declining). "
            "If points scatter differently by color, it proves the system has 'memory' of where we are in the 11-year cycle."
        )
    
    col_h1, col_h2 = st.columns([1, 2])
    
    with col_h1:
        st.subheader("Configuration")
        sn_col_h = st.selectbox("Sunspot Proxy", [c for c in df_filtered.columns if 'ssn' in c.lower() or 'sn' in c.lower()], index=0)
        kp_col_h = st.selectbox("Geomagnetic Index", [c for c in df_filtered.columns if 'kp' in c.lower()], index=0)
    
    with col_h2:
        if sn_col_h in df_filtered.columns and kp_col_h in df_filtered.columns:
            try:
                hysteresis_df = analyze_hysteresis(
                    df_filtered, sn_col=sn_col_h, kp_col=kp_col_h, save_path="hysteresis_plot_tmp.png"
                )
                
                rising = hysteresis_df[hysteresis_df['phase_type'] == 'Rising']
                falling = hysteresis_df[hysteresis_df['phase_type'] == 'Falling']
                
                fig_hyst = go.Figure()
                fig_hyst.add_trace(go.Scattergl(
                    x=rising[sn_col_h], y=rising[kp_col_h],
                    mode='markers', marker=dict(size=5, color='#2563eb', opacity=0.4),
                    name='Rising Phase'
                ))
                fig_hyst.add_trace(go.Scattergl(
                    x=falling[sn_col_h], y=falling[kp_col_h],
                    mode='markers', marker=dict(size=5, color='#dc2626', opacity=0.4),
                    name='Falling Phase'
                ))
                fig_hyst.update_layout(
                    height=500, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(title=sn_col_h),
                    yaxis=dict(title=kp_col_h),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_hyst, width='stretch')
                
                st.info(f"**Rising Phase:** {len(rising)} days | **Falling Phase:** {len(falling)} days")
            except Exception as e:
                st.error(f"Hysteresis analysis failed: {e}")
        else:
            st.error("Required columns not found.")


# ---------------------------------------------------------------------------
# PAGE 9: Monthly Statistics & Aggregation
# ---------------------------------------------------------------------------

def page_monthly_stats():
    st.title("Monthly Statistics & Temporal Aggregation")
    st.markdown("Roll up daily observations into monthly summaries to identify medium-term patterns.")
    
    with st.expander("📖 Understanding Aggregated Statistics", expanded=False):
        st.info(
            "**Why aggregate to monthly?**\n"
            "Daily data is noisy and influenced by local oscillations. Monthly aggregation reveals true trends by:\n"
            "- Computing **mean** values (smooth trends)\n"
            "- Tracking **max** values (peak events)\n"
            "- Summing **storm hours** (cumulative impact)"
        )
    
    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        st.subheader("Aggregation Settings")
        agg_period = st.radio("Period", ["Monthly", "Quarterly", "Yearly"], horizontal=False)
        period_map = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
        period_code = period_map[agg_period]
    
    with col_m2:
        try:
            if "sn" in df_filtered.columns:
                monthly_stats = compute_monthly_stats(df_filtered.copy())
                
                fig_month = go.Figure()
                if "sn_mean" in monthly_stats.columns:
                    fig_month.add_trace(go.Scattergl(
                        x=monthly_stats.index, y=monthly_stats["sn_mean"],
                        mode='lines', line=dict(color='#d97706', width=2),
                        name="SSN Mean"
                    ))
                if "Kp_mean" in monthly_stats.columns:
                    fig_month.add_trace(go.Scattergl(
                        x=monthly_stats.index, y=monthly_stats["Kp_mean"],
                        mode='lines', line=dict(color='#2563eb', width=2),
                        name="Kp Mean", yaxis='y2'
                    ))
                
                fig_month.update_layout(
                    height=400, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="SSN"),
                    yaxis2=dict(title="Kp", overlaying='y', side='right'),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_month, width='stretch')
                
                st.subheader("Monthly Statistics Summary")
                st.dataframe(monthly_stats.round(3).head(20), height=400)
            else:
                st.error("SSN column not found.")
        except Exception as e:
            st.error(f"Monthly statistics failed: {e}")


# ---------------------------------------------------------------------------
# Router Execution
# ---------------------------------------------------------------------------

if page == "System Overview":
    page_overview()
elif page == "Data Smoothing":
    page_smoothing()
elif page == "Correlation Matrix":
    page_correlations()
elif page == "Lag-Time Analysis":
    page_lag_analysis()
elif page == "Extreme Events":
    page_extreme_events()
elif page == "Periodicity Analysis":
    page_periodicity()
elif page == "Phase-Locked Climatology":
    page_phase_climatology()
elif page == "Hysteresis Analysis":
    page_hysteresis()
elif page == "Monthly Statistics":
    page_monthly_stats()
elif page == "Geospatial Impact":
    page_geospatial_impact()
elif page == "Physics Primer":
    page_physics_primer()

# ---------------------------------------------------------------------------
# PAGE 7: Data Sources
# ---------------------------------------------------------------------------

def page_data_sources():
    st.title("Data Sources & Provenance")
    st.markdown("This dashboard relies on high-fidelity telemetry from authoritative space weather institutions.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SILSO")
        st.markdown("**Sunspot Index and Long-term Solar Observations (SILSO)**, World Data Center, Royal Observatory of Belgium, Brussels. We utilize the Version 2.0 total daily sunspot number.")
        st.write("[Visit SILSO](https://www.sidc.be/SILSO/home)")
        
        st.subheader("NASA OMNIWeb")
        st.markdown("**NASA Goddard Space Flight Center, Space Physics Data Facility (SPDF)**. We extract low-resolution (hourly) OMNI2 parameters, heavily focusing on the **Dst** (Disturbance Storm Time) index.")
        st.write("[Visit OMNIWeb](https://omniweb.gsfc.nasa.gov/)")
    
    with col2:
        st.subheader("NOAA NCEI & SWPC")
        st.markdown("**National Centers for Environmental Information** & **Space Weather Prediction Center**. We ingest annual Daily Solar Data (DSD) to retrieve historical F10.7 fluxes and daily flare counts, and tap SWPC JSON feeds for live telemetry.")
        st.write("[Visit SWPC](https://www.swpc.noaa.gov/)")
        
        st.subheader("WDC Kyoto")
        st.markdown("**World Data Center for Geomagnetism, Kyoto**. Supplemental provider of definitive and provisional Dst indices for resolving modern timeline telemetry prior to final NASA validation.")
        st.write("[Visit WDC Kyoto](https://wdc.kugi.kyoto-u.ac.jp/)")


def page_project_details():
    st.title("Project Details & Team")
    st.markdown("### The Mission")
    st.markdown(
        "This project bridges the gap between raw astrophysical telemetry and actionable data science. "
        "By dynamically ingesting decadal data from satellites and ground stations, our automated pipeline "
        "cleans, imputes, and harmonizes multiple variables (Sunspots, F10.7, Flares, Kp, and Dst). "
        "The resulting dataset powers custom superposed epoch analyses and lag-time correlation matrices "
        "to clearly illustrate how events on the solar surface dictate geomagnetic instability on Earth."
    )
    
    st.markdown("---")
    st.subheader("The Team")
    st.markdown(
        "Built with ❤️ by a dedicated team:\n"
        "- **Mulumudi Dinesh Karthik**\n"
        "- **Vansh Gupta**\n"
        "- **Abhishek Menon**\n"
        "- **Shailendra Pratap Singh**"
    )

if page == "Data Sources":
    page_data_sources()
elif page == "Project Details":
    page_project_details()
