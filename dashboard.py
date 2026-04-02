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
    analyze_phase_locked_climatology, analyze_hysteresis, compute_periodly_stats
)

# ---------------------------------------------------------------------------
# Config & Professional UI Setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Space Weather Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Professional CSS Theme (Fonts, Hiding Sidebar Arrow, Responsive blocks)
# Optimized for performance: minimal animations, async rendering
st.markdown("""
<style>
    /* Uniform professional font */
    html, body, [class*=\"css\"]  {
        font-family: 'Inter', 'Roboto', 'Helvetica Neue', sans-serif !important;
    }
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    
    /* Main body styling - optimized for performance */
    .main .block-container { padding-top: 1rem; max-width: 98%; }
    
    /* Headers */
    h1 { color: #f59e0b; font-weight: 700; }
    h2 { color: var(--text-color); font-weight: 600; font-size: 1.5rem; opacity: 0.9; }
    h3 { color: var(--text-color); font-weight: 500; font-size: 1.25rem; opacity: 0.8; }
    
    /* Info boxes styling */
    .stInfo { background-color: rgba(245, 158, 11, 0.1) !important; border-left: 4px solid #f59e0b !important; }
    
    /* Optimize Plotly rendering */
    .plotly { will-change: transform; }
    .js-plotly-plot { pointer-events: auto; }
    
    /* Reduce re-renders */
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading (Cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900)
def load_data():
    csv_path = Path("data/clean/solar_weather_daily.csv")
    
    if not csv_path.exists():
        st.warning("⚠️ **First-time setup detected.** High-fidelity solar telemetry is missing from the cloud environment.")
        
        with st.status("🚀 **Initializing Data Pipeline...**", expanded=True) as status:
            import subprocess
            import os
            
            st.write("1. 📥 **Fetching raw data from NASA/NOAA/SILSO...**")
            ingest_process = subprocess.run([sys.executable, "ingest.py"], capture_output=True, text=True)
            if ingest_process.returncode != 0:
                st.error("Data ingestion failed. Check network connectivity.")
                st.code(ingest_process.stderr)
                st.stop()
            
            st.write("2. 🧹 **Harmonizing and cleaning datasets...**")
            clean_process = subprocess.run([sys.executable, "clean.py"], capture_output=True, text=True)
            if clean_process.returncode != 0:
                st.error("Data cleaning failed.")
                st.code(clean_process.stderr)
                st.stop()
            
            status.update(label="✅ **Data Pipeline Synchronization Complete!**", state="complete", expanded=False)
        
        st.success("Modernized dataset is now synchronized. Reloading dashboard components...")
        st.rerun()

    dataframe = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    return dataframe

@st.cache_data(ttl=900)
def load_stats_file(filename):
    path = Path(f"data/analysis/stats/{filename}")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=900)
def downsample_data(series, max_points=2000):
    """Downsample series data for performance if too many points"""
    if len(series) <= max_points:
        return series
    # Use every nth point to reduce rendering load
    step = max(1, len(series) // max_points)
    return series.iloc[::step]

@st.cache_data(ttl=900)
def load_sunspots_data():
    """Load historical sunspot number data"""
    sunspot_path = Path("data/sunspots_daily_clean.csv")
    if not sunspot_path.exists():
        st.warning(f"Sunspot data not found at {sunspot_path}")
        return pd.DataFrame()
    sunspot_df = pd.read_csv(sunspot_path, parse_dates=["date"])
    sunspot_df.set_index("date", inplace=True)
    return sunspot_df

dataframe = load_data()

# Global default fallback columns
if "year" not in dataframe.columns:
    dataframe["year"] = dataframe.index.year

# ---------------------------------------------------------------------------
# Top Navigation Menu & Data Controls
# ---------------------------------------------------------------------------

st.title("☀️ Space Weather Analytics")
st.markdown("Data Intelligence & Forecasting")

min_date = dataframe.index.min().date()
max_date = dataframe.index.max().date()

# Date range selector at top
column_date_label, column_date_input = st.columns([1, 3])
with column_date_label:
    st.markdown("<div style='padding-top: 0.5rem'><b>Observation Window:</b></div>", unsafe_allow_html=True)
with column_date_input:
    date_range = st.slider(
        "Observation Window",
        min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD", label_visibility="collapsed"
    )

start_date, end_date = date_range
dataframe_filtered = dataframe.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]

st.markdown(f"**Valid Records:** {len(dataframe_filtered):,} | **Data Source:** NOAA NCEI, SWPC, WDC Kyoto, SILSO")
st.markdown("---")

page = option_menu(
    menu_title=None,
    options=["Solar Number Time Series", "System Overview", "Physics Primer", "Storm Simulator", "Geospatial Impact", "Monthly Statistics", "Data Smoothing", "Correlation Matrix", "Lag-Time Analysis", "Periodicity Analysis", "Phase-Locked Climatology", "Hysteresis Analysis", "Extreme Events", "Data Sources", "Project Details"],
    icons=["graph-up", "activity", "lightning", "thermometer-half", "globe-americas", "calendar", "bar-chart-line", "diagram-3", "clock-history", "wave", "circle-half", "arrows-expand", "alert-triangle", "server", "info-square"],
    menu_icon="cast",
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "#64748b", "font-size": "18px"},
        "nav-link": {
            "font-size": "15px", "text-align": "center", "margin":"5px", 
            "--hover-color": "rgba(245, 158, 11, 0.2)", "border-radius": "8px"
        },
        "nav-link-selected": {
            "background-color": "#f59e0b", "color": "white", "font-weight": "600",
            "box-shadow": "0 4px 6px -1px rgba(245, 158, 11, 0.4)"
        },
    }
)

st.markdown("---")


# ---------------------------------------------------------------------------
# PAGE 1: System Overview
# ---------------------------------------------------------------------------

def page_overview():
    st.title("System Overview: Time-Series Diagnostics")
    st.markdown("Macro-level inspection of solar drivers and terrestrial responses.")

    st.info(
        "**How to read these charts:**\n\n"
        "This tab directly contrasts the **cause** (Solar emissions shown on top) against the **effect** (Terrestrial disruptions on the bottom). \n"
        "- **The 11-Year Cycle:** The 27-day rolling mean lines on the SSN and F10.7 charts make the 11-year solar maximums and minimums obvious, filtering out daily rotational noise.\n"
        "- **Storm Thresholds:** Look at the Geomagnetic Kp and Dst indices. During Solar Maximums (when SSN peaks), notice how the Dst plot frequently plummets past the red dotted line (-100 nT), indicating severe auroral storms disrupting Earth's magnetosphere.\n\n"
        "**Definitions:** Sunspot Number (SSN), F10.7 Solar Radio Flux at 10.7 cm, Kp Index (Quasi-logarithmic planetary index), Dst Index (Disturbance Storm Time)."
    )

    column1, column2 = st.columns(2)
    
    with column1:
        st.subheader("Sunspot Number (SSN)")
        figure_ssn = go.Figure()
        if "ssn" in dataframe_filtered.columns:
            figure_ssn.add_trace(go.Scattergl(x=dataframe_filtered.index, y=dataframe_filtered["ssn"], 
                                         line=dict(color='#fbbf24', width=1), opacity=0.5, name="Daily SSN"))
            ssn_27day_mean = dataframe_filtered["ssn"].rolling(27, center=True, min_periods=5).mean()
            figure_ssn.add_trace(go.Scattergl(x=ssn_27day_mean.index, y=ssn_27day_mean, line=dict(color='#f59e0b', width=2), name="27d Mean"))
        figure_ssn.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", 
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(title="Date"), yaxis=dict(title="Sunspot Number (SSN)"),
                            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1))
        st.plotly_chart(figure_ssn, width="stretch")

    with column2:
        st.subheader("F10.7 Solar Flux")
        figure_f107 = go.Figure()
        if "f107" in dataframe_filtered.columns:
            figure_f107.add_trace(go.Scattergl(x=dataframe_filtered.index, y=dataframe_filtered["f107"], 
                                          line=dict(color='#fb923c', width=1), opacity=0.5, name="Daily F10.7"))
            f107_27day_mean = dataframe_filtered["f107"].rolling(27, center=True, min_periods=5).mean()
            figure_f107.add_trace(go.Scattergl(x=f107_27day_mean.index, y=f107_27day_mean, line=dict(color='#ea580c', width=2), name="27d Mean"))
        figure_f107.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(title="Date"), yaxis=dict(title="F10.7 Solar Flux (sfu)"),
                             legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1))
        st.plotly_chart(figure_f107, width="stretch")

    st.markdown("---")
    column3, column4 = st.columns(2)
    
    with column3:
        st.subheader("Major X-ray Flares (M & X class)")
        figure_flare_comparison = go.Figure()
        if "flare_M" in dataframe_filtered.columns:
            figure_flare_comparison.add_trace(go.Bar(x=dataframe_filtered.index, y=dataframe_filtered.get("flare_M", 0), name="M-Class", marker_color='#fbbf24'))
        if "flare_X" in dataframe_filtered.columns:
            figure_flare_comparison.add_trace(go.Bar(x=dataframe_filtered.index, y=dataframe_filtered.get("flare_X", 0), name="X-Class", marker_color='#ef4444'))
        figure_flare_comparison.update_layout(barmode='stack', height=350, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(title="Date"), yaxis=dict(title="Number of Flares"),
                              legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1))
        st.plotly_chart(figure_flare_comparison, width="stretch")

    with column4:
        st.subheader("Geomagnetic Kp Index (Max)")
        figure_kp_maximum = go.Figure()
        if "kp_daily_max" in dataframe_filtered.columns:
            figure_kp_maximum.add_trace(go.Scattergl(x=dataframe_filtered.index, y=dataframe_filtered["kp_daily_max"], 
                                        mode='lines', line=dict(color='#22c55e', width=1),
                                        fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.2)', name="Kp Max"))
            figure_kp_maximum.add_hline(y=5, line_dash="dash", line_color="#fbbf24", annotation_text="G1 Storm")
            figure_kp_maximum.add_hline(y=7, line_dash="dash", line_color="#ef4444", annotation_text="G3 Storm")
        figure_kp_maximum.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", yaxis_range=[0, 9.5],
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(title="Date"), yaxis=dict(title="Kp Index (0-9 scale)"),
                            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1))
        st.plotly_chart(figure_kp_maximum, width="stretch")

    st.markdown("---")
    st.subheader("Geomagnetic Dst Index (Equatorial Disturbance)")
    if "dst_daily_mean" in dataframe_filtered.columns and "dst_daily_max" in dataframe_filtered.columns and "dst_daily_min" in dataframe_filtered.columns:
        figure_dst_timeline = go.Figure()
        figure_dst_timeline.add_trace(go.Scattergl(
            x=dataframe_filtered.index.tolist() + dataframe_filtered.index[::-1].tolist(),
            y=dataframe_filtered["dst_daily_max"].tolist() + dataframe_filtered["dst_daily_min"][::-1].tolist(),
            fill='toself', fillcolor='rgba(34, 197, 94, 0.2)', line=dict(color='rgba(255,255,255,0)'),
            name="Daily Range", hoverinfo="skip"
        ))
        figure_dst_timeline.add_trace(go.Scattergl(x=dataframe_filtered.index, y=dataframe_filtered["dst_daily_mean"], line=dict(color='#22c55e', width=1.5), name="Daily Mean"))
        figure_dst_timeline.add_trace(go.Scattergl(x=dataframe_filtered.index, y=dataframe_filtered["dst_daily_min"], mode='markers', marker=dict(size=3, color='#ef4444'), name="Daily Min", opacity=0.5))
        figure_dst_timeline.add_hline(y=-50, line_dash="dash", line_color="#fbbf24", annotation_text="Moderate")
        figure_dst_timeline.add_hline(y=-100, line_dash="dash", line_color="#f97316", annotation_text="Intense")
        figure_dst_timeline.add_hline(y=-200, line_dash="dash", line_color="#dc2626", annotation_text="Severe")
        figure_dst_timeline.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(title="Date"), yaxis=dict(title="Dst Index (nanoTesla, nT)"),
                             legend=dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1))
        figure_dst_timeline.update_yaxes(autorange="reversed")
        st.plotly_chart(figure_dst_timeline, width="stretch")


# ---------------------------------------------------------------------------
# PAGE 2: Data Smoothing
# ---------------------------------------------------------------------------

def page_smoothing():
    st.title("Data Smoothing & Noise Filtration")
    st.markdown("Apply digital filters to isolate macro trends from raw solar indices.")
    
    st.info(
        "**Why do we smooth this data?**\n"
        "Raw space weather telemetry is incredibly noisy. Sunspots appear and disappear daily as the sun rotates every 27 days, introducing rapid spikes that hide the true 11-year underlying cycle.\n\n"
        "**SMA vs Savitzky-Golay:**\n"
        "- **Simple Moving Average (SMA):** Great for general trends, but it computationally *flattens* the massive peaks of violent solar flares.\n"
        "- **Savitzky-Golay:** A superior polynomial filter used in astrophysics to eliminate noise while mathematically preserving the height and timing of absolute peaks. Notice how the orange line perfectly tracks extreme flare spikes that the blue SMA line completely misses."
    )
    
    column_smoothing_config, column_smoothing_viz = st.columns([1, 2])
    with column_smoothing_config:
        st.subheader("Filter Parameters")
        smooth_variable = st.selectbox("Variable Target", [column for column in dataframe_filtered.select_dtypes(include=np.number).columns], index=0)
        if smooth_variable in dataframe_filtered.columns:
            st.markdown("**(1) Rolling Mean (SMA)**")
            sma_window = st.slider("Window Length (Days)", 3, 365, 27, 2)
            st.markdown("**(2) Savitzky-Golay Filter**")
            savgol_window = st.slider("Kernel Window (Odd)", 5, 365, 51, 2)
            savgol_poly = st.slider("Polynomial Order", 1, 5, 3, 1)
            valid_parameters = savgol_window > savgol_poly
            if not valid_parameters: st.error("Kernel Window must strictly exceed Polynomial Order.")
        else:
            st.warning("Variable unavailable.")
    
    with column_smoothing_viz:
        if smooth_variable in dataframe_filtered.columns and valid_parameters:
            valid_series = dataframe_filtered[smooth_variable].dropna()
            sma_series = valid_series.rolling(sma_window, center=True, min_periods=1).mean()
            savgol_series = savgol_filter(valid_series, window_length=savgol_window, polyorder=savgol_poly)
            
            figure_smoothing_comparison = go.Figure()
            figure_smoothing_comparison.add_trace(go.Scattergl(x=valid_series.index, y=valid_series, mode='lines', line=dict(color='#d1d5db', width=1), name=f"Raw Data", opacity=0.6))
            figure_smoothing_comparison.add_trace(go.Scattergl(x=sma_series.index, y=sma_series, mode='lines', line=dict(color='#22c55e', width=2), name=f"SMA ({sma_window}d)"))
            figure_smoothing_comparison.add_trace(go.Scattergl(x=valid_series.index, y=savgol_series, mode='lines', line=dict(color='#f97316', width=2), name=f"SavGol (w={savgol_window}, p={savgol_poly})"))
            figure_smoothing_comparison.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", 
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                   xaxis=dict(title="Date"), yaxis=dict(title=f"{smooth_variable} (Units: {smooth_variable})"),
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figure_smoothing_comparison, width="stretch")


# ---------------------------------------------------------------------------
# PAGE 3: Correlation Matrix
# ---------------------------------------------------------------------------

def page_correlations():
    st.title("Correlation & Relationship Analytics")
    
    st.info(
        "**How to read this Matrix:**\n"
        "- A value near **+1.00** shows strong positive correlation. For example, SSN and F10.7 are almost perfectly correlated (+0.95+), proving they are reliable proxies for the exact same underlying solar mechanism.\n"
        "- A value near **-1.00** shows strong negative correlation. For example, look at the matrix value between Flares and the Dst Minimum.\n"
        "- **Why it matters:** A strong negative correlation between Solar Flares (cause) and Dst Index (effect) statistically proves that massive solar explosions directly cause Earth's equatorial magnetic field to crash into the negatives."
    )
    
    column_config_left, column_config_right = st.columns([1, 2])
    with column_config_left:
        st.subheader("Multivariate Heatmap")
        correlation_method = st.radio("Statistical Method", ["Pearson (Linear)", "Spearman (Rank)"])
        heatmap_feature_variables = st.multiselect("Feature Selection", ["ssn", "f107", "sunspot_area", "flare_xray_total", "flare_M", "flare_X", "kp_daily_max", "dst_daily_mean", "dst_daily_min"], default=[column for column in ["ssn", "f107", "flare_xray_total", "kp_daily_max", "dst_daily_min"] if column in dataframe_filtered.columns])
        if len(heatmap_feature_variables) > 1:
            statistical_method = "pearson" if "Pearson" in correlation_method else "spearman"
            correlation_matrix = dataframe_filtered[heatmap_feature_variables].corr(method=statistical_method)
            figure_correlation_heatmap = px.imshow(correlation_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            figure_correlation_heatmap.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0),
                                 coloraxis_colorbar=dict(title="Correlation<br>Coefficient"))
            st.plotly_chart(figure_correlation_heatmap, width="stretch")
            
    with column_config_right:
        st.subheader("Bivariate Scatter Configuration")
        numeric_columns = dataframe_filtered.select_dtypes(include=np.number).columns.tolist()
        x_variable = st.selectbox("Regressor (X)", numeric_columns, index=numeric_columns.index("f107") if "f107" in numeric_columns else 0)
        y_variable = st.selectbox("Response (Y)", numeric_columns, index=numeric_columns.index("kp_daily_max") if "kp_daily_max" in numeric_columns else min(1, len(numeric_columns)-1))
        if x_variable and y_variable:
            figure_scatter_regression = px.scatter(dataframe_filtered.reset_index(), x=x_variable, y=y_variable, color="year", hover_data=["date"], trendline="ols", trendline_color_override="#ef4444", opacity=0.6)
            figure_scatter_regression.update_layout(height=550, 
                                 xaxis=dict(title=f"{x_variable}"), yaxis=dict(title=f"{y_variable}"),
                                 legend=dict(title="Year", x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1),
                                 hovermode="closest")
            st.plotly_chart(figure_scatter_regression, width="stretch")
            try:
                r_squared_value = px.get_trendline_results(figure_scatter_regression).iloc[0]["px_fit_results"].rsquared
                st.info(f"**Calculated OLS $R^2$:** {r_squared_value:.4f}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# PAGE 4: Lag-Time Analysis
# ---------------------------------------------------------------------------

def page_lag_analysis():
    st.title("Lag-Time & Superposed Epoch Analysis")
    st.markdown("Chronological delay isolating solar causality from terrestrial consequence.")
    
    st.info(
        "**What is this mathematical model showing?**\n"
        "Superposed Epoch Analysis is a mathematical technique used to align multiple independent events (like 100 different solar flares) at a common '**Day 0**' to find their average delayed impact.\n\n"
        "- **The Transit Delay:** X-Ray flares hit Earth at the speed of light (8 minutes), but the physical plasma clouds (CMEs) that cause geomagnetic storms travel much slower. \n"
        "- **Lag Relation:** This tab allows you to visualize and calculate the Pearson correlation (r) at various daily offsets, proving the physical transit speed of the solar wind."
    )
    
    # --- New Section: Bivariate Lag Correlation ---
    st.subheader("Bivariate Lag-Correlation (Pearson r vs Lag)")
    column_lag_rel_config, column_lag_rel_viz = st.columns([1, 2])
    
    with column_lag_rel_config:
        numeric_cols = [c for c in dataframe_filtered.columns if dataframe_filtered[c].dtype in [np.float64, np.int64]]
        driver_x = st.selectbox("Driver (X)", numeric_cols, index=numeric_cols.index("ssn") if "ssn" in numeric_cols else 0)
        response_y = st.selectbox("Response (Y)", numeric_cols, index=numeric_cols.index("kp_daily_max") if "kp_daily_max" in numeric_cols else 0)
        correlation_max_lag = st.slider("Max Lag (Days)", 0, 60, 28)
        
    with column_lag_rel_viz:
        if driver_x and response_y:
            corr_df = cross_correlation(dataframe_filtered[driver_x], dataframe_filtered[response_y], max_lag=correlation_max_lag)
            
            fig_corr = px.line(corr_df, x="lag_days", y="pearson_r", markers=True,
                               title=f"Correlation Impact: {driver_x} → {response_y}",
                               labels={"lag_days": "Lag (Days)", "pearson_r": "Pearson Correlation (r)"})
            fig_corr.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            # Highlight the optimal lag
            opt_row = corr_df.loc[corr_df["pearson_r"].idxmax()]
            fig_corr.add_annotation(x=opt_row["lag_days"], y=opt_row["pearson_r"],
                                  text=f"Peak: {opt_row['lag_days']}d (r={opt_row['pearson_r']:.2f})",
                                  showarrow=True, arrowhead=1)
            
            fig_corr.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0), hovermode="x unified")
            st.plotly_chart(fig_corr, width="stretch")

    st.markdown("---")

    column_lag_analysis_config, column_lag_analysis_viz = st.columns(2)
    with column_lag_analysis_config:
        st.subheader("Iterative Delay Optimization")
        lag_analysis_dataframe = load_stats_file("lag_time_detailed.csv")
        if not lag_analysis_dataframe.empty:
            daily_lag_records = lag_analysis_dataframe[lag_analysis_dataframe["resolution"] == "Daily"]
            st.dataframe(daily_lag_records[["driver_label", "response_label", "optimal_lag", "correlation_at_peak"]], hide_index=True, height=250)
        else:
            st.info("Pre-calculated statistics not found in `data/analysis/stats/`. Run the analysis pipeline to populate this table.")
        
        st.subheader("Dynamic Superposed Epoch Analysis")
        trigger_metric_name = st.selectbox("Trigger Variable", ["flare_X", "flare_M", "ssn", "f107", "kp_daily_max"], index=2 if "ssn" in dataframe.columns else 0)
        trigger_operator = st.selectbox("Operator", [">=", "<=", "==", ">", "<"])
        trigger_threshold_value = st.number_input("Threshold", value=1.0 if "flare" in trigger_metric_name else 100.0)
        sea_response_variable = st.selectbox("Response Variable", numeric_cols, index=numeric_cols.index("dst_daily_mean") if "dst_daily_mean" in numeric_cols else 0)
        
        if trigger_metric_name in dataframe.columns:
            # Fixed operator logic
            if trigger_operator == ">=": trigger_indices = dataframe[dataframe[trigger_metric_name] >= trigger_threshold_value].index
            elif trigger_operator == "<=": trigger_indices = dataframe[dataframe[trigger_metric_name] <= trigger_threshold_value].index
            elif trigger_operator == ">": trigger_indices = dataframe[dataframe[trigger_metric_name] > trigger_threshold_value].index
            elif trigger_operator == "<": trigger_indices = dataframe[dataframe[trigger_metric_name] < trigger_threshold_value].index
            else: trigger_indices = dataframe[dataframe[trigger_metric_name] == trigger_threshold_value].index
            
            st.markdown(f"**Trigger Events Found:** {len(trigger_indices)}")
            
            if len(trigger_indices) > 0 and sea_response_variable in dataframe.columns:
                epoch_window = 10
                epoch_array = []
                for trigger_time in trigger_indices:
                    try:
                        position_index = dataframe.index.get_loc(trigger_time)
                        if position_index >= epoch_window and position_index < len(dataframe) - epoch_window:
                            epoch_array.append(dataframe[sea_response_variable].iloc[position_index-epoch_window:position_index+epoch_window+1].values)
                    except Exception:
                        continue
                
                if epoch_array:
                    epoch_mean_values = np.nanmean(epoch_array, axis=0)
                    epoch_day_range = np.arange(-epoch_window, epoch_window+1)
                    figure_superposed_epoch = px.line(x=epoch_day_range, y=epoch_mean_values, 
                                                     labels={"x": "Days from Trigger (Day 0)", "y": f"Mean {sea_response_variable}"}, 
                                                     title=f"Response Analysis: {sea_response_variable} aligned to {trigger_metric_name} {trigger_operator} {trigger_threshold_value}")
                    figure_superposed_epoch.add_vline(x=0, line_dash="dash", line_color="#f59e0b", annotation_text="Event Start")
                    figure_superposed_epoch.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0),
                                        xaxis=dict(title="Days from Day 0 (Trigger)"), yaxis=dict(title=f"Mean {sea_response_variable}"),
                                        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1))
                    st.plotly_chart(figure_superposed_epoch, width="stretch")
                else:
                    st.warning("Trigger events are too sparse or close to edges to generate a mean epoch.")
        else:
            st.warning(f"Metric {trigger_metric_name} not found in historical record.")

    with column_lag_analysis_viz:
        st.subheader("Event Transit Tracking: Flare to Storm")
        flare_storm_events = load_stats_file("flare_storm_lag_events.csv")
        if not flare_storm_events.empty and "dst_lag_days" in flare_storm_events.columns:
            triggered_storms = flare_storm_events[flare_storm_events.get("storm_triggered", False)]
            if not triggered_storms.empty:
                figure_transit_delay_histogram = px.histogram(triggered_storms, x="dst_lag_days", nbins=8, 
                                                             title=f"Transit Delay (n={len(triggered_storms)} storms)", 
                                                             labels={"dst_lag_days": "Lag (days)"}, color_discrete_sequence=["#22c55e"])
                mean_transit_lag = triggered_storms["dst_lag_days"].mean()
                figure_transit_delay_histogram.add_vline(x=mean_transit_lag, line_dash="dash", line_color="#ef4444", annotation_text=f"Mean: {mean_transit_lag:.1f}d")
                figure_transit_delay_histogram.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0),
                                     xaxis=dict(title="Transit Lag (days)"), yaxis=dict(title="Frequency (Count)"),
                                     legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1))
                st.plotly_chart(figure_transit_delay_histogram, width="stretch")
            else:
                st.warning("No triggered storms identified in the current window.")

            extreme_storms_dataframe = load_stats_file("extreme_storms_dst.csv")
            if not extreme_storms_dataframe.empty and set(["date", "dst_daily_min", "kp_daily_max"]).issubset(extreme_storms_dataframe.columns):
                st.subheader("Worst Global Disruptions (Dst ≤ -100)")
                st.dataframe(extreme_storms_dataframe[["date", "dst_daily_min", "kp_daily_max"]].head(10), hide_index=True, height=250)
        else:
            st.info("Additional storm-event data is missing from current cleaned records.")


# Helper function for aurora visibility locations
def get_aurora_visibility_locations(latitude):
    """Return major cities/regions visible at given latitude"""
    locations_north = {
        80: "Northern Canada, Greenland, Northern Scandinavia",
        70: "Alaska, Northern Canada, Northern Scandinavia",
        60: "Mid-Canada, Scotland, Southern Scandinavia",
        55: "Northern US (Minnesota, Wisconsin), Northern UK",
        50: "Northern US (Michigan, New York), Central Europe",
        45: "US Northern tier (Oregon, Montana, Vermont), France/Germany",
        40: "Northern US (New York, Pennsylvania), Southern Europe",
        35: "Mid US (Missouri, Ohio), Southern Mediterranean"
    }
    
    for lat_threshold in sorted(locations_north.keys(), reverse=True):
        if latitude >= lat_threshold:
            return locations_north[lat_threshold]
    return "Equatorial regions"

# ---------------------------------------------------------------------------
# PAGE 5: Geospatial Impact (Aurora Map)
# ---------------------------------------------------------------------------

def page_geospatial_impact():
    st.title("Geospatial Impact: Auroral Extent Mapping")
    st.markdown("Slide through time to observe how exact planetary disruption pushes auroral boundaries southwards.")
    
    st.info(
        "**Why does the mapped boundary expand?**\n"
        "During a quiet solar day (Kp 0-3), the Northern and Southern Lights are confined to the extreme poles (>65° Latitude). \n\n"
        "When a massive CME hits Earth (pushing Kp to 8 or 9), it drastically compresses the magnetosphere on the day-side and stretches the night-side tail, forcing the auroral oval dramatically down towards the equator. This interactive globe maps exactly how far south these dangerous atmospheric disruptions (and their accompanying satellite drag effects) penetrated on any historical date we possess."
    )

    dates_available = np.unique(dataframe.index.date)
    dates_available.sort()
    default_date_index = int(len(dates_available)/2)
    selected_date_value = st.select_slider("Drag to Shift Time", options=dates_available, value=dates_available[default_date_index])
    date_row_data = dataframe.loc[pd.Timestamp(selected_date_value)] if pd.Timestamp(selected_date_value) in dataframe.index else pd.Series()
    kp_index_value = float(date_row_data.get("kp_daily_max", 0)) if not date_row_data.empty else 0
    aurora_visibility_boundary = max(35, 68 - (3.6 * kp_index_value))

    column_metric1, column_metric2 = st.columns(2)
    with column_metric1:
        st.metric(label="Selected Baseline Kp", value=f"{kp_index_value:.2f}/9.00")
    with column_metric2:
        st.metric(label="Aurora Visible Threshold", value=f"±{aurora_visibility_boundary:.1f}° Lat")
    
    # Show visible locations
    visible_aurora_locations = get_aurora_visibility_locations(aurora_visibility_boundary)
    st.markdown(f"**🌍 Aurora Visible In:** {visible_aurora_locations}")

    # Optimize grid for performance - use coarser grid
    latitude_grid = np.arange(-90, 91, 6)
    longitude_grid = np.arange(-180, 181, 8)
    longitude_mesh, latitude_mesh = np.meshgrid(longitude_grid, latitude_grid)
    latitude_flat = latitude_mesh.flatten()
    longitude_flat = longitude_mesh.flatten()
    aurora_mask = (latitude_flat >= aurora_visibility_boundary) | (latitude_flat <= -aurora_visibility_boundary)
    aurora_map_latitudes = latitude_flat[aurora_mask]
    aurora_map_longitudes = longitude_flat[aurora_mask]

    if kp_index_value <= 4:
        aurora_color_hex, aurora_opacity_alpha = "#22c55e", 0.4
    elif kp_index_value <= 6:
        aurora_color_hex, aurora_opacity_alpha = "#fbbf24", 0.6
    elif kp_index_value <= 8:
        aurora_color_hex, aurora_opacity_alpha = "#ef4444", 0.7
    else:
        aurora_color_hex, aurora_opacity_alpha = "#dc2626", 0.8

    figure_aurora_geographic_map = go.Figure(go.Scattergeo(
        lat=aurora_map_latitudes, lon=aurora_map_longitudes, mode='markers',
        marker=dict(size=4, color=aurora_color_hex, opacity=aurora_opacity_alpha, symbol='circle', line=dict(width=0)),
        hoverinfo='none',
    ))
    figure_aurora_geographic_map.update_geos(
        projection_type="orthographic",
        showcoastlines=True, coastlinecolor="#374151",
        showland=True, landcolor="#1f2937",
        showocean=True, oceancolor="#0f172a",
    )
    figure_aurora_geographic_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure_aurora_geographic_map, width="stretch")


# ---------------------------------------------------------------------------
# PAGE 6: Physics Primer
# ---------------------------------------------------------------------------

def page_storm_simulator():
    st.title("NOAA G-Scale Storm Simulator")
    st.markdown("Interactive exploration of how arbitrary scale metrics map to planetary consequences.")
    
    st.markdown("### ⚡ Storm Severity Simulation")
    storm_cat = st.select_slider(
        "Slide to simulate storm intensity (Kp correlation):", 
        options=['Quiet (Kp 0-3)', 'Active (Kp 4)', 'G1 Minor (Kp 5)', 'G2 Moderate (Kp 6)', 'G3 Strong (Kp 7)', 'G4 Severe (Kp 8)', 'G5 Extreme (Kp 9)'],
        value='Quiet (Kp 0-3)'
    )

    column_physics_explanation, column_physics_simulator = st.columns([1, 1])
    with column_physics_explanation:
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

    with column_physics_simulator:
        kp_index_value_map = {'Quiet (Kp 0-3)': 2, 'Active (Kp 4)': 4, 'G1 Minor (Kp 5)': 5, 'G2 Moderate (Kp 6)': 6, 'G3 Strong (Kp 7)': 7, 'G4 Severe (Kp 8)': 8, 'G5 Extreme (Kp 9)': 9}
        gauge_indicator_value = kp_index_value_map[storm_cat]
        figure_kp_gauge_simulator = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = gauge_indicator_value, domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Kp Index Equivalent"},
            gauge = {
                'axis': {'range': [0, 9]},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [0, 4], 'color': "#10b981"},
                    {'range': [4, 6], 'color': "#f59e0b"},
                    {'range': [6, 8], 'color': "#ef4444"},
                    {'range': [8, 9], 'color': "#d946ef"}
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': gauge_indicator_value}
            }
        ))
        figure_kp_gauge_simulator.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure_kp_gauge_simulator, width="stretch")

def page_physics_primer():
    st.title("Physics Primer")
    st.markdown("Detailed exploration of solar drivers and terrestrial magnetic responses.")
    
    st.subheader("What is the Sunspot Number?")
    column_ssn_images, column_ssn_text = st.columns([1, 2])
    
    with column_ssn_images:
        import os
        if os.path.exists("Images_dashboard/sunspot_11_year_cycle.avif"):
            st.image("Images_dashboard/sunspot_11_year_cycle.avif", caption="Sunspot Number Time Series (1700-Present). Source: Historical Data.", width="stretch")
        if os.path.exists("Images_dashboard/sunspot_images.avif"):
            st.image("Images_dashboard/sunspot_images.avif", caption="Sunspot Observations and Classifications.", width="stretch")
            
    with column_ssn_text:
        st.markdown(
            "The abundance of sunspots on the Sun varies on timescales from a few hours to many years. "
            "Historically, an index called the 'sunspot number' has been used to quantify the abundance of spots. "
            "This index is still in wide use today, although for some purposes it has been replaced by more readily and "
            "consistently measured indices such as the 10.7 centimetre solar flux. The main advantage of the "
            "sunspot number is that it is the only index for which we have a long and detailed historical record."
        )
        st.markdown(
            "**Sunspot Number (denoted R) is defined as:**"
        )
        st.latex(r"R = K \cdot (10 \cdot G + I)")
        st.markdown(
            "where **G** is the number of sunspot groups visible on the Sun; **I** is the total number of individual spots visible; "
            "and **K** is an instrumental factor to take into account differences between observers and observatories."
        )
        st.markdown(
            "Sunspot Number as an index can be defined on a daily basis but because of the large day-to-day variation is usually "
            "averaged over longer periods, the most common being the monthly and the yearly average. When averaged over a year, "
            "the sunspot number varies smoothly charting the progress of the solar cycle. On the other hand the daily and the monthly "
            "averages exhibit considerable variation with respect to the yearly curve. This variation is due to bursts of rapid solar region "
            "growth often associated with solar flares and other interesting events."
        )
        st.markdown(
            "The most widely quoted average sunspot number is the Zurich number (Rz) which was replaced from January 1981 with the "
            "International Sunspot Number (RI). The American Sunspot Number is another series to which the ASWFC Culgoora Observatory "
            "contributed its observations."
        )
        st.caption("Text Source: [SWS BOM Educational](https://www.sws.bom.gov.au/Educational/2/3/3) | Image Source: [UCAR Center for Science Education](https://scied.ucar.edu/activity/sunspots-and-climate)")

    st.markdown("---")
    
    st.subheader("Other Critical Telemetry Data Points")
    st.markdown("While Sunspot Numbers track optical surface activity, we rely on three other primary measurements to quantify the true physical intensity of Space Weather events impacting Earth.")
    
    column_f107_telemetry, column_xray_flares_telemetry, column_dst_telemetry = st.columns(3)
    
    with column_f107_telemetry:
        st.markdown("#### F10.7 Solar Flux")
        if os.path.exists("Images_dashboard/f107_flux.png"):
            st.image("Images_dashboard/f107_flux.png", width="stretch")
        st.markdown(
            "The **F10.7 cm Radio Flux** is a measurement of the solar radio emission at a wavelength of 10.7 centimeters (2800 MHz). "
            "Unlike sunspots which require clear optical viewing, radio emissions can be reliably measured in any terrestrial weather condition."
        )
        st.markdown(
            "It is highly correlated with the sunspot number but serves as a much better proxy for the **Extreme Ultraviolet (EUV)** output of the Sun, "
            "which directly controls the heating and expansion of the Earth's upper atmosphere (the thermosphere). Higher flux values mean increased satellite drag."
        )
        
    with column_xray_flares_telemetry:
        st.markdown("#### The Kp Index")
        if os.path.exists("Images_dashboard/kp_index.png"):
            st.image("Images_dashboard/kp_index.png", width="stretch")
        st.markdown(
            "The **Planetary K-index (Kp)** is the most widely recognized indicator of global geomagnetic storm magnitude. "
            "It quantifies disturbances in the Earth's magnetic field with a 3-hour resolution."
        )
        st.markdown(
            "It is a quasi-logarithmic integer scale ranging from **0 (quiet)** to **9 (extreme storm)**. "
            "It is derived from the maximum fluctuations of horizontal magnetic components measured by a global network of ground-based magnetometers. "
            "The NOAA G-Scale (G1 to G5) maps directly mathematically to Kp values of 5 through 9."
        )
        
    with column_dst_telemetry:
        st.markdown("#### The Dst Index")
        if os.path.exists("Images_dashboard/dst_ring_current.png"):
            st.image("Images_dashboard/dst_ring_current.png", width="stretch")
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
# PAGE 2: Solar Number Time Series
# ---------------------------------------------------------------------------

def page_solar_number_timeseries():
    st.title("Solar Number Time Series & Historical Analysis")
    st.markdown("Explore historical sunspot numbers, cyclical patterns, and major solar events.")
    
    st.info(
        "**What you'll see:**\n"
        "- **Two centuries of sunspot data** from historical records (1818-present)\n"
        "- **11-year solar cycles** clearly visible in the long-term trends\n"
        "- **Monthly heatmap** revealing seasonal patterns and peak activity periods\n"
        "- **Major solar events** marked on the timeline for context"
    )
    
    # Load sunspot data
    sunspot_data = load_sunspots_data()
    
    if sunspot_data.empty:
        st.error("Could not load sunspot data.")
        return
    
    # Filter out rows with no sunspot number data
    sunspot_data_clean = sunspot_data[sunspot_data["sn"].notna()].copy()
    
    st.subheader("Historical Sunspot Number (1818-Present)")
    
    # Create time series plot with 11-year cycle overlay
    figure_sunspot_timeseries = go.Figure()
    
    # Main time series
    figure_sunspot_timeseries.add_trace(go.Scattergl(
        x=sunspot_data_clean.index,
        y=sunspot_data_clean["sn"],
        mode="lines",
        line=dict(color="#f59e0b", width=1),
        opacity=0.6,
        name="Daily Sunspot Number"
    ))
    
    # 365-day rolling mean to highlight cycles
    rolling_mean_365 = sunspot_data_clean["sn"].rolling(365, center=True, min_periods=100).mean()
    figure_sunspot_timeseries.add_trace(go.Scattergl(
        x=rolling_mean_365.index,
        y=rolling_mean_365,
        mode="lines",
        line=dict(color="#d946ef", width=3),
        name="365-Day Rolling Mean (Cycle Trend)"
    ))
    
    figure_sunspot_timeseries.update_layout(
        height=450,
        hovermode="x unified",
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis=dict(title="Sunspot Number"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(figure_sunspot_timeseries, width="stretch")
    
    # Monthly Heatmap - Year vs Month
    st.subheader("Monthly Mean Sunspot Activity Heatmap")
    
    # Create monthly aggregates
    sunspot_data_clean["year"] = sunspot_data_clean.index.year
    sunspot_data_clean["month"] = sunspot_data_clean.index.month
    
    monthly_pivot = sunspot_data_clean.groupby(["year", "month"])["sn"].mean().reset_index()
    pivot_table = monthly_pivot.pivot(index="year", columns="month", values="sn")
    
    figure_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        y=pivot_table.index,
        colorscale="YlOrRd",
        colorbar=dict(title="Sunspot Number"),
        hoverongaps=False
    ))
    
    figure_heatmap.update_layout(
        height=600,
        xaxis=dict(title="Month"),
        yaxis=dict(title="Year"),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(figure_heatmap, width="stretch")
    
    # Major Solar Events Context
    st.subheader("Notable Historical Solar Events & Cycles")
    
    column_events_info, column_events_list = st.columns([1, 1])
    
    with column_events_info:
        st.markdown("""
        **Key Historical Solar Events:**
        
        - **Dalton Minimum (1790-1830):** Extended period of low sunspot activity, coinciding with the 'Year Without a Summer' (1816)
        - **Carrington Event (1859):** Most powerful geomagnetic storm ever recorded. If it occurred today, would cause trillions in damage
        - **Halloween Events (2003):** Series of powerful X-class solar flares and CMEs
        - **Maunder Minimum Discussion:** Some historical periods show near-zero sunspot activity
        
        **Current Cycle:** Solar Cycle 25 (began 2019, maximum predicted 2024-2025)
        """)
    
    with column_events_list:
        major_events = pd.DataFrame({
            "Event": ["Dalton Minimum", "Carrington Event", "Halloween Storms", "Solar Max (Cycle 24)", "Solar Min (2008)", "Solar Max (Cycle 25)"],
            "Year": [1810, 1859, 2003, 2014, 2008, 2024],
            "Type": ["Extended Minimum", "Extreme Flare/CME", "X-Class Flares", "Maximum", "Minimum", "Maximum"],
            "Impact": ["Low activity", "Extreme", "Severe", "Moderate", "Minimal", "High"]
        })
        st.dataframe(major_events, width="stretch")
    
    st.markdown("---")
    
    # Statistics
    st.subheader("Sunspot Data Statistics")
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("Records", f"{len(sunspot_data_clean):,}")
    
    with col_stats2:
        st.metric("Mean SSN", f"{sunspot_data_clean['sn'].mean():.1f}")
    
    with col_stats3:
        st.metric("Max SSN", f"{sunspot_data_clean['sn'].max():.0f}")
    
    with col_stats4:
        st.metric("Date Range", f"{sunspot_data_clean.index.min().year}-{sunspot_data_clean.index.max().year}")



# ---------------------------------------------------------------------------
# PAGE 5.5: Extreme Events Analysis
# ---------------------------------------------------------------------------

def page_extreme_events():
    st.title("Extreme Events Detection & Analysis")
    st.markdown("Identify and analyze outlier days with abnormal geomagnetic and solar activity.")
    
    st.info(
        "**What constitutes an extreme event?**\n"
        "Using statistical z-scores (standard deviations from the mean), we identify days with unusual activity levels. "
        "Events beyond 2.5σ are considered extreme. This helps discover:\n"
        "- **Geomagnetic Storms:** Dst plummeting to record lows\n"
        "- **Solar Eruptions:** Massive X-ray flares or unusually high sunspot activity\n"
        "- **Solar System Disruptions:** Extreme F10.7 radio flux spikes"
    )
    
    column_extreme_event_config, column_extreme_event_viz = st.columns([1, 2])
    
    with column_extreme_event_config:
        st.subheader("Threshold Configuration")
        zscore_threshold_sigma = st.slider("Z-Score Threshold (σ)", 1.5, 4.0, 2.5, 0.1)
        extreme_event_variable = st.selectbox("Variable", ["kp_daily_max", "ssn", "f107", "flare_xray_total"])
    
    with column_extreme_event_viz:
        if extreme_event_variable in dataframe_filtered.columns:
            extreme_events_dataframe = find_extreme_events(dataframe_filtered, extreme_event_variable, zscore_threshold_sigma)
            
            if len(extreme_events_dataframe) > 0:
                figure_extreme_events_visualization = go.Figure()
                figure_extreme_events_visualization.add_trace(go.Scattergl(
                    x=dataframe_filtered.index, y=dataframe_filtered[extreme_event_variable],
                    mode='lines', line=dict(color='#e5e7eb', width=1),
                    name="Normal Days", opacity=0.5
                ))
                figure_extreme_events_visualization.add_trace(go.Scattergl(
                    x=extreme_events_dataframe.index, y=extreme_events_dataframe[extreme_event_variable],
                    mode='markers', marker=dict(size=8, color=extreme_events_dataframe["z_score"], 
                    colorscale="RdYlGn_r", showscale=True, colorbar=dict(title="Z-Score")),
                    name="Extreme Events"
                ))
                figure_extreme_events_visualization.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                                    xaxis=dict(title="Date"), yaxis=dict(title=f"{extreme_event_variable}"),
                                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1))
                st.plotly_chart(figure_extreme_events_visualization, width="stretch")
                
                st.subheader("Top Extreme Events")
                st.dataframe(extreme_events_dataframe[[extreme_event_variable, "z_score"]].head(10), height=250)
            else:
                st.warning("No extreme events found with current threshold.")
        else:
            st.error(f"Column {extreme_event_variable} not found in data.")


# ---------------------------------------------------------------------------
# PAGE 6: Periodicity & Cycle Analysis
# ---------------------------------------------------------------------------

def page_periodicity():
    st.title("Periodicity & Solar Cycle Analysis")
    st.markdown("Discover dominant cycles and periodic patterns in solar/geomagnetic activity.")
    
    st.info(
        "**FFT (Fast Fourier Transform):** Global frequency analysis to identify dominant cycles across the entire dataset.\n"
        "**Wavelet (CWT):** Time-localized spectral power showing how cycles evolve through time.\n"
        "The 11-year solar cycle should dominate. Secondary peaks may reveal the 27-day solar rotation (~monthly signature) or other harmonic patterns."
    )
    
    column_periodicity_config, column_periodicity_viz = st.columns([1, 2])
    
    with column_periodicity_config:
        st.subheader("Analysis Configuration")
        periodicity_analysis_variable = st.selectbox("Variable", ["ssn", "f107", "kp_daily_max", "dst_daily_min"], index=0)
        periodicity_analysis_method = st.radio("Method", ["FFT (Global)", "Wavelet (Temporal)"])
    
    with column_periodicity_viz:
        if periodicity_analysis_variable in dataframe_filtered.columns:
            try:
                periodicity_analysis_results = analyze_periodicity(dataframe_filtered[periodicity_analysis_variable].dropna(), 
                                                         method='fft' if 'FFT' in periodicity_analysis_method else 'wv')
                
                if 'FFT' in periodicity_analysis_method:
                    st.subheader("Dominant Periods (FFT)")
                    top_periodic_periods = periodicity_analysis_results.nlargest(10, "amplitude")
                    figure_periodicity_fft_chart = px.bar(top_periodic_periods.head(10), x="period_years", y="amplitude",
                                    labels={"period_years": "Period (Years)", "amplitude": "Amplitude"},
                                    title=f"Top 10 Periods in {periodicity_analysis_variable}")
                    figure_periodicity_fft_chart.update_layout(height=400, showlegend=False,
                                        xaxis=dict(title="Period (Years)"), yaxis=dict(title="Amplitude (Power)"))
                    st.plotly_chart(figure_periodicity_fft_chart, width="stretch")
                    
                    st.dataframe(top_periodic_periods[["period_years", "amplitude"]].head(10), hide_index=True, height=250)
                else:
                    st.subheader("Temporal Periodicity (Wavelet)")
                    st.line_chart(periodicity_analysis_results["dominant_period_yrs"], height=400)
                    
            except Exception as analysis_error:
                st.error(f"Periodicity analysis failed: {analysis_error}")
        else:
            st.error(f"Column {periodicity_analysis_variable} not found.")


# ---------------------------------------------------------------------------
# PAGE 7: Phase-Locked Climatology
# ---------------------------------------------------------------------------

def page_phase_climatology():
    st.title("Phase-Locked Climatology: Solar Cycle Risk Zones")
    st.markdown("Identify 'danger zones' within the 11-year solar cycle where geomagnetic storms are most likely.")
    
    st.info(
        "**What is phase-locking?**\n"
        "By dividing the 11-year solar cycle into 20 phase bins (5% intervals), we bin all Kp observations by their cycle phase and compute statistics. "
        "This reveals if certain phases of the cycle are inherently 'stormier' than others, independent of absolute solar activity magnitude."
    )
    
    column_climate_config, column_climate_viz = st.columns([1, 2])
    
    with column_climate_config:
        st.subheader("Configuration")
        phase_bin_count = st.slider("Phase Bins", 10, 40, 20, 2)
        kp_column_selection = st.selectbox("Kp Column", [column for column in dataframe_filtered.columns if 'kp' in column.lower()], index=0)
        storm_threshold_phase_locked = st.number_input("Storm Threshold (Kp)", 0.0, 9.0, 6.0, 0.5)
    
    with column_climate_viz:
        if kp_column_selection in dataframe_filtered.columns:
            try:
                phase_climatology_data = analyze_phase_locked_climatology(
                    dataframe_filtered, kp_col=kp_column_selection, storm_threshold=storm_threshold_phase_locked, bins=phase_bin_count
                )
                
                figure_phase_climatology_chart = go.Figure()
                figure_phase_climatology_chart.add_trace(go.Scattergl(
                    x=phase_climatology_data.index.astype(float), y=phase_climatology_data["kp_mean"],
                    mode='lines+markers', line=dict(color='#22c55e', width=2),
                    marker=dict(size=6), name="Mean Kp", fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.2)'
                ))
                figure_phase_climatology_chart.add_trace(go.Scattergl(
                    x=phase_climatology_data.index.astype(float), y=phase_climatology_data["storm_prob_pct"],
                    mode='lines+markers', line=dict(color='#ef4444', width=2),
                    marker=dict(size=6), name="Storm Prob (%)", yaxis='y2'
                ))
                figure_phase_climatology_chart.update_layout(
                    height=400, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(title="Solar Cycle Phase (0=min, 1=max)"),
                    yaxis=dict(title="Mean Kp (0-9 scale)"),
                    yaxis2=dict(title="Storm Probability (%)", overlaying='y', side='right'),
                    hovermode="x unified",
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1)
                )
                st.plotly_chart(figure_phase_climatology_chart, width="stretch")
                
                st.subheader("Climatology Statistics Table")
                st.dataframe(phase_climatology_data.astype(float).round(3), height=400)
            except Exception as climatology_error:
                st.error(f"Phase climatology analysis failed: {climatology_error}")
        else:
            st.error(f"Column {kp_column_selection} not found.")


# ---------------------------------------------------------------------------
# PAGE 8: Hysteresis Analysis
# ---------------------------------------------------------------------------

def page_hysteresis():
    st.title("Hysteresis Analysis: Solar Cycle Phase Effects")
    st.markdown("Compare geomagnetic response (Kp) to sunspot activity across rising vs. falling cycle phases.")
    
    st.info(
        "**What is hysteresis?**\n"
        "Hysteresis occurs when a system's output depends not just on current input, but also on its history. "
        "In solar physics, this reveals whether the magnetosphere responds differently to the same sunspot number depending on whether the cycle is rising (building) or falling (declining). "
        "If points scatter differently by color, it proves the system has 'memory' of where we are in the 11-year cycle."
    )
    
    column_hysteresis_config, column_hysteresis_viz = st.columns([1, 2])
    
    with column_hysteresis_config:
        st.subheader("Configuration")
        sunspot_column_hysteresis = st.selectbox("Sunspot Proxy", [column for column in dataframe_filtered.columns if 'ssn' in column.lower() or 'sn' in column.lower()], index=0)
        kp_column_hysteresis = st.selectbox("Geomagnetic Index", [column for column in dataframe_filtered.columns if 'kp' in column.lower()], index=0)
    
    with column_hysteresis_viz:
        if sunspot_column_hysteresis in dataframe_filtered.columns and kp_column_hysteresis in dataframe_filtered.columns:
            try:
                hysteresis_analysis_dataframe = analyze_hysteresis(
                    dataframe_filtered, sn_col=sunspot_column_hysteresis, kp_col=kp_column_hysteresis, save_path="Images_dashboard/hysteresis_phase.png"
                )
                
                rising_phase_data = hysteresis_analysis_dataframe[hysteresis_analysis_dataframe['phase_type'] == 'Rising']
                falling_phase_data = hysteresis_analysis_dataframe[hysteresis_analysis_dataframe['phase_type'] == 'Falling']
                
                figure_hysteresis_scatter = go.Figure()
                figure_hysteresis_scatter.add_trace(go.Scattergl(
                    x=rising_phase_data[sunspot_column_hysteresis], y=rising_phase_data[kp_column_hysteresis],
                    mode='markers', marker=dict(size=5, color='#22c55e', opacity=0.4),
                    name='Rising Phase'
                ))
                figure_hysteresis_scatter.add_trace(go.Scattergl(
                    x=falling_phase_data[sunspot_column_hysteresis], y=falling_phase_data[kp_column_hysteresis],
                    mode='markers', marker=dict(size=5, color='#ef4444', opacity=0.4),
                    name='Falling Phase'
                ))
                figure_hysteresis_scatter.update_layout(
                    height=500, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(title=f"{sunspot_column_hysteresis} (Solar Activity)"),
                    yaxis=dict(title=f"{kp_column_hysteresis} (0-9 scale)"),
                    hovermode="x unified",
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1)
                )
                st.plotly_chart(figure_hysteresis_scatter, width="stretch")
                
                st.info(f"**Rising Phase:** {len(rising_phase_data)} days | **Falling Phase:** {len(falling_phase_data)} days")
                

            except Exception as hysteresis_error:
                st.error(f"Hysteresis analysis failed: {hysteresis_error}")
        else:
            st.error("Required columns not found.")


# ---------------------------------------------------------------------------
# PAGE 9: Monthly Statistics & Aggregation
# ---------------------------------------------------------------------------

def page_monthly_stats():
    column_monthly_config, column_monthly_viz = st.columns([1, 2])
    
    with column_monthly_config:
        st.subheader("Aggregation Settings")
        aggregation_period_selection = st.radio("Period", ["Monthly", "Quarterly", "Yearly"], horizontal=False)
        period_selection_map = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
        period_aggregation_code = period_selection_map[aggregation_period_selection]

    st.title(f"{aggregation_period_selection} Statistics & Temporal Aggregation")
    st.markdown(f"Roll up daily observations into {aggregation_period_selection.lower()} summaries to identify medium-term patterns.")
    
    st.info(
        f"**Why aggregate to {aggregation_period_selection.lower()}?**\n"
        "Daily data is noisy and influenced by local oscillations. Period aggregation reveals true trends by:\n"
        "- Computing **mean** values (smooth trends)\n"
        "- Tracking **max** values (peak events)\n"
        "- Summing **storm hours** (cumulative impact)"
    )
    
    with column_monthly_viz:
        try:
            if "ssn" in dataframe_filtered.columns:
                # Use the dynamic period code instead of hardcoded monthly stats
                periodic_aggregated_statistics = compute_periodly_stats(dataframe_filtered.copy(), period_aggregation_code)
                
                figure_periodic_statistics_chart = go.Figure()
                if "ssn_mean" in periodic_aggregated_statistics.columns:
                    figure_periodic_statistics_chart.add_trace(go.Scattergl(
                        x=periodic_aggregated_statistics.index, y=periodic_aggregated_statistics["ssn_mean"],
                        mode='lines', line=dict(color='#f59e0b', width=2),
                        name="SSN Mean"
                    ))
                if "kp_daily_mean" in periodic_aggregated_statistics.columns:
                    figure_periodic_statistics_chart.add_trace(go.Scattergl(
                        x=periodic_aggregated_statistics.index, y=periodic_aggregated_statistics["kp_daily_mean"],
                        mode='lines', line=dict(color='#22c55e', width=2),
                        name="Kp Mean", yaxis='y2'
                    ))
                
                figure_periodic_statistics_chart.update_layout(
                    height=400, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="SSN (Sunspot Number)"),
                    yaxis2=dict(title="Kp Index (0-9 scale)", overlaying='y', side='right'),
                    hovermode="x unified",
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1)
                )
                st.plotly_chart(figure_periodic_statistics_chart, width="stretch")
                
                st.subheader(f"{aggregation_period_selection} Statistics Summary")
                st.dataframe(periodic_aggregated_statistics.round(3).head(20), height=400)
            else:
                st.error("SSN column ('ssn') not found in dataset.")
        except Exception as monthly_stats_error:
            st.error(f"Monthly statistics failed: {monthly_stats_error}")



# ---------------------------------------------------------------------------
# PAGE 7: Data Sources
# ---------------------------------------------------------------------------

def page_data_sources():
    st.title("Data Sources & Provenance")
    st.markdown("This dashboard relies on high-fidelity telemetry from authoritative space weather institutions.")
    
    column_sources_left, column_sources_right = st.columns(2)
    with column_sources_left:
        st.subheader("SILSO")
        st.markdown("**Sunspot Index and Long-term Solar Observations (SILSO)**, World Data Center, Royal Observatory of Belgium, Brussels. We utilize the Version 2.0 total daily sunspot number.")
        st.write("[Visit SILSO](https://www.sidc.be/SILSO/home)")
        
        st.subheader("NASA OMNIWeb")
        st.markdown("**NASA Goddard Space Flight Center, Space Physics Data Facility (SPDF)**. We extract low-resolution (hourly) OMNI2 parameters, heavily focusing on the **Dst** (Disturbance Storm Time) index.")
        st.write("[Visit OMNIWeb](https://omniweb.gsfc.nasa.gov/)")
    
    with column_sources_right:
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

# ---------------------------------------------------------------------------
# Router Execution
# ---------------------------------------------------------------------------

if page == "Solar Number Time Series":
    page_solar_number_timeseries()
elif page == "System Overview":
    page_overview()
elif page == "Physics Primer":
    page_physics_primer()
elif page == "Storm Simulator":
    page_storm_simulator()
elif page == "Geospatial Impact":
    page_geospatial_impact()
elif page == "Monthly Statistics":
    page_monthly_stats()
elif page == "Data Smoothing":
    page_smoothing()
elif page == "Correlation Matrix":
    page_correlations()
elif page == "Lag-Time Analysis":
    page_lag_analysis()
elif page == "Periodicity Analysis":
    page_periodicity()
elif page == "Phase-Locked Climatology":
    page_phase_climatology()
elif page == "Hysteresis Analysis":
    page_hysteresis()
elif page == "Extreme Events":
    page_extreme_events()
elif page == "Data Sources":
    page_data_sources()
elif page == "Project Details":
    page_project_details()
