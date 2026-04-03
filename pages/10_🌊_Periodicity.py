"""pages/10_🌊_Periodicity.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.express as px
import streamlit as st

from utils.data_loader import load_main_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar, render_date_filter
from utils.theme import apply_theme, layout, axis
from analysis import analyze_periodicity

st.set_page_config(page_title="Periodicity", page_icon="🌊", layout="wide")
apply_theme(); auto_refresh_check(); render_refresh_sidebar()

st.title("🌊 Periodicity & Solar Cycle Analysis")
df = render_date_filter(load_main_data())
if df.empty:
    st.warning("No data."); st.stop()

st.info("**FFT** identifies dominant cycles globally.  \n"
        "**Wavelet** shows how cycles evolve over time (requires pywavelets).")

c_cfg, c_viz = st.columns([1, 2])
with c_cfg:
    candidates = [c for c in ["ssn","f107","kp_daily_max","dst_daily_min"]
                  if c in df.columns and df[c].notna().sum() > 100]
    if not candidates:
        st.info("No suitable columns found."); st.stop()
    var    = st.selectbox("Variable", candidates)
    method = st.radio("Method", ["FFT (Global)", "Wavelet (Temporal)"])

with c_viz:
    try:
        results = analyze_periodicity(df[var].dropna(),
                                      method="fft" if "FFT" in method else "wv")
        if "FFT" in method:
            top = results.nlargest(10, "amplitude")
            fig = px.bar(top, x="period_years", y="amplitude",
                         labels={"period_years":"Period (years)","amplitude":"Amplitude"},
                         title=f"Top 10 periods in {var}",
                         color_discrete_sequence=["#f59e0b"])
            fig.update_layout(**layout(400))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top[["period_years","amplitude"]].round(4),
                         hide_index=True, use_container_width=True)
        else:
            st.line_chart(results["dominant_period_yrs"], height=400)
    except ImportError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Periodicity error: {e}")
