"""pages/4_🌡️_Storm_Simulator.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import streamlit as st
from utils.refresh import auto_refresh_check, render_refresh_sidebar
from utils.theme import apply_theme, layout

st.set_page_config(page_title="Storm Simulator", page_icon="🌡️", layout="wide")
apply_theme()
auto_refresh_check()
render_refresh_sidebar()

st.title("🌡️ NOAA G-Scale Storm Simulator")
st.markdown("Slide through storm categories to see real-world consequences.")

LEVELS = ["Quiet (Kp 0–3)", "Active (Kp 4)", "G1 Minor (Kp 5)",
          "G2 Moderate (Kp 6)", "G3 Strong (Kp 7)", "G4 Severe (Kp 8)", "G5 Extreme (Kp 9)"]
KP_MAP  = dict(zip(LEVELS, [2, 4, 5, 6, 7, 8, 9]))

storm = st.select_slider("Storm intensity", options=LEVELS, value="Quiet (Kp 0–3)")
kp_val = KP_MAP[storm]

c1, c2 = st.columns(2)

with c1:
    if kp_val <= 4:
        st.success("**Aurora:** Confined to extreme northern latitudes (Alaska, Scandinavia).")
        st.markdown("**Infra:** Nominal power grid and spacecraft operations.")
        st.markdown("**Cause:** Background solar wind; no large CMEs.")
    elif kp_val <= 6:
        st.warning("**Aurora:** Visible down to mid-high latitudes (Michigan, Scotland).")
        st.markdown("**Infra:** Weak grid fluctuations. Increased drag on high-latitude orbits.")
        st.markdown("**Cause:** Glancing CME from M-class flare or coronal-hole stream.")
    elif kp_val <= 8:
        st.error("**Aurora:** Pushed to mid-latitudes (Oregon, Illinois).")
        st.markdown("**Infra:** Voltage corrections needed. GPS degradation. HF radio blackouts.")
        st.markdown("**Cause:** Direct Earth-directed CME from X-class flare.")
    else:
        st.error("🚨 **EXTREME — G5**")
        st.markdown("**Aurora:** Visible at low latitudes (Florida, Spain).")
        st.markdown("**Infra:** Widespread grid problems; complete HF blackout; massive orbit corrections.")
        st.markdown("**Historic analog:** Halloween Storms 2003 (Dst −383 nT).")

with c2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=kp_val,
        title={"text": "Kp Equivalent"},
        gauge={
            "axis": {"range": [0, 9]},
            "bar": {"color": "rgba(0,0,0,0)"},
            "steps": [
                {"range": [0, 4], "color": "#10b981"},
                {"range": [4, 6], "color": "#f59e0b"},
                {"range": [6, 8], "color": "#ef4444"},
                {"range": [8, 9], "color": "#d946ef"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": kp_val},
        },
    ))
    fig.update_layout(**layout(260))
    st.plotly_chart(fig, use_container_width=True)
