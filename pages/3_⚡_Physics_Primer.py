"""pages/3_⚡_Physics_Primer.py"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.refresh import auto_refresh_check, render_refresh_sidebar
from utils.theme import apply_theme

st.set_page_config(page_title="Physics Primer", page_icon="⚡", layout="wide")
apply_theme()
auto_refresh_check()
render_refresh_sidebar()

st.title("⚡ Physics Primer")
st.subheader("What is the Sunspot Number?")

c1, c2 = st.columns([1, 2])
with c1:
    for img in ["Images_dashboard/sunspot_11_year_cycle.avif", "Images_dashboard/sunspot_images.avif"]:
        if os.path.exists(img):
            st.image(img, use_container_width=True)

with c2:
    st.markdown(
        "The sunspot number quantifies surface magnetic activity. It is the only index "
        "with a detailed historical record stretching back centuries."
    )
    st.latex(r"R = K \cdot (10 \cdot G + I)")
    st.markdown(
        "**G** = sunspot group count  |  **I** = individual spots  |  **K** = observatory correction factor.  \n"
        "Daily values are noisy due to the sun's 27-day rotation; monthly/yearly averages reveal the 11-year cycle."
    )
    st.caption("Source: [SWS BOM Educational](https://www.sws.bom.gov.au/Educational/2/3/3)")

st.markdown("---")
st.subheader("Other Key Telemetry")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### F10.7 Solar Flux")
    if os.path.exists("Images_dashboard/f107_flux.png"):
        st.image("Images_dashboard/f107_flux.png", use_container_width=True)
    st.markdown(
        "Measured at 10.7 cm (2800 MHz). Correlates tightly with SSN but better proxies "
        "**EUV output** that heats the thermosphere and increases satellite drag."
    )

with c2:
    st.markdown("#### Kp Index")
    if os.path.exists("Images_dashboard/kp_index.png"):
        st.image("Images_dashboard/kp_index.png", use_container_width=True)
    st.markdown(
        "Quasi-logarithmic 0–9 scale derived from ground magnetometers. "
        "NOAA G-scale (G1–G5) maps directly to Kp 5–9."
    )

with c3:
    st.markdown("#### Dst Index")
    if os.path.exists("Images_dashboard/dst_ring_current.png"):
        st.image("Images_dashboard/dst_ring_current.png", use_container_width=True)
    st.markdown(
        "Raw nanotesla measurement of equatorial ring current. "
        "Quiet day ≈ 0 nT; severe superstorm < −200 nT (e.g. Halloween 2003: −383 nT)."
    )
