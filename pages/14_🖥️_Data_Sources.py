"""pages/14_🖥️_Data_Sources.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.refresh import auto_refresh_check, render_refresh_sidebar
from utils.theme import apply_theme

st.set_page_config(page_title="Data Sources", page_icon="🖥️", layout="wide")
apply_theme()
auto_refresh_check()
render_refresh_sidebar()

st.title("🖥️ Data Sources & Provenance")

c1, c2 = st.columns(2)
with c1:
    st.subheader("SILSO")
    st.markdown(
        "**Sunspot Index and Long-term Solar Observations**, Royal Observatory of Belgium. "
        "Version 2.0 total daily sunspot number."
    )
    st.write("[Visit SILSO →](https://www.sidc.be/SILSO/home)")

    st.subheader("NASA OMNIWeb / SPDF")
    st.markdown(
        "**NASA Goddard SPDF**. OMNI2 hourly data used for Dst index (1986–2004). "
        "Low-resolution (~1.5 MB/year) bulk files."
    )
    st.write("[Visit OMNIWeb →](https://omniweb.gsfc.nasa.gov/)")

with c2:
    st.subheader("NOAA NCEI & SWPC")
    st.markdown(
        "**National Centers for Environmental Information** & **Space Weather Prediction Center**. "
        "Daily Solar Data (DSD) for F10.7, flare counts (1986–present); "
        "live JSON feeds for real-time supplement."
    )
    st.write("[Visit SWPC →](https://www.swpc.noaa.gov/)")

    st.subheader("WDC Kyoto")
    st.markdown(
        "**World Data Center for Geomagnetism, Kyoto**. "
        "Definitive and provisional Dst indices (2005–present)."
    )
    st.write("[Visit WDC Kyoto →](https://wdc.kugi.kyoto-u.ac.jp/)")

    st.subheader("GFZ Potsdam")
    st.markdown(
        "**German Research Centre for Geosciences**. "
        "Kp/ap/Ap 3-hourly index via JSON API (1932–present)."
    )
    st.write("[Visit GFZ KP →](https://www.gfz-potsdam.de/en/kp-index)")
