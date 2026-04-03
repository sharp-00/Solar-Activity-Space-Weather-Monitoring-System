"""pages/15_ℹ️_Project_Details.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.refresh import auto_refresh_check, render_refresh_sidebar
from utils.theme import apply_theme

st.set_page_config(page_title="Project Details", page_icon="ℹ️", layout="wide")
apply_theme()
auto_refresh_check()
render_refresh_sidebar()

st.title("ℹ️ Project Details")

st.markdown("### The Mission")
st.markdown(
    "This project bridges raw astrophysical telemetry and actionable data science. "
    "An automated pipeline ingests decadal data from satellites and ground stations, "
    "cleans and harmonises multiple variables (SSN, F10.7, Flares, Kp, Dst), "
    "then powers custom analyses to illustrate how solar events drive geomagnetic instability on Earth."
)

st.markdown("---")
st.subheader("Architecture")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
**Pipeline:**
- `ingest.py` — downloads raw data from 6 sources
- `clean.py` — harmonises, imputes, exports **Parquet**
- `analysis.py` — statistical functions (FFT, wavelet, etc.)

**Data storage:**
- `data/raw/` — original downloaded files (gitignored)
- `data/clean/solar_weather_daily.parquet` — master merged dataset
- `data/clean/sunspots_daily_clean.parquet` — SSN only (SILSO)
- `data/clean/kp_daily_clean.parquet` — Kp/ap index (GFZ + DGD)
- `data/clean/dst_daily_clean.parquet` — Dst index (OMNI2 + Kyoto)
- `data/clean/f107_daily_clean.parquet` — F10.7 solar flux
- `data/clean/flares_daily_clean.parquet` — Solar flare counts
- `data/analysis/stats/` — pre-computed stat tables
""")
with c2:
    st.markdown("""
**Dashboard:**
- `app.py` — landing page + KPI cards
- `pages/` — 15 analysis pages (Streamlit multipage)
- `utils/` — shared loaders, refresh logic, theme

**Refresh:**
- Auto-fetch every **6 hours** via session state timer
- Manual **⚡ Fetch Latest** button in every sidebar
- Old CSV files auto-migrated to Parquet on first load
""")

st.markdown("---")
st.subheader("The Team")
st.markdown(
    "Built with ❤️ by:\n"
    "- **Mulumudi Dinesh Karthik**\n"
    "- **Vansh Gupta**\n"
    "- **Abhishek Menon**\n"
    "- **Shailendra Pratap Singh**"
)
