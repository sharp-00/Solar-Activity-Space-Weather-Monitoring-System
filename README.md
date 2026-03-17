# Solar Activity & Space Weather Monitoring System

A reproducible monitoring and analytics dashboard for solar activity and space weather indicators, built with Python and Plotly Dash.

## Project Structure

```
solar_monitor/
├── download_data.py     # Fetch and parse all raw data to data/
├── ingest.py            # Load from local data/ into DataFrames
├── pipeline.py          # Cleaning, resampling, harmonisation, smoothing
├── analysis.py          # Correlation, anomaly detection, cycle phase
├── app.py               # Plotly Dash dashboard (main entry point)
├── requirements.txt
└── data/                # Populated by download_data.py
    ├── silso_sunspots_daily.csv       # raw SILSO daily (downloaded once)
    ├── silso_sunspots_monthly.csv     # raw SILSO monthly (downloaded once)
    ├── kp_historic_gfz.txt            # raw GFZ Kp 1932-present (downloaded once)
    ├── noaa_kp_index.json             # raw NOAA recent feed (refreshed each run)
    ├── sunspots_daily_clean.csv       # parsed + cleaned daily SN
    ├── sunspots_monthly_clean.csv     # parsed + cleaned monthly SN
    └── kp_merged_clean.csv            # GFZ historic + NOAA recent, merged
```

## Setup & Run

### 1. Create a virtual environment and install dependencies

On **Arch-based systems** (EndeavourOS, Manjaro, etc.) pip is blocked system-wide.
Use a virtual environment instead — this is the recommended approach on any system:

```bash
# Create the venv inside the project folder
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies into the venv
pip install -r requirements.txt
```

> You only need to create and install once. On future sessions just run
> `source venv/bin/activate` before starting the app.

### 2. First-time data download

```bash
python3 download_data.py
```

This fetches:
- **SILSO daily sunspot numbers** (Royal Observatory of Belgium) — 1818 to present
- **SILSO monthly sunspot numbers** — same source, monthly resolution, back to 1749
- **GFZ Kp archive** (German Research Centre for Geosciences) — 3-hourly Kp index, 1932 to present
- **NOAA real-time Kp** (NOAA SWPC) — last ~7 days, merged on top of GFZ

Large historical files are downloaded **once** and never re-fetched unless you delete them.

### 3. Daily refresh (fast)

```bash
python5 download_data.py --refresh
```

Only re-fetches the last ~30 days of sunspot data and the latest NOAA Kp feed.
The large historical files are untouched. Run this whenever you want up-to-date data.

### 4. Launch the dashboard

```bash
python3 app.py
```

Open your browser at [http://localhost:8050](http://localhost:8050).

---

## Dashboard Overview

The dashboard has four tabs, all filtered by a shared date range picker:

| Tab | Content |
|-----|---------|
| **Time Series** | Dual-axis SN + Kp chart, storm markers, Kp=5 threshold line, monthly SN heatmap |
| **Correlation** | Cross-correlation bar chart with peak lag detection, lagged scatter with LOWESS trendline |
| **Extreme Events** | Sigma-threshold event tables for SN and Kp, annotated timeline, rolling z-score chart |
| **Smoothing** | Overlay of raw / 27-day rolling / 365-day rolling / Savitzky-Golay; residuals bar chart; data limitations table |

Global controls:
- **Date range picker** — filters all tabs simultaneously
- **Sigma slider** (1.5–4.0 σ) — threshold for extreme event detection (Tab 3)
- **Max lag slider** (5–60 days) — range for cross-correlation (Tab 2)

---

## Data Sources

| Dataset | Source | Coverage | Notes |
|---------|--------|----------|-------|
| Daily Sunspot Number | [SILSO WDC](https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv) | 1818–present | International SN v2, downloaded once |
| Monthly Sunspot Number | [SILSO WDC](https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv) | 1749–present | Monthly mean SN, downloaded once |
| Kp Index (historic) | [GFZ Potsdam](https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt) | 1932–present | 3-hourly, downloaded once |
| Kp Index (recent) | [NOAA SWPC](https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json) | Last ~7 days | Re-fetched on every refresh |

### Why two Kp sources?

The NOAA real-time JSON only provides the most recent ~7 days. For long-term analysis the GFZ (German Research Centre for Geosciences) maintains a definitive archive of all Kp values since 1932 in a single text file. The pipeline downloads the GFZ file once, then on each `--refresh` run overlays the latest NOAA readings on top. This gives a continuous 3-hourly Kp series from 1932 to today without re-downloading 90 years of data every time.

---

## Data Flow

```
download_data.py
    |
    |-- SILSO daily/monthly  →  sunspots_daily_clean.csv
    |                            sunspots_monthly_clean.csv
    |
    |-- GFZ Kp (once)  ─┐
    |-- NOAA Kp (live) ─┴─→  kp_merged_clean.csv
    
ingest.py           fetch_sunspots() / fetch_kp_index()
    ↓
pipeline.py         clean → resample → align → smooth → z-score
    ↓
analysis.py         cross_correlation / find_extreme_events / monthly_stats
    ↓
app.py              Plotly Dash dashboard
```

---

## Key Design Decisions

**Why daily resolution instead of monthly?**
The CME travel-time lag between a solar eruption and the resulting geomagnetic storm is 1–4 days. This structure is completely invisible at monthly resolution and only becomes visible in the cross-correlation at daily granularity.

**Why GFZ for historic Kp?**
The NOAA real-time API is not designed for bulk historical download — it only serves recent data. GFZ provides the definitive long-term archive maintained by the same scientific community that defined the Kp index, updated daily.

**Why Savitzky-Golay for the cycle baseline?**
A centred rolling mean creates a phase shift artefact near the edges of the data window. Savitzky-Golay (polynomial fitting in a sliding window) avoids this while preserving the shape of solar cycle peaks better than a simple moving average.
