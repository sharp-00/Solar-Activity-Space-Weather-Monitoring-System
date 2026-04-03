# ☀️ Space Weather Analytics

Real-time & historical solar / geomagnetic data intelligence dashboard.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first launch the dashboard auto-detects missing data and runs the pipeline.
Use the **⚡ Fetch Latest Data** button in the sidebar at any time.

---

## Structure

```
solar/
├── app.py                  # Home page — KPI cards + global date filter
├── pages/                  # 15 analysis pages (Streamlit auto-builds sidebar nav)
│   ├── 1_🔭_Solar_Timeseries.py
│   ├── 2_📊_System_Overview.py
│   ├── 3_⚡_Physics_Primer.py
│   ├── 4_🌡️_Storm_Simulator.py
│   ├── 5_🌍_Geospatial_Impact.py
│   ├── 6_📅_Monthly_Stats.py
│   ├── 7_📈_Data_Smoothing.py
│   ├── 8_🔗_Correlations.py
│   ├── 9_⏱️_Lag_Analysis.py
│   ├── 10_🌊_Periodicity.py
│   ├── 11_🔄_Phase_Climatology.py
│   ├── 12_↔️_Hysteresis.py
│   ├── 13_⚠️_Extreme_Events.py
│   ├── 14_🖥️_Data_Sources.py
│   └── 15_ℹ️_Project_Details.py
├── utils/
│   ├── data_loader.py      # Cached parquet loaders + auto-migration from old CSVs
│   ├── refresh.py          # 6h auto-refresh + manual button logic
│   └── theme.py            # Shared CSS + Plotly layout defaults
├── ingest.py               # Downloads raw data from 6 sources
├── clean.py                # Harmonises & exports → data/clean/*.parquet
├── analysis.py             # Statistical functions (FFT, wavelet, cross-corr…)
└── requirements.txt
```

---

## Data Pipeline

```
ingest.py  →  data/raw/           (raw downloaded files)
clean.py   →  data/clean/*.parquet (processed, typed, interpolated)
```

Run manually:
```bash
python ingest.py   # ~10–20 min (first time; Kyoto has per-request sleep)
python clean.py    # ~30 sec
```

---

## Data Refresh

| Trigger | Behaviour |
|---|---|
| App start (no data) | Pipeline runs automatically |
| `> 6 hours` since last fetch | Auto-refresh on next page load |
| **⚡ Fetch Latest** button | Manual, instant, visible on every page |

---

## Data Sources

| Source | Variables | URL |
|---|---|---|
| SILSO / Royal Observatory Brussels | Daily SSN | sidc.be |
| NOAA NCEI / SWPC | F10.7, Flares, Kp | swpc.noaa.gov |
| GFZ Potsdam | Kp/ap 3-hourly | kp.gfz.de |
| NASA SPDF OMNI2 | Dst 1986–2004 | spdf.gsfc.nasa.gov |
| WDC Kyoto | Dst 2005–present | wdc.kugi.kyoto-u.ac.jp |
