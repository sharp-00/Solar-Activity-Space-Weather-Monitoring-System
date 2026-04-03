# Space Weather Analytics System

An end-to-end data pipeline and interactive analytics dashboard that ingests, harmonises, and visualises key solar and geomagnetic indices—spanning nearly four decades (1986–present). The platform serves as a unified system to monitor the dynamic conditions of the Sun and the Earth's magnetosphere, addressing the fragmentation of existing data portals by bringing multiple historical and near-real-time indices into a single dashboard.

## Features

- **Unified Intelligence:** Integrates 5 key indices (Sunspot Number, F10.7 solar radio flux, Kp index, Dst storm-time index, and Solar Flare counts).
- **Streamlit Analytics Dashboard:** A 15-page interactive dashboard providing real-time exploration of time series, cross-correlations, periodicity, hysteresis, and extreme events.
- **Robust Data Pipeline:** Features automated data extraction from 6 authoritative sources with mirror failover, gap-filling linear interpolation, and automatic migration protocols.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run app.py
```

On first launch, the dashboard will auto-detect missing data and run the data ingestion and cleaning pipeline automatically. You can also trigger data updates via the "Fetch Latest Data" button available in the dashboard interface at any time.

---

## Data Pipeline Architecture

The system follows a three-stage Extract-Transform-Load (ETL) architecture:

1. **Ingest (`ingest.py`)**: Downloads raw observations from 6 distinct providers across 13 feeds. It features retry-with-backoff logic and mirror failovers for reliability. Original data is stored in `data/raw/`.
2. **Clean (`clean.py`)**: Harmonises the heterogeneous data formats (fixed-width, CSV, JSON, Fortran-formatted) and merges overlapping sources using a priority-based scheme. Small data gaps (up to 7 days) are handled via linear interpolation for continuous variables, and up to 3 days using forward-fill for count variables. Data outputs are saved to `data/clean/*.parquet`.
3. **Analyze (`analysis.py`)**: Produces statistical derivations like cross-correlations, Fast Fourier Transform (FFT) and Continuous Wavelet Transform (CWT) periodicity detection, phase-locked climatology, and categorised activity levels for geomagnetics.

**Manual Pipeline Execution:**
```bash
python ingest.py   # ~10-20 min (First time will be slower due to throttling limits like Kyoto's per-request sleep)
python clean.py    # ~30 sec
```

---

## Project Structure

```text
solar/
├── app.py                  # Home page displaying KPI cards and global date filters
├── pages/                  # 15 distinct analysis pages (Streamlit sidebar navigation)
│   ├── 1_Solar_Timeseries.py
│   ├── 2_System_Overview.py
│   ├── 3_Physics_Primer.py
│   ├── 4_Storm_Simulator.py
│   ├── 5_Geospatial_Impact.py
│   ├── 6_Monthly_Stats.py
│   ├── 7_Data_Smoothing.py
│   ├── 8_Correlations.py
│   ├── 9_Lag_Analysis.py
│   ├── 10_Periodicity.py
│   ├── 11_Phase_Climatology.py
│   ├── 12_Hysteresis.py
│   ├── 13_Extreme_Events.py
│   ├── 14_Data_Sources.py
│   └── 15_Project_Details.py
├── utils/
│   ├── data_loader.py      # Cached parquet loaders
│   ├── refresh.py          # Auto-refresh session configurations and button logic
│   └── theme.py            # Global theme configurations, Plotly settings and CSS overrides
├── ingest.py               # Raw data extraction module
├── clean.py                # Data harmonization and Parquet generation module
├── analysis.py             # Advanced statistics suite (FFT, CWT, Correlator)
└── requirements.txt        # Python package dependencies
```

---

## Data Sources & Refresh Cycle

### Refresh Logic
| Trigger | Behaviour |
|---|---|
| App start (no data) | Full pipeline runs automatically |
| > 6 hours from fetch | Background auto-refresh on subsequent page load |
| Manual "Fetch Latest" | Instant retrieval, button available across all pages |

### Supported Sources
The pipeline aggregates approximately 200 MB of raw logs into over 14,000 daily observations tracking indices like Sunspot Numbers, Solar radio flux, Kp/ap, and Dst from the following authorities:

| Source Provider | Monitored Variables | Primary Endpoint URL |
|---|---|---|
| SILSO / Observatory Brussels | Daily Sunspot Number (SSN) | sidc.be |
| NOAA NCEI / SWPC | F10.7 flux, Flares, Kp index | swpc.noaa.gov |
| GFZ Potsdam | Kp/ap 3-hourly indices | kp.gfz.de |
| NASA SPDF OMNI2 | Dst index (1986–2004) | spdf.gsfc.nasa.gov |
| WDC Kyoto | Dst index (2005–present) | wdc.kugi.kyoto-u.ac.jp |

---

## Technical Outcomes
- Fully interactive visual analysis showcasing insights like the ~11-year solar cycle periodicities and the lag/hysteresis effects connecting peak sunspot periods to eventual high-scale geomagnetic storms observed on Earth.
- High-performance `pandas` and `parquet` backends delivering quick rendering speeds for nearly 40 years of continuous daily analytical data.

## License
Provided as-is for space-weather monitoring and educational capabilities.
