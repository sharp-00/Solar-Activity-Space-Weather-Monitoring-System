# ☀️ Solar Activity & Space Weather Monitoring System

An advanced, interactive analytics dashboard for monitoring solar activity and its geospatial impacts on Earth. Built with **Python 3.x** and **Streamlit**, this project bridges the gap between raw astrophysical telemetry and actionable data science.

## Quick Start

### 1. Install Dependencies and Create Environment
```bash
conda env create -f environment.yml
```
### 2. Activate Environment
```bash
conda activate solar_analysis
```

### 3. Fetch Data
First-time data ingestion (fetches SILSO, OMNIWeb, and NOAA feeds):
```bash
python3 download_data.py
```

### 4. Launch Dashboard
```bash
streamlit run dashboard.py
```

---

## Project Architecture

```bash
Solar-Activity-Monitoring/
├── dashboard.py         # Main entry point (Streamlit UI)
├── analysis.py          # Core statistical engine (Cross-correlation, FFT, Wavelets)
├── ingest.py            # Data loading and ingestion utilities
├── download_data.py     # Data fetching and update pipeline
├── clean.py             # Data cleaning and harmonization
├── Images_dashboard/     # Visual resources for the primer and simulator
└── data/                # Data storage (Cleaned and Analyzed stats)
    ├── clean/           # Harmonized daily solar-weather datasets
    └── analysis/stats/   # Pre-calculated statistical results (Lags, Epochs, etc.)
```

---

## Dashboard Modules

| Module | Purpose | Key Features |
|:---|:---|:---|
| **Solar Number Time Series** | Historical Analysis | 200 years of SILSO data, rolling means, and monthly heatmaps. |
| **System Overview** | Cause & Effect | Direct contrast of solar drivers (SSN, F10.7) vs terrestrial response (Kp, Dst). |
| **Physics Primer** | Education | Detailed science behind flares, CMEs, and the 11-year solar cycle. |
| **Storm Simulator** | Interactive Tool | NOAA G-Scale simulator mapping Kp indices to global infrastructure impacts. |
| **Geospatial Impact** | Aurora Mapping | Interactive 3D globe showing the expansion of the auroral oval during storms. |
| **Lag-Time Analysis** | Causality Analysis | Superposed Epoch Analysis (SEA) to visualize the ~2-4 day delay of CME transit. |
| **Periodicity Analysis** | Frequency Domain | FFT and Wavelet transforms to detect the 11-year and 27-day solar cycles. |
| **Hysteresis & Climatology** | Cycle Dynamics | Analysis of the rising vs. falling phase effects on geomagnetic sensitivity. |

---

## Science Behind the Data

-   **Temporal Alignment**: Uses Daily resolution to capture the 1-4 day lag between solar eruptions and geomagnetic impacts (invisible at monthly resolution).
-   **Noise Filtration**: Implements **Savitzky-Golay** polynomial filters to preserve the timing of extreme flare peaks that simple moving averages might flatten.
-   **Statistical Detection**: Uses **Z-Score analysis** to automatically flag extreme space weather events beyond 2.5σ deviations.

---

## Data Sources

-   **SILSO**: Sunspot Index and Long-term Solar Observations (Royal Observatory of Belgium).
-   **NASA OMNIWeb**: Hourly and Daily OMNI parameters (Dst Index, Solar Wind properties).
-   **NOAA SWPC**: Real-time X-ray flares and Kp telemetry.
-   **GFZ Potsdam**: Definitive historical Kp index archive.

---

## The Team
-   **Mulumudi Dinesh Karthik**
-   **Vansh Gupta**
-   **Abhishek Menon**
-   **Shailendra Pratap Singh**

Built with ❤️.
