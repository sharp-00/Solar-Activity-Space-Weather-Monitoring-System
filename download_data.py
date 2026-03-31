"""
download_data.py
Fetch solar/space-weather datasets and save clean CSVs to data/.

Two modes
---------
python3 download_data.py            # first-time setup: download everything
python3 download_data.py --refresh  # hourly/daily use: fetch & append new data

Strategy
--------
Sunspots & Kp: 
  - Pulls deep historical files ONCE. 
  - --refresh pulls real-time feeds and appends new timestamps.

Solar Flare & DST (NOAA):
  - Deep history for 1-minute flux/DST requires heavy NetCDF/monthly scraping.
  - Instead, this script uses the 7-day rolling real-time feeds.
  - Running `--refresh` hourly/daily seamlessly appends the newly available 
    data to your local CSVs, organically building your historical archive 
    moving forward without duplicating timestamps.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Remote URLs
# ---------------------------------------------------------------------------
SILSO_DAILY_URL   = "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv"
SILSO_MONTHLY_URL = "https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv"
SILSO_RECENT_URL  = "https://www.sidc.be/silso/DATA/EISN/EISN_current.txt"
NOAA_KP_URL       = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
GFZ_KP_HIST_URL   = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"

# High-resolution 7-day rolling feeds
NOAA_FLARE_URL    = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
NOAA_DST_URL      = "https://services.swpc.noaa.gov/json/geospace/geospace_dst_7_day.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 120) -> bytes:
    """Download URL, raise on HTTP error, return raw bytes."""
    print(f"  [fetch] {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content

def _save(name: str, data: bytes) -> str:
    """Save raw bytes to disk."""
    path = os.path.join(DATA_DIR, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path

def _skip_or_fetch(name: str, url: str) -> str:
    """Download to data/<n> only if the file does not already exist."""
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        print(f"  [skip]  {name} already on disk — delete it to re-download")
        return path
    return _save(name, _get(url))

def _merge_timeseries(historic: pd.DataFrame, recent: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """
    Safely merges new hourly/daily data into the historical archive.
    Uses index deduplication so overlapping 7-day data never creates duplicates,
    and recent API revisions overwrite older local data.
    """
    if historic is None or historic.empty:
        recent.index.name = index_name
        return recent
    
    # Concat both, sorting by index. 
    # keep='last' ensures the most recently downloaded data overwrites the old.
    combined = pd.concat([historic, recent])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    combined.index.name = index_name
    return combined

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_silso_daily(raw_path: str) -> pd.DataFrame:
    cols = ["year", "month", "day", "frac_year", "sn", "sn_err", "n_obs", "definitive"]
    df = pd.read_csv(raw_path, sep=";", header=None, names=cols)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    df = df.set_index("date").sort_index()
    df["sn"]     = df["sn"].replace(-1, np.nan).astype(float)
    df["sn_err"] = df["sn_err"].replace(-1, np.nan).astype(float)
    return df[["sn", "sn_err", "n_obs", "definitive"]]

def _parse_silso_monthly(raw_path: str) -> pd.DataFrame:
    cols = ["year", "month", "frac_year", "sn", "sn_err", "n_obs", "definitive"]
    df = pd.read_csv(raw_path, sep=";", header=None, names=cols)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.set_index("date").sort_index()
    df["sn"]     = df["sn"].replace(-1, np.nan).astype(float)
    df["sn_err"] = df["sn_err"].replace(-1, np.nan).astype(float)
    return df[["sn", "sn_err", "n_obs", "definitive"]]

def _parse_silso_recent(raw_path: str) -> pd.DataFrame:
    rows = []
    with open(raw_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 4: continue
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                sn     = float(parts[3]) if parts[3] not in ("-1", "999") else np.nan
                sn_err = float(parts[4]) if len(parts) > 4 and parts[4] not in ("-1", "999") else np.nan
                n_obs  = int(parts[5])   if len(parts) > 5 else 0
                rows.append({"date": pd.Timestamp(year, month, day),
                             "sn": sn, "sn_err": sn_err,
                             "n_obs": n_obs, "definitive": 0})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows).set_index("date").sort_index()

def _parse_gfz_kp_historic(raw_path: str) -> pd.DataFrame:
    records = []
    offsets_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    with open(raw_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 15: continue
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                kp_vals = [float(p) for p in parts[7:15]]
                base = pd.Timestamp(year, month, day)
                for i, kp in enumerate(kp_vals):
                    if kp < 0: kp = float("nan")
                    ts = base + pd.Timedelta(hours=offsets_hours[i])
                    records.append({"time_tag": ts, "Kp": kp})
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(records).set_index("time_tag").sort_index()
    df["Kp"] = df["Kp"].clip(0, 9)
    return df

def _parse_noaa_kp_recent(raw_path: str) -> pd.DataFrame:
    with open(raw_path, "r") as fh: data = json.load(fh)
    df = pd.DataFrame(data[1:], columns=data[0])
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True).dt.tz_localize(None)
    df = df.set_index("time_tag").sort_index()
    df["Kp"] = pd.to_numeric(df["Kp"], errors="coerce").clip(0, 9)
    return df[["Kp"]]

def _parse_noaa_flare(raw_path: str) -> pd.DataFrame:
    with open(raw_path, "r") as fh: data = json.load(fh)
    df = pd.DataFrame(data)
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True).dt.tz_localize(None)
    df = df.rename(columns={"flux": "xray_flux", "energy": "energy_band"})
    df = df[["time_tag", "energy_band", "xray_flux"]].dropna()
    return df.set_index("time_tag").sort_index()

def _parse_noaa_dst(raw_path: str) -> pd.DataFrame:
    with open(raw_path, "r") as fh: data = json.load(fh)
    df = pd.DataFrame(data)
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True).dt.tz_localize(None)
    if "dst" in df.columns:
        df["dst"] = pd.to_numeric(df["dst"], errors="coerce")
    df = df[["time_tag", "dst"]].dropna()
    return df.set_index("time_tag").sort_index()

# ---------------------------------------------------------------------------
# Core Routines
# ---------------------------------------------------------------------------

def setup():
    """First-time run: Fetches available history and establishes baselines."""
    print("\n=== Solar Monitor: First-Time Data Setup ===\n")

    # -- Sunspots --
    raw_daily = _skip_or_fetch("silso_sunspots_daily.csv", SILSO_DAILY_URL)
    df = _parse_silso_daily(raw_daily)
    df.to_csv(os.path.join(DATA_DIR, "sunspots_daily_clean.csv"))
    print(f"  [clean] sunspots_daily_clean.csv — {len(df):,} rows")

    raw_monthly = _skip_or_fetch("silso_sunspots_monthly.csv", SILSO_MONTHLY_URL)
    df = _parse_silso_monthly(raw_monthly)
    df.to_csv(os.path.join(DATA_DIR, "sunspots_monthly_clean.csv"))
    print(f"  [clean] sunspots_monthly_clean.csv — {len(df):,} rows")

    # -- Kp Index --
    raw_gfz = _skip_or_fetch("kp_historic_gfz.txt", GFZ_KP_HIST_URL)
    kp_hist = _parse_gfz_kp_historic(raw_gfz)
    noaa_raw = _save("noaa_kp_index.json", _get(NOAA_KP_URL))
    kp_recent = _parse_noaa_kp_recent(noaa_raw)
    kp_merged = _merge_timeseries(kp_hist, kp_recent, "time_tag")
    kp_merged.to_csv(os.path.join(DATA_DIR, "kp_merged_clean.csv"))
    print(f"  [clean] kp_merged_clean.csv — {len(kp_merged):,} rows")

    # -- Solar Flare (Init with 7-days) --
    flare_raw = _save("solar_flare_raw.json", _get(NOAA_FLARE_URL))
    flare_df = _parse_noaa_flare(flare_raw)
    flare_df.to_csv(os.path.join(DATA_DIR, "solar_flare_clean.csv"))
    print(f"  [clean] solar_flare_clean.csv — {len(flare_df):,} rows")

    # -- DST (Init with 7-days) --
    dst_raw = _save("dst_raw.json", _get(NOAA_DST_URL))
    dst_df = _parse_noaa_dst(dst_raw)
    dst_df.to_csv(os.path.join(DATA_DIR, "dst_clean.csv"))
    print(f"  [clean] dst_clean.csv — {len(dst_df):,} rows\n")

def refresh():
    """
    Hourly/Daily update routine. Safe to run on a cron job.
    Fetches newly available data and dynamically appends it to your CSVs.
    """
    print("\n=== Solar Monitor: Hourly/Daily Refresh ===\n")

    # -- Sunspots --
    try:
        recent_path = _save("silso_sunspots_recent.txt", _get(SILSO_RECENT_URL))
        df_recent   = _parse_silso_recent(recent_path)
        csv_path    = os.path.join(DATA_DIR, "sunspots_daily_clean.csv")
        if os.path.exists(csv_path):
            df_hist = pd.read_csv(csv_path, index_col="date", parse_dates=True)
            df_merged = _merge_timeseries(df_hist, df_recent, "date")
            df_merged.to_csv(csv_path)
            print(f"  [appended] sunspots_daily_clean.csv — Latest: {df_merged.index.max().date()}")
    except Exception as exc: print(f"  [warn] Sunspot refresh failed: {exc}")

    # -- Kp Index --
    try:
        noaa_raw  = _save("noaa_kp_index.json", _get(NOAA_KP_URL))
        kp_recent = _parse_noaa_kp_recent(noaa_raw)
        csv_path = os.path.join(DATA_DIR, "kp_merged_clean.csv")
        if os.path.exists(csv_path):
            kp_hist = pd.read_csv(csv_path, index_col="time_tag", parse_dates=True)
            kp_merged = _merge_timeseries(kp_hist, kp_recent, "time_tag")
            kp_merged.to_csv(csv_path)
            print(f"  [appended] kp_merged_clean.csv — Latest: {kp_merged.index.max()}")
    except Exception as exc: print(f"  [warn] Kp refresh failed: {exc}")

    # -- Solar Flare --
    try:
        flare_raw = _save("solar_flare_raw.json", _get(NOAA_FLARE_URL))
        flare_recent = _parse_noaa_flare(flare_raw)
        csv_path = os.path.join(DATA_DIR, "solar_flare_clean.csv")
        if os.path.exists(csv_path):
            flare_hist = pd.read_csv(csv_path, index_col="time_tag", parse_dates=True)
            flare_merged = _merge_timeseries(flare_hist, flare_recent, "time_tag")
            flare_merged.to_csv(csv_path)
            print(f"  [appended] solar_flare_clean.csv — Latest: {flare_merged.index.max()}")
    except Exception as exc: print(f"  [warn] Solar Flare refresh failed: {exc}")

    # -- DST --
    try:
        dst_raw = _save("dst_raw.json", _get(NOAA_DST_URL))
        dst_recent = _parse_noaa_dst(dst_raw)
        csv_path = os.path.join(DATA_DIR, "dst_clean.csv")
        if os.path.exists(csv_path):
            dst_hist = pd.read_csv(csv_path, index_col="time_tag", parse_dates=True)
            dst_merged = _merge_timeseries(dst_hist, dst_recent, "time_tag")
            dst_merged.to_csv(csv_path)
            print(f"  [appended] dst_clean.csv — Latest: {dst_merged.index.max()}")
    except Exception as exc: print(f"  [warn] DST refresh failed: {exc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Space Weather Data Downloader")
    parser.add_argument("--refresh", action="store_true", help="Fetch and safely append newly available hourly/daily data.")
    args = parser.parse_args()

    if args.refresh:
        refresh()
    else:
        setup()
        print("Done. Ready for pipeline/analysis.")
















