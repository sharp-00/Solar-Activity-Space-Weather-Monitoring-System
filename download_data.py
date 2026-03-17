"""
download_data.py
Fetch solar/space-weather datasets and save clean CSVs to data/.

Two modes
---------
python3 download_data.py            # first-time setup: download everything
python3 download_data.py --refresh  # daily use: only fetch recent data (fast)

Strategy
--------
Sunspots (SILSO)
  - Historical file (silso_sunspots_daily.csv / monthly) is downloaded ONCE
    and never re-fetched unless you delete it.
  - --refresh re-downloads just the last ~30 days from SILSO's EISN feed
    and merges them into the existing clean CSV, keeping the rest untouched.

Kp index (GFZ + NOAA)
  - kp_historic_gfz.txt  : GFZ full archive 1932-present, downloaded ONCE.
  - noaa_kp_index.json   : NOAA real-time feed (~7 days), ALWAYS re-fetched.
  - kp_merged_clean.csv  : result of merging both; rebuilt on every run.
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
SILSO_RECENT_URL  = "https://www.sidc.be/silso/DATA/EISN/EISN_current.txt"  # last ~30 days
NOAA_KP_URL       = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
GFZ_KP_HIST_URL   = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"

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
    path = os.path.join(DATA_DIR, name)
    with open(path, "wb") as fh:
        fh.write(data)
    print(f"  [saved] {name}  ({os.path.getsize(path):,} bytes)")
    return path


def _skip_or_fetch(name: str, url: str) -> str:
    """Download to data/<n> only if the file does not already exist."""
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        print(f"  [skip]  {name} already on disk — delete it to re-download")
        return path
    return _save(name, _get(url))


# ---------------------------------------------------------------------------
# Sunspot parsers
# ---------------------------------------------------------------------------

def _parse_silso_daily(raw_path: str) -> pd.DataFrame:
    """Parse the full SILSO daily semicolon-separated file."""
    cols = ["year", "month", "day", "frac_year", "sn", "sn_err", "n_obs", "definitive"]
    df = pd.read_csv(raw_path, sep=";", header=None, names=cols)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    df = df.set_index("date").sort_index()
    df["sn"]     = df["sn"].replace(-1, np.nan).astype(float)
    df["sn_err"] = df["sn_err"].replace(-1, np.nan).astype(float)
    return df[["sn", "sn_err", "n_obs", "definitive"]]


def _parse_silso_monthly(raw_path: str) -> pd.DataFrame:
    """Parse the full SILSO monthly semicolon-separated file."""
    cols = ["year", "month", "frac_year", "sn", "sn_err", "n_obs", "definitive"]
    df = pd.read_csv(raw_path, sep=";", header=None, names=cols)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.set_index("date").sort_index()
    df["sn"]     = df["sn"].replace(-1, np.nan).astype(float)
    df["sn_err"] = df["sn_err"].replace(-1, np.nan).astype(float)
    return df[["sn", "sn_err", "n_obs", "definitive"]]


def _parse_silso_recent(raw_path: str) -> pd.DataFrame:
    """
    Parse SILSO EISN_current.txt — estimated daily sunspot number, last ~30 days.
    Format: YYYY MM DD  SN  SN_err  n_obs
    Lines starting with # are comments.
    """
    rows = []
    with open(raw_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
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
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


# ---------------------------------------------------------------------------
# Kp parsers
# ---------------------------------------------------------------------------

def _parse_gfz_kp_historic(raw_path: str) -> pd.DataFrame:
    """
    Parse the GFZ Kp_ap_Ap_SN_F107_since_1932.txt file.

    Actual format (one row per UT day, columns space-separated):
      YYYY MM DD days days_m BSR dB Kp1 Kp2 Kp3 Kp4 Kp5 Kp6 Kp7 Kp8 ...
      col index:  0    1   2    3      4      5    6   7    8    9   10   11   12   13   14

    Kp values are floats (e.g. 3.333) at column indices 7-14.
    Missing values are indicated by -1.000.
    Lines starting with # are header/comment lines.

    Returns a DataFrame indexed by DatetimeIndex (3-hourly) with column 'Kp'.
    """
    records = []
    offsets_hours = [0, 3, 6, 9, 12, 15, 18, 21]  # 8 three-hour slots per day

    with open(raw_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                # Kp1-Kp8 are float values at index 7 through 14
                kp_vals = [float(p) for p in parts[7:15]]
                base = pd.Timestamp(year, month, day)
                for i, kp in enumerate(kp_vals):
                    if kp < 0:          # -1.000 means missing
                        kp = float("nan")
                    ts = base + pd.Timedelta(hours=offsets_hours[i])
                    records.append({"time_tag": ts, "Kp": kp})
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(records).set_index("time_tag").sort_index()
    df["Kp"] = df["Kp"].clip(0, 9)
    return df


def _parse_noaa_kp_recent(raw_path: str) -> pd.DataFrame:
    """Parse NOAA 3-hourly Kp JSON. Row 0 is the header."""
    with open(raw_path, "r") as fh:
        data = json.load(fh)
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True).dt.tz_localize(None)
    df = df.set_index("time_tag").sort_index()
    df["Kp"] = pd.to_numeric(df["Kp"], errors="coerce").clip(0, 9)
    return df[["Kp"]]


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _merge_kp(historic: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    """
    Combine GFZ historic and NOAA recent Kp into one 3-hourly series.
    Recent data takes priority for any overlapping timestamps.
    """
    combined = historic[~historic.index.isin(recent.index)]
    combined = pd.concat([combined, recent]).sort_index()
    combined.index.name = "time_tag"
    return combined


def _merge_sunspots(historic: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    """
    Overlay recent sunspot readings on top of the historic series.
    Recent rows overwrite historic rows for the same dates.
    """
    combined = historic[~historic.index.isin(recent.index)]
    combined = pd.concat([combined, recent]).sort_index()
    combined.index.name = "date"
    return combined


# ---------------------------------------------------------------------------
# First-time setup
# ---------------------------------------------------------------------------

def setup():
    """Download all data for the first time. Skips files already on disk."""
    print("\n=== Solar Monitor: first-time data setup ===\n")

    # -- Sunspots historical (skip if already downloaded) --
    raw_daily = _skip_or_fetch("silso_sunspots_daily.csv", SILSO_DAILY_URL)
    df = _parse_silso_daily(raw_daily)
    out = os.path.join(DATA_DIR, "sunspots_daily_clean.csv")
    df.to_csv(out)
    print(f"  [clean] sunspots_daily_clean.csv — {len(df):,} rows "
          f"({df.index.min().date()} -> {df.index.max().date()})\n")

    raw_monthly = _skip_or_fetch("silso_sunspots_monthly.csv", SILSO_MONTHLY_URL)
    df = _parse_silso_monthly(raw_monthly)
    out = os.path.join(DATA_DIR, "sunspots_monthly_clean.csv")
    df.to_csv(out)
    print(f"  [clean] sunspots_monthly_clean.csv — {len(df):,} rows "
          f"({df.index.min().date()} -> {df.index.max().date()})\n")

    # -- Kp historic (GFZ, skip if already downloaded) --
    raw_gfz = _skip_or_fetch("kp_historic_gfz.txt", GFZ_KP_HIST_URL)
    kp_hist = _parse_gfz_kp_historic(raw_gfz)
    print(f"  [parsed] GFZ Kp: {len(kp_hist):,} 3-hourly rows "
          f"({kp_hist.index.min().date()} -> {kp_hist.index.max().date()})")

    # -- Kp recent (NOAA, always fetch) --
    noaa_raw  = _save("noaa_kp_index.json", _get(NOAA_KP_URL))
    kp_recent = _parse_noaa_kp_recent(noaa_raw)
    print(f"  [parsed] NOAA Kp: {len(kp_recent):,} rows "
          f"({kp_recent.index.min().date()} -> {kp_recent.index.max().date()})")

    # -- Merge and save Kp --
    kp_merged = _merge_kp(kp_hist, kp_recent)
    out = os.path.join(DATA_DIR, "kp_merged_clean.csv")
    kp_merged.to_csv(out)
    print(f"  [clean] kp_merged_clean.csv — {len(kp_merged):,} rows "
          f"({kp_merged.index.min().date()} -> {kp_merged.index.max().date()})\n")

    _print_summary()


# ---------------------------------------------------------------------------
# Refresh (fast — only recent data)
# ---------------------------------------------------------------------------

def refresh():
    """
    Quick daily refresh: only re-fetch the last ~30 days of sunspots and
    the latest NOAA Kp feed, then merge into existing clean CSVs.
    The large historical files are never re-downloaded.
    """
    print("\n=== Solar Monitor: refreshing recent data ===\n")

    # -- Verify historical files are present --
    for fname in ("sunspots_daily_clean.csv", "kp_historic_gfz.txt"):
        if not os.path.exists(os.path.join(DATA_DIR, fname)):
            print(f"  [error] {fname} not found — run without --refresh first.")
            sys.exit(1)

    # -- Recent sunspots --
    try:
        recent_path = _save("silso_sunspots_recent.txt", _get(SILSO_RECENT_URL))
        df_recent   = _parse_silso_recent(recent_path)

        existing_path = os.path.join(DATA_DIR, "sunspots_daily_clean.csv")
        df_hist   = pd.read_csv(existing_path, index_col="date", parse_dates=True)
        df_merged = _merge_sunspots(df_hist, df_recent)
        df_merged.to_csv(existing_path)
        print(f"  [merged] sunspots_daily_clean.csv — "
              f"latest: {df_merged.index.max().date()}\n")
    except Exception as exc:
        print(f"  [warn] Could not refresh sunspot data: {exc}")

    # -- Recent Kp --
    try:
        noaa_raw  = _save("noaa_kp_index.json", _get(NOAA_KP_URL))
        kp_recent = _parse_noaa_kp_recent(noaa_raw)

        gfz_path  = os.path.join(DATA_DIR, "kp_historic_gfz.txt")
        kp_hist   = _parse_gfz_kp_historic(gfz_path)
        kp_merged = _merge_kp(kp_hist, kp_recent)

        out = os.path.join(DATA_DIR, "kp_merged_clean.csv")
        kp_merged.to_csv(out)
        print(f"  [merged] kp_merged_clean.csv — "
              f"latest: {kp_merged.index.max().date()}\n")
    except Exception as exc:
        print(f"  [warn] Could not refresh Kp data: {exc}")

    _print_summary()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary():
    print("=== data/ contents ===")
    for fname in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, fname)
        print(f"  {fname:45s}  {os.path.getsize(fpath):>10,} bytes")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solar Monitor data downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 download_data.py            # first-time setup\n"
            "  python3 download_data.py --refresh  # quick daily update\n"
        ),
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Only re-fetch recent data; skip large historical downloads."
    )
    args = parser.parse_args()

    if args.refresh:
        refresh()
    else:
        setup()
        print("Done. Run: python3 app.py")
