"""
clean.py  —  Data Cleaning & Harmonisation Pipeline
----------------------------------------------------
Reads raw files produced by ingest.py, cleans each source,
and merges them into a single daily-resolution CSV.

Output:
    data/clean/solar_weather_daily.csv

Sources parsed:
    1. SSN   — SILSO daily sunspot number
    2. DSD   — NCEI Daily Solar Data (F10.7, flare counts, sunspot area)
    3. DGD   — NCEI Daily Geomagnetic Data (Kp indices, A indices)
    4. Dst   — Kyoto WDC hourly Dst → aggregated to daily
    5. F10.7 — NOAA SWPC JSON (recent supplement)

Usage:
    python clean.py
"""

import re
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR   = Path("data/raw")
CLEAN_DIR = Path("data/clean")

START_DATE = "1986-01-01"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. SSN — SILSO Daily Sunspot Number
# ---------------------------------------------------------------------------

def parse_ssn(path: Path) -> pd.DataFrame:
    """
    Parse SILSO daily total sunspot number file.

    Format: year month day decimal_year SSN std_dev n_obs definitive_flag
    Missing SSN = -1. Space-delimited, no header.
    """
    log.info("[SSN] Parsing SILSO daily sunspot number ...")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["year", "month", "day", "decimal_year", "ssn", "ssn_std", "n_obs", "definitive"],
        dtype={"year": int, "month": int, "day": int},
    )

    # Build date column
    df["date"] = pd.to_datetime(
        df[["year", "month", "day"]].rename(columns={"year": "year", "month": "month", "day": "day"}),
        errors="coerce",
    )
    df = df.dropna(subset=["date"])

    # Filter to 1996+
    df = df[df["date"] >= START_DATE].copy()

    # Replace missing values (-1)
    df["ssn"] = df["ssn"].replace(-1, np.nan)
    df["ssn_std"] = df["ssn_std"].replace(-1.0, np.nan)

    # Keep only what we need
    result = df[["date", "ssn", "ssn_std"]].copy()
    result = result.set_index("date").sort_index()

    log.info(f"[SSN] ✓ {len(result)} daily records, "
             f"{result['ssn'].notna().sum()} with valid SSN, "
             f"range: {result.index.min().date()} → {result.index.max().date()}")

    return result


# ---------------------------------------------------------------------------
# 2. DSD — NCEI Daily Solar Data
# ---------------------------------------------------------------------------

def parse_dsd(path: Path) -> pd.DataFrame:
    """
    Parse concatenated NCEI Daily Solar Data files.

    Columns (fixed-width, space-delimited):
        year month day  f107  ssn_sesc  spot_area  new_regions  mean_field
        xray_bkgd  flare_C  flare_M  flare_X  flare_S  opt_1  opt_2  opt_3

    Missing: -999 for numeric, '*' for X-ray background.
    Header lines start with ':', '#', or are blank.
    """
    log.info("[DSD] Parsing NCEI Daily Solar Data ...")

    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(":") or line.startswith("#") or line.startswith("Product"):
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue

            # Parse fields with missing-value handling
            def safe_int(val, missing=(-999,)):
                try:
                    v = int(val)
                    return np.nan if v in missing else v
                except (ValueError, TypeError):
                    return np.nan

            def safe_float(val, missing=(-999,)):
                try:
                    v = float(val)
                    return np.nan if v in missing else v
                except (ValueError, TypeError):
                    return np.nan

            row = {
                "year": year, "month": month, "day": day,
                "f107_dsd": safe_float(parts[3]),
                "ssn_sesc": safe_int(parts[4]),
                "sunspot_area": safe_int(parts[5]),
                "new_regions": safe_int(parts[6]),
                "solar_mean_field": safe_float(parts[7]),
                "xray_bkgd": parts[8] if parts[8] != "*" else np.nan,
            }

            # Flare counts (may have fewer columns in some files)
            flare_cols = ["flare_C", "flare_M", "flare_X", "flare_S",
                          "flare_opt1", "flare_opt2", "flare_opt3"]
            for i, col in enumerate(flare_cols):
                idx = 9 + i
                row[col] = safe_int(parts[idx]) if idx < len(parts) else np.nan

            rows.append(row)

    df = pd.DataFrame(rows)

    # Build date
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= START_DATE].copy()

    # Total X-ray flare count
    df["flare_xray_total"] = df[["flare_C", "flare_M", "flare_X"]].sum(axis=1, min_count=1)

    # Total optical flare count
    df["flare_optical_total"] = df[["flare_opt1", "flare_opt2", "flare_opt3"]].sum(axis=1, min_count=1)

    # Drop duplicates (concatenated files may have overlap)
    df = df.drop_duplicates(subset=["date"], keep="last")

    keep = ["date", "f107_dsd", "ssn_sesc", "sunspot_area", "new_regions",
            "solar_mean_field", "xray_bkgd",
            "flare_C", "flare_M", "flare_X", "flare_S",
            "flare_xray_total", "flare_optical_total"]
    result = df[keep].set_index("date").sort_index()

    log.info(f"[DSD] ✓ {len(result)} daily records, "
             f"F10.7 valid: {result['f107_dsd'].notna().sum()}, "
             f"range: {result.index.min().date()} → {result.index.max().date()}")

    return result


# ---------------------------------------------------------------------------
# 3. DGD — NCEI Daily Geomagnetic Data
# ---------------------------------------------------------------------------

def parse_dgd(path: Path) -> pd.DataFrame:
    """
    Parse concatenated NCEI Daily Geomagnetic Data files.

    Two formats:
      Old (1996-2022):
        date  A_fred  K1..K8_fred  A_coll  K1..K8_coll  A_plan  K1..K8_plan
        = 30 fields: 3 date + 9 fred + 9 college + 9 planetary

      New (2023+):
        date  A_fred  K1..K8_fred  A_coll  K1..K8_coll  A_plan  Kp1..Kp8_plan(decimal)
        = 30 fields: 3 date + 9 fred + 9 college + 9 planetary (with decimal Kp)

    We extract: date, A_planetary, 8 × Kp_planetary, daily_mean_Kp.
    """
    log.info("[DGD] Parsing NCEI Daily Geomagnetic Data ...")

    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(":") or line.startswith("#") or line.startswith("Product"):
                continue
            parts = line.split()
            if len(parts) < 21:
                continue
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue

            # Fredericksburg: A index at [3], K1-K8 at [4:12]
            # College:        A index at [12], K1-K8 at [13:21]
            # Planetary:      A index at [21], K1-K8 or Kp1-Kp8 at [22:30]

            def safe_float(val):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return np.nan

            def safe_int(val, missing=(-1, -999)):
                try:
                    v = int(val)
                    return np.nan if v in missing else v
                except (ValueError, TypeError):
                    return np.nan

            a_fred = safe_int(parts[3]) if len(parts) > 3 else np.nan
            a_college = safe_int(parts[12]) if len(parts) > 12 else np.nan
            a_planetary = safe_int(parts[21]) if len(parts) > 21 else np.nan

            # Planetary K/Kp indices (positions 22-29)
            kp_vals = []
            for i in range(22, min(30, len(parts))):
                kp_vals.append(safe_float(parts[i]))

            row = {
                "year": year, "month": month, "day": day,
                "a_fredericksburg": a_fred,
                "a_college": a_college,
                "ap_planetary": a_planetary,
            }

            # Store 8 Kp values
            for i, kp in enumerate(kp_vals):
                row[f"kp_{i+1}"] = kp

            # Daily mean Kp (average of 8 three-hourly values)
            valid_kp = [v for v in kp_vals if not np.isnan(v)]
            row["kp_daily_mean"] = np.mean(valid_kp) if valid_kp else np.nan

            # Daily max Kp
            row["kp_daily_max"] = np.max(valid_kp) if valid_kp else np.nan

            rows.append(row)

    df = pd.DataFrame(rows)

    # Build date
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= START_DATE].copy()

    # Drop duplicates
    df = df.drop_duplicates(subset=["date"], keep="last")

    keep = ["date", "ap_planetary", "a_fredericksburg", "a_college",
            "kp_daily_mean", "kp_daily_max",
            "kp_1", "kp_2", "kp_3", "kp_4", "kp_5", "kp_6", "kp_7", "kp_8"]
    result = df[[c for c in keep if c in df.columns]].set_index("date").sort_index()

    log.info(f"[DGD] ✓ {len(result)} daily records, "
             f"Kp valid: {result['kp_daily_mean'].notna().sum()}, "
             f"range: {result.index.min().date()} → {result.index.max().date()}")

    return result


# ---------------------------------------------------------------------------
# 4. Dst — Kyoto WDC Hourly → Daily
# ---------------------------------------------------------------------------

def parse_dst_kyoto(path: Path) -> pd.DataFrame:
    """
    Parse concatenated Kyoto WDC Dst .for.request files.

    WDC format (fixed-width, one line per day):
        Cols 1-3:   'DST'
        Cols 4-5:   2-digit year (YY)
        Cols 6-7:   2-digit month (MM)
        Col  8:     '*' separator
        Cols 9-10:  day of month
        Cols 15-16: century or version marker
        Col  21:    zero marker
        Cols 21-116: 24 hourly Dst values, each 4 chars wide
        Cols 117-120: daily mean Dst (4 chars)

    Each line has: header + 24 × 4-char hourly values + 4-char daily mean.
    Total line length = ~120 chars.
    """
    log.info("[Dst] Parsing Kyoto WDC Dst data ...")

    hourly_rows = []
    daily_rows = []

    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line.startswith("DST"):
                continue
            if len(line) < 116:
                continue

            try:
                # Parse header: DSTyyMM*DD
                yy = int(line[3:5])
                mm = int(line[5:7])
                dd = int(line[8:10])

                # Determine century
                century = 20 if yy < 57 else 19
                year = century * 100 + yy

                # Parse 24 hourly values (each 4 chars wide, starting at position 20)
                hourly_vals = []
                for h in range(24):
                    start = 20 + h * 4
                    end = start + 4
                    val_str = line[start:end].strip()
                    try:
                        val = int(val_str)
                        # 9999 is the fill value for missing
                        hourly_vals.append(val if abs(val) < 9999 else np.nan)
                    except ValueError:
                        hourly_vals.append(np.nan)

                # Parse daily mean (last 4 chars of the data section)
                try:
                    daily_mean_str = line[116:120].strip()
                    daily_mean = int(daily_mean_str)
                    if abs(daily_mean) >= 9999:
                        daily_mean = np.nan
                except (ValueError, IndexError):
                    daily_mean = np.nan

                date = pd.Timestamp(year=year, month=mm, day=dd)

                # Store hourly records
                for h, val in enumerate(hourly_vals):
                    hourly_rows.append({
                        "datetime": date + pd.Timedelta(hours=h),
                        "dst_hourly": val,
                    })

                # Store daily aggregate
                valid = [v for v in hourly_vals if not np.isnan(v)]
                daily_rows.append({
                    "date": date,
                    "dst_daily_mean": np.mean(valid) if valid else np.nan,
                    "dst_daily_min": np.min(valid) if valid else np.nan,
                    "dst_daily_max": np.max(valid) if valid else np.nan,
                    "dst_daily_std": np.std(valid) if len(valid) > 1 else np.nan,
                    "dst_wdc_mean": daily_mean,
                    "dst_hours_valid": len(valid),
                })

            except (ValueError, IndexError) as e:
                continue

    # Build daily DataFrame
    df_daily = pd.DataFrame(daily_rows)
    if df_daily.empty:
        log.warning("[Dst] No daily records parsed!")
        return pd.DataFrame()

    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily[df_daily["date"] >= START_DATE].copy()
    df_daily = df_daily.drop_duplicates(subset=["date"], keep="last")
    df_daily = df_daily.set_index("date").sort_index()

    # Also save hourly data separately
    df_hourly = pd.DataFrame(hourly_rows)
    if not df_hourly.empty:
        df_hourly["datetime"] = pd.to_datetime(df_hourly["datetime"])
        df_hourly = df_hourly[df_hourly["datetime"] >= START_DATE].copy()
        df_hourly = df_hourly.drop_duplicates(subset=["datetime"], keep="last")
        df_hourly = df_hourly.set_index("datetime").sort_index()

        hourly_path = CLEAN_DIR / "dst_hourly.csv"
        df_hourly.to_csv(hourly_path)
        log.info(f"[Dst] Hourly data saved → {hourly_path} ({len(df_hourly)} records)")

    log.info(f"[Dst] ✓ {len(df_daily)} daily records, "
             f"range: {df_daily.index.min().date()} → {df_daily.index.max().date()}")

    return df_daily


# ---------------------------------------------------------------------------
# 5. F10.7 — NOAA SWPC JSON (recent supplement)
# ---------------------------------------------------------------------------


def parse_dst_omni2(raw_dir: Path) -> pd.DataFrame:
    """
    Parse NASA SPDF OMNI2 yearly data files for Dst index.

    OMNI2 format: whitespace-delimited, ~55 columns per line, one line per hour.
    Column index 40 (0-based) is Dst in nT. Fill value = 99999.
    Columns 0,1,2 = Year, DOY, Hour.
    """
    log.info("[Dst/OMNI2] Parsing NASA SPDF OMNI2 yearly files ...")

    omni_files = sorted(raw_dir.glob("omni2_*.dat"))
    if not omni_files:
        log.warning("[Dst/OMNI2] No OMNI2 files found.")
        return pd.DataFrame()

    daily_rows = []

    for fpath in omni_files:
        log.info(f"[Dst/OMNI2] Parsing {fpath.name} ...")
        with open(fpath, "r") as f:
            hourly_by_date = {}
            for line in f:
                parts = line.split()
                if len(parts) < 41:
                    continue
                try:
                    year = int(parts[0])
                    doy = int(parts[1])
                    dst_val = int(parts[40])
                except (ValueError, IndexError):
                    continue

                if abs(dst_val) >= 99999:
                    continue

                try:
                    date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
                except (ValueError, OverflowError):
                    continue

                if date not in hourly_by_date:
                    hourly_by_date[date] = []
                hourly_by_date[date].append(dst_val)

            for date, vals in hourly_by_date.items():
                daily_rows.append({
                    "date": date,
                    "dst_daily_mean": np.mean(vals),
                    "dst_daily_min": np.min(vals),
                    "dst_daily_max": np.max(vals),
                    "dst_daily_std": np.std(vals) if len(vals) > 1 else np.nan,
                    "dst_hours_valid": len(vals),
                })

    if not daily_rows:
        log.warning("[Dst/OMNI2] No valid Dst records parsed.")
        return pd.DataFrame()

    df = pd.DataFrame(daily_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= START_DATE].copy()
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date").sort_index()

    log.info(f"[Dst/OMNI2] ✓ {len(df)} daily records, "
             f"range: {df.index.min().date()} → {df.index.max().date()}")
    return df

def parse_f107_json(path: Path) -> pd.DataFrame:
    """
    Parse NOAA SWPC F10.7 cm flux JSON feed.

    Fields: time_tag, frequency, flux, reporting_schedule,
            avg_begin_date, ninety_day_mean, rec_count

    We keep only 'Noon' measurements (standard adjusted-to-1AU).
    """
    log.info("[F10.7] Parsing NOAA SWPC JSON feed ...")

    with open(path, "r") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        if entry.get("reporting_schedule") != "Noon":
            continue
        try:
            dt = pd.Timestamp(entry["time_tag"])
            flux = float(entry["flux"]) if entry["flux"] is not None else np.nan
            rows.append({
                "date": dt.normalize(),
                "f107_swpc": flux,
                "f107_90d_mean": float(entry["ninety_day_mean"]) if entry.get("ninety_day_mean") else np.nan,
            })
        except (ValueError, TypeError, KeyError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("[F10.7] No records parsed from JSON!")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date").sort_index()

    log.info(f"[F10.7] ✓ {len(df)} daily records, "
             f"range: {df.index.min().date()} → {df.index.max().date()}")

    return df


# ---------------------------------------------------------------------------
# 6. Flare Events (1996-2016) — legacy XRS archive
# ---------------------------------------------------------------------------

def parse_flare_events(path: Path) -> pd.DataFrame:
    """
    Parse concatenated NGDC legacy XRS flare event files.

    Format (fixed-width):
        Col 0-4:   satellite ID
        Col 5-6:   year (2-digit)
        Col 6-8:   month
        Col 8-10:  day
        Col 11-14: start time HHMM
        Col 15-19: end time HHMM
        Col 20-23: peak time HHMM
        Col 24-29: location (lat/lon)
        ...
        Col ~59:   class letter (B/C/M/X)
        Col ~60-61: class magnitude

    We aggregate to daily flare event counts by class for cross-check.
    """
    log.info("[Flares] Parsing legacy XRS event files (1996-2016) ...")

    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if len(line) < 60:
                continue
            # Try to extract date and class
            try:
                # Date portion
                date_str = line[0:10].strip()
                if not date_str or not date_str[0].isdigit():
                    continue

                # The format varies — try parsing the first 10 chars
                # Format: SSYYMMDDnn (SS=sat, YYMMDD=date)
                sat = line[0:5].strip()
                yy = int(line[5:7])
                mm = int(line[7:9])
                dd = int(line[9:11])

                century = 20 if yy < 50 else 19
                year = century * 100 + yy

                # Find flare class — search for B/C/M/X followed by digit
                class_match = re.search(r'([BCMX])\s*(\d+\.?\d*)', line[40:])
                if class_match:
                    flare_class = class_match.group(1)
                    flare_mag = float(class_match.group(2))
                else:
                    flare_class = "unknown"
                    flare_mag = np.nan

                rows.append({
                    "date": pd.Timestamp(year=year, month=mm, day=dd),
                    "flare_class": flare_class,
                    "flare_magnitude": flare_mag,
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("[Flares] No events parsed!")
        return pd.DataFrame()

    # Aggregate to daily counts by class
    daily = df.groupby(["date", "flare_class"]).size().unstack(fill_value=0)
    daily.columns = [f"flare_event_{c}" for c in daily.columns]

    # Total events per day
    daily["flare_events_total"] = daily.sum(axis=1)

    daily = daily[daily.index >= START_DATE].sort_index()

    log.info(f"[Flares] ✓ {len(daily)} days with events, "
             f"{len(rows)} total events parsed, "
             f"range: {daily.index.min().date()} → {daily.index.max().date()}")

    return daily


# ---------------------------------------------------------------------------
# Harmonisation & Merge
# ---------------------------------------------------------------------------

def harmonise_f107(dsd: pd.DataFrame, f107_json: pd.DataFrame) -> pd.Series:
    """
    Create a unified F10.7 series.

    Priority: DSD historical data first, SWPC JSON as supplement for recent dates.
    """
    log.info("[Harmonise] Creating unified F10.7 series ...")

    f107 = dsd["f107_dsd"].copy()

    if not f107_json.empty and "f107_swpc" in f107_json.columns:
        # Fill gaps in DSD with SWPC JSON values
        swpc = f107_json["f107_swpc"]
        # Only use SWPC for dates where DSD is missing
        missing_dates = f107.index[f107.isna()]
        new_dates = swpc.index.difference(f107.index)
        fill_dates = missing_dates.union(new_dates)
        f107 = f107.combine_first(swpc)

        n_filled = swpc.reindex(fill_dates).notna().sum()
        log.info(f"[Harmonise] F10.7: {n_filled} values supplemented from SWPC JSON")

    return f107


def merge_all(
    ssn: pd.DataFrame,
    dsd: pd.DataFrame,
    dgd: pd.DataFrame,
    dst: pd.DataFrame,
    f107_json: pd.DataFrame,
    flare_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all cleaned sources into a single daily DataFrame.
    """
    log.info("[Merge] Combining all sources ...")

    # Start with a complete daily date range
    all_dates = set()
    for df in [ssn, dsd, dgd, dst, f107_json]:
        if not df.empty:
            all_dates.update(df.index)

    if not all_dates:
        raise RuntimeError("No data to merge!")

    date_range = pd.date_range(start=min(all_dates), end=max(all_dates), freq="D")
    merged = pd.DataFrame(index=date_range)
    merged.index.name = "date"

    # 1. SSN
    if not ssn.empty:
        merged = merged.join(ssn[["ssn", "ssn_std"]], how="left")
        log.info(f"  + SSN: {merged['ssn'].notna().sum()} values")

    # 2. DSD — F10.7 (harmonised), sunspot area, flare counts
    if not dsd.empty:
        # Harmonise F10.7
        f107_unified = harmonise_f107(dsd, f107_json)
        merged["f107"] = f107_unified

        dsd_cols = ["sunspot_area", "new_regions", "solar_mean_field",
                    "flare_C", "flare_M", "flare_X", "flare_S",
                    "flare_xray_total", "flare_optical_total"]
        for col in dsd_cols:
            if col in dsd.columns:
                merged = merged.join(dsd[[col]], how="left")

        log.info(f"  + DSD: F10.7={merged['f107'].notna().sum()}, "
                 f"flare_C={merged.get('flare_C', pd.Series()).notna().sum()}")

    # 3. DGD — Kp, Ap
    if not dgd.empty:
        dgd_cols = ["ap_planetary", "kp_daily_mean", "kp_daily_max",
                    "kp_1", "kp_2", "kp_3", "kp_4", "kp_5", "kp_6", "kp_7", "kp_8"]
        dgd_available = [c for c in dgd_cols if c in dgd.columns]
        merged = merged.join(dgd[dgd_available], how="left")
        log.info(f"  + DGD: Kp={merged['kp_daily_mean'].notna().sum()}, "
                 f"Ap={merged['ap_planetary'].notna().sum()}")

    # 4. Dst
    if not dst.empty:
        dst_cols = ["dst_daily_mean", "dst_daily_min", "dst_daily_max",
                    "dst_daily_std", "dst_hours_valid"]
        dst_available = [c for c in dst_cols if c in dst.columns]
        merged = merged.join(dst[dst_available], how="left")
        log.info(f"  + Dst: {merged['dst_daily_mean'].notna().sum()} values")

    # 5. Flare events (2008-2016, event-level)
    if not flare_events.empty:
        # Prefix to avoid collision with DSD flare counts
        merged = merged.join(flare_events, how="left")
        log.info(f"  + Flare events: {merged.get('flare_events_total', pd.Series()).notna().sum()} days")

    return merged


# ---------------------------------------------------------------------------
# Interpolation — fill missing data
# ---------------------------------------------------------------------------

# Columns that are smooth physical time series → linear interpolation
INTERP_LINEAR_COLS = [
    "ssn", "ssn_std",
    "f107",
    "sunspot_area",
    "dst_daily_mean", "dst_daily_min", "dst_daily_max", "dst_daily_std",
    "ap_planetary", "a_fredericksburg", "a_college",
    "kp_daily_mean", "kp_daily_max",
    "kp_1", "kp_2", "kp_3", "kp_4", "kp_5", "kp_6", "kp_7", "kp_8",
    "solar_mean_field",
]

# Columns that are counts → forward-fill (today's count is ~yesterday's)
INTERP_FFILL_COLS = [
    "flare_C", "flare_M", "flare_X", "flare_S",
    "flare_xray_total", "flare_optical_total",
    "new_regions",
    "flare_event_B", "flare_event_C", "flare_event_M", "flare_event_X",
    "flare_event_unknown", "flare_events_total",
]

# Maximum gap (days) we're willing to interpolate across
MAX_LINEAR_GAP = 7
MAX_FFILL_GAP  = 3


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing values using column-appropriate strategies.

    Strategies:
      1. Linear interpolation for smooth physical series (SSN, F10.7,
         Dst, Kp, Ap). Only fills gaps ≤ MAX_LINEAR_GAP days.
      2. Forward-fill for count/event data (flares), gaps ≤ MAX_FFILL_GAP.
      3. Remaining gaps are left as NaN (data truly missing).

    Adds a companion column '{col}_interpolated' (bool) so downstream
    analysis can distinguish measured from imputed values.
    """
    log.info("=" * 60)
    log.info("INTERPOLATION — Filling missing data")
    log.info("=" * 60)

    total_filled = 0

    # --- 1. Linear interpolation for smooth series ---
    for col in INTERP_LINEAR_COLS:
        if col not in df.columns:
            continue

        before_na = df[col].isna().sum()
        if before_na == 0:
            continue

        # Mark what was originally missing
        was_missing = df[col].isna()

        # Linear interpolation with gap limit
        df[col] = df[col].interpolate(method="linear", limit=MAX_LINEAR_GAP, limit_area="inside")

        after_na = df[col].isna().sum()
        n_filled = before_na - after_na

        # Create interpolation flag
        df[f"{col}_interpolated"] = was_missing & df[col].notna()

        if n_filled > 0:
            log.info(f"  [linear]  {col:<25} filled {n_filled:>5} / {before_na:>5} missing "
                     f"(gap ≤ {MAX_LINEAR_GAP}d)")
            total_filled += n_filled

    # --- 2. Forward-fill for count/event data ---
    for col in INTERP_FFILL_COLS:
        if col not in df.columns:
            continue

        before_na = df[col].isna().sum()
        if before_na == 0:
            continue

        was_missing = df[col].isna()

        # Forward-fill with limit, then fill remaining with 0
        # (no flares reported = 0 flares is a reasonable assumption)
        df[col] = df[col].ffill(limit=MAX_FFILL_GAP)

        after_na = df[col].isna().sum()
        n_filled = before_na - after_na

        df[f"{col}_interpolated"] = was_missing & df[col].notna()

        if n_filled > 0:
            log.info(f"  [ffill]   {col:<25} filled {n_filled:>5} / {before_na:>5} missing "
                     f"(gap ≤ {MAX_FFILL_GAP}d)")
            total_filled += n_filled

    # --- 3. Fill remaining flare NaNs with 0 (no report = no flares) ---
    flare_zero_cols = [c for c in INTERP_FFILL_COLS if c in df.columns]
    for col in flare_zero_cols:
        before_na = df[col].isna().sum()
        if before_na > 0:
            was_missing = df[col].isna()
            df[col] = df[col].fillna(0)
            n_filled = before_na
            # Update interpolation flag
            flag_col = f"{col}_interpolated"
            if flag_col in df.columns:
                df[flag_col] = df[flag_col] | was_missing
            else:
                df[flag_col] = was_missing
            log.info(f"  [zero]    {col:<25} filled {n_filled:>5} remaining with 0")
            total_filled += n_filled

    # --- 4. dst_hours_valid — fill with 0 where Dst is missing ---
    if "dst_hours_valid" in df.columns:
        df["dst_hours_valid"] = df["dst_hours_valid"].fillna(0)

    log.info(f"  Total values interpolated/filled: {total_filled}")
    log.info("=" * 60)

    return df


# ---------------------------------------------------------------------------
# Quality checks & derived features
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful derived columns."""
    log.info("[Derived] Adding computed features ...")

    # Year, month, day-of-year for convenience
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear

    # Geomagnetic storm classification based on Dst
    if "dst_daily_min" in df.columns:
        conditions = [
            df["dst_daily_min"] <= -200,
            df["dst_daily_min"] <= -100,
            df["dst_daily_min"] <= -50,
            df["dst_daily_min"] > -50,
        ]
        choices = ["severe", "intense", "moderate", "quiet"]
        df["storm_category"] = np.select(conditions, choices, default="unknown")
        df.loc[df["dst_daily_min"].isna(), "storm_category"] = np.nan

    # Solar activity level based on F10.7
    if "f107" in df.columns:
        conditions = [
            df["f107"] >= 200,
            df["f107"] >= 150,
            df["f107"] >= 100,
            df["f107"] < 100,
        ]
        choices = ["very_high", "high", "moderate", "low"]
        df["solar_activity_level"] = np.select(conditions, choices, default="unknown")
        df.loc[df["f107"].isna(), "solar_activity_level"] = np.nan

    # Kp storm level (NOAA G-scale proxy)
    if "kp_daily_max" in df.columns:
        conditions = [
            df["kp_daily_max"] >= 9,
            df["kp_daily_max"] >= 8,
            df["kp_daily_max"] >= 7,
            df["kp_daily_max"] >= 6,
            df["kp_daily_max"] >= 5,
            df["kp_daily_max"] < 5,
        ]
        choices = ["G5_extreme", "G4_severe", "G3_strong", "G2_moderate", "G1_minor", "quiet"]
        df["kp_storm_level"] = np.select(conditions, choices, default="unknown")
        df.loc[df["kp_daily_max"].isna(), "kp_storm_level"] = np.nan

    return df


def quality_report(df: pd.DataFrame, label: str = "DATA QUALITY REPORT") -> None:
    """Print data quality summary."""
    log.info("=" * 60)
    log.info(label)
    log.info("=" * 60)
    log.info(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    log.info(f"Total days: {len(df)}")
    log.info("")

    # Coverage per column (exclude interpolation flags)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if not c.endswith("_interpolated")]
    log.info(f"{'Column':<25} {'Valid':>8} {'Missing':>8} {'Coverage':>10}")
    log.info("-" * 55)
    for col in numeric_cols:
        valid = df[col].notna().sum()
        missing = df[col].isna().sum()
        pct = valid / len(df) * 100
        log.info(f"{col:<25} {valid:>8} {missing:>8} {pct:>9.1f}%")

    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    """Execute the full clean → harmonise → merge → interpolate pipeline."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Starting data cleaning & harmonisation pipeline")
    log.info("=" * 60)

    # --- Parse each source ---
    ssn = parse_ssn(RAW_DIR / "ssn_daily.txt")

    dsd = parse_dsd(RAW_DIR / "dsd_1996_present.txt")

    dgd = parse_dgd(RAW_DIR / "dgd_1996_present.txt")

    # Dst — combine OMNI2 (1996-2004) + Kyoto (2005+)
    dst_omni = parse_dst_omni2(RAW_DIR)

    dst_kyoto_path = RAW_DIR / "dst_kyoto_hourly.txt"
    if dst_kyoto_path.exists() and dst_kyoto_path.stat().st_size > 1000:
        dst_kyoto = parse_dst_kyoto(dst_kyoto_path)
    else:
        log.warning("[Dst] Kyoto data not available.")
        dst_kyoto = pd.DataFrame()

    # Merge: Kyoto takes priority where both exist
    if not dst_omni.empty and not dst_kyoto.empty:
        dst = pd.concat([dst_omni, dst_kyoto])
        dst = dst[~dst.index.duplicated(keep='last')].sort_index()
        log.info(f"[Dst] Merged OMNI2 ({len(dst_omni)}d) + Kyoto ({len(dst_kyoto)}d) → {len(dst)} unique days")
    elif not dst_kyoto.empty:
        dst = dst_kyoto
    elif not dst_omni.empty:
        dst = dst_omni
    else:
        log.warning("[Dst] No Dst data from any source.")
        dst = pd.DataFrame()

    # F10.7 recent supplement
    f107_path = RAW_DIR / "f107_recent.json"
    f107_json = parse_f107_json(f107_path) if f107_path.exists() else pd.DataFrame()

    # Flare events (legacy 1996-2016)
    flare_path = RAW_DIR / "flares_events_1996_2016.txt"
    flare_events = parse_flare_events(flare_path) if flare_path.exists() else pd.DataFrame()

    # --- Merge ---
    merged = merge_all(ssn, dsd, dgd, dst, f107_json, flare_events)

    # --- Pre-interpolation quality ---
    quality_report(merged, label="QUALITY BEFORE INTERPOLATION")

    # --- Interpolation ---
    merged = interpolate_missing(merged)

    # --- Derived features (after interpolation so classifications use filled data) ---
    merged = add_derived_features(merged)

    # --- Post-interpolation quality ---
    quality_report(merged, label="QUALITY AFTER INTERPOLATION")

    # --- Save ---
    output_path = CLEAN_DIR / "solar_weather_daily.csv"
    merged.to_csv(output_path, float_format="%.2f")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(f"✓ Output saved → {output_path} "
             f"({size_mb:.1f} MB, {len(merged)} rows, {len(merged.columns)} columns)")

    # Column listing
    log.info(f"Columns: {list(merged.columns)}")

    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
