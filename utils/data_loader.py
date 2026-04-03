"""utils/data_loader.py — Cached loaders for all parquet datasets + pipeline runner."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

CLEAN_DIR = Path("data/clean")
RAW_DIR   = Path("data/raw")
STATS_DIR = Path("data/analysis/stats")


# ── Generic reader ────────────────────────────────────────────────────────────

def _read(stem: str, **pd_kwargs) -> pd.DataFrame:
    """Try parquet first, then CSV in data/clean/, then data/ root (legacy)."""
    for candidate in [
        CLEAN_DIR / f"{stem}.parquet",
        CLEAN_DIR / f"{stem}.csv",
        Path("data") / f"{stem}.csv",
    ]:
        if candidate.exists():
            df = (pd.read_parquet(candidate)
                  if candidate.suffix == ".parquet"
                  else pd.read_csv(candidate, **pd_kwargs))
            if candidate.suffix != ".parquet":
                try:
                    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(CLEAN_DIR / f"{stem}.parquet")
                except Exception:
                    pass
            return df
    return pd.DataFrame()


# ── Cached dataset loaders ────────────────────────────────────────────────────

@st.cache_data(ttl=900)
def load_main_data() -> pd.DataFrame:
    """Master merged dataset — all variables, daily resolution."""
    df = _read("solar_weather_daily", parse_dates=["date"], index_col="date")
    if not df.empty and "year" not in df.columns:
        df["year"] = df.index.year
    return df


@st.cache_data(ttl=900)
def load_sunspots_data() -> pd.DataFrame:
    """SILSO daily sunspot number (sn, sn_err).
    Falls back to the master merged file if no dedicated sunspot file exists."""
    for stem in ["sunspots_daily_clean", "sunspots_daily"]:
        for candidate in [CLEAN_DIR / f"{stem}.parquet",
                          CLEAN_DIR / f"{stem}.csv",
                          Path("data") / f"{stem}.csv"]:
            if candidate.exists():
                df = (pd.read_parquet(candidate)
                      if candidate.suffix == ".parquet"
                      else pd.read_csv(candidate, parse_dates=["date"])
                           .set_index("date").sort_index())
                if candidate.suffix != ".parquet":
                    try:
                        df.to_parquet(CLEAN_DIR / "sunspots_daily_clean.parquet")
                    except Exception:
                        pass
                return df

    # Fallback: extract from master merged dataset
    main = load_main_data()
    if not main.empty:
        if "ssn" in main.columns:
            cols = {"ssn": "sn"}
            if "ssn_std" in main.columns:
                cols["ssn_std"] = "sn_err"
            df = main[list(cols.keys())].rename(columns=cols)
            try:
                CLEAN_DIR.mkdir(parents=True, exist_ok=True)
                df.to_parquet(CLEAN_DIR / "sunspots_daily_clean.parquet")
            except Exception:
                pass
            return df
        if "sn" in main.columns:
            return main[["sn"] + (["sn_err"] if "sn_err" in main.columns else [])]
    return pd.DataFrame()


@st.cache_data(ttl=900)
def load_kp_data() -> pd.DataFrame:
    """Daily Kp/ap index. Checks all known sources in priority order."""
    df = _read("kp_daily_clean", parse_dates=["date"], index_col="date")
    if not df.empty:
        return df

    # Fallback 1: legacy kp_merged_clean.csv from old download_data.py
    for legacy in [CLEAN_DIR / "kp_merged_clean.csv",
                   Path("data") / "kp_merged_clean.csv"]:
        if legacy.exists():
            try:
                tmp = pd.read_csv(legacy, parse_dates=["time_tag"]).set_index("time_tag")
                tmp.index = tmp.index.normalize()          # 3-hourly → daily
                tmp.index.name = "date"
                if "Kp" in tmp.columns:
                    daily = tmp["Kp"].resample("D").agg(
                        kp_daily_mean="mean", kp_daily_max="max").dropna(how="all")
                    if not daily.empty:
                        return daily
            except Exception:
                pass

    # Fallback 2: pull kp columns from main dataset
    main = load_main_data()
    kp_cols = [c for c in main.columns if "kp" in c.lower() and main[c].notna().any()]
    return main[kp_cols] if kp_cols else pd.DataFrame()


@st.cache_data(ttl=900)
def load_dst_data() -> pd.DataFrame:
    """Daily Dst index (OMNI2 + Kyoto WDC)."""
    df = _read("dst_daily_clean", parse_dates=["date"], index_col="date")
    if df.empty:
        main = load_main_data()
        dst_cols = [c for c in main.columns if "dst" in c.lower()]
        return main[dst_cols] if dst_cols else pd.DataFrame()
    return df


@st.cache_data(ttl=900)
def load_f107_data() -> pd.DataFrame:
    """Daily F10.7 solar flux (DSD + SWPC JSON + GFZ)."""
    df = _read("f107_daily_clean", parse_dates=["date"], index_col="date")
    if df.empty:
        main = load_main_data()
        cols = [c for c in main.columns if "f107" in c.lower()]
        return main[cols] if cols else pd.DataFrame()
    return df


@st.cache_data(ttl=900)
def load_flares_data() -> pd.DataFrame:
    """Daily solar flare counts (DSD + legacy XRS events)."""
    df = _read("flares_daily_clean", parse_dates=["date"], index_col="date")
    if df.empty:
        main = load_main_data()
        cols = [c for c in main.columns if "flare" in c.lower()]
        return main[cols] if cols else pd.DataFrame()
    return df


@st.cache_data(ttl=900)
def load_stats_file(filename: str) -> pd.DataFrame:
    stem = filename.replace(".csv", "")
    for ext in (".parquet", ".csv"):
        p = STATS_DIR / f"{stem}{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    return pd.DataFrame()


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _raw_data_exists() -> bool:
    key_files = ["ssn_daily.txt", "dsd_1996_present.txt", "dgd_1996_present.txt"]
    return any((RAW_DIR / f).exists() for f in key_files)

def _parquets_exist() -> bool:
    """True if the main output parquet has already been built by the new pipeline."""
    return (CLEAN_DIR / "solar_weather_daily.parquet").exists()


def _bust_caches():
    for fn in [load_main_data, load_sunspots_data, load_kp_data,
               load_dst_data, load_f107_data, load_flares_data, load_stats_file]:
        fn.clear()


def _run_script(script: str, label: str, args: list,
                status_writer=None, root: Path = None) -> None:
    if root is None:
        root = Path(__file__).parent.parent
    if status_writer:
        status_writer(label)
    result = subprocess.run(
        [sys.executable, str(root / script)] + args,
        capture_output=True, text=True, cwd=str(root),
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{result.stderr[-3000:]}")


def run_pipeline(status_writer=None) -> None:
    """Smart pipeline: incremental if raw data exists, full ingest otherwise."""
    root = Path(__file__).parent.parent
    if _raw_data_exists():
        _run_script("ingest.py", "📥 Fetching latest data…",
                    ["--latest"], status_writer, root)
        if _parquets_exist():
            # Parquets exist → merge new raw data in (fast)
            _run_script("clean.py", "🧹 Merging new data into parquets…",
                        ["--update"], status_writer, root)
        else:
            # First time running new pipeline → full rebuild from all raw files
            _run_script("clean.py", "🧹 Building parquet files from raw data…",
                        [], status_writer, root)
    else:
        _run_script("ingest.py", "📥 Full data download (first time, ~20 min)…",
                    [], status_writer, root)
        _run_script("clean.py", "🧹 Building parquet files…",
                    [], status_writer, root)
    _bust_caches()
