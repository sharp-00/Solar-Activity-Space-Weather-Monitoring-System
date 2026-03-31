"""
ingest.py
Load clean CSVs from data/ into typed DataFrames.
All functions raise FileNotFoundError with instructions if a file is missing.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_MISSING_MSG = "Run: python3 download_data.py"


def _require(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}\n{_MISSING_MSG}")
    return path


def fetch_sunspots(freq: str = "daily") -> pd.DataFrame:
    """
    Load sunspot data.

    Parameters
    ----------
    freq : 'daily' | 'monthly'

    Returns
    -------
    DataFrame indexed by 'date' (DatetimeIndex) with columns:
      sn, sn_err, n_obs, definitive
    """
    fname = f"sunspots_{freq}_clean.csv"
    path = _require(os.path.join(DATA_DIR, fname))
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.index.name = "date"
    return df


def fetch_kp_index() -> pd.DataFrame:
    """
    Load the merged Kp index (GFZ historic 1932-present + NOAA recent).

    Falls back to the legacy kp_recent_clean.csv if the merged file does
    not yet exist.

    Returns
    -------
    DataFrame indexed by 'time_tag' (DatetimeIndex) with column: Kp
    """
    merged_path = os.path.join(DATA_DIR, "kp_merged_clean.csv")
    legacy_path = os.path.join(DATA_DIR, "kp_recent_clean.csv")

    if os.path.exists(merged_path):
        df = pd.read_csv(merged_path, index_col="time_tag", parse_dates=True)
    elif os.path.exists(legacy_path):
        print("[ingest] kp_merged_clean.csv not found, using legacy kp_recent_clean.csv. "
              "Run: python3 download_data.py to get the full 1932-present archive.")
        df = pd.read_csv(legacy_path, index_col="time_tag", parse_dates=True)
    else:
        raise FileNotFoundError(
            f"No Kp file found in {DATA_DIR}\n{_MISSING_MSG}"
        )

    df.index.name = "time_tag"
    return df


def load_historic_kp() -> pd.DataFrame | None:
    """
    Load historic Kp flat file (GFZ) if available; returns None gracefully.
    This is provided for direct access to the raw GFZ file if needed.
    """
    path = os.path.join(DATA_DIR, "kp_historic_gfz.txt")
    if not os.path.exists(path):
        return None
    try:
        from download_data import _parse_gfz_kp_historic
        return _parse_gfz_kp_historic(path)
    except Exception:
        return None


def fetch_flares() -> pd.DataFrame:
    """
    Load the continuous GOES X-ray flux (Solar Flare) dataset.
    
    Returns
    -------
    DataFrame indexed by 'time_tag' (DatetimeIndex) with columns:
      energy_band, xray_flux
    """
    path = _require(os.path.join(DATA_DIR, "solar_flare_clean.csv"))
    df = pd.read_csv(path, index_col="time_tag", parse_dates=True)
    df.index.name = "time_tag"
    return df


def fetch_dst() -> pd.DataFrame:
    """
    Load the continuous Geospace DST dataset.
    
    Returns
    -------
    DataFrame indexed by 'time_tag' (DatetimeIndex) with column:
      dst
    """
    path = _require(os.path.join(DATA_DIR, "dst_clean.csv"))
    df = pd.read_csv(path, index_col="time_tag", parse_dates=True)
    df.index.name = "time_tag"
    return df
