"""
pipeline.py
Data cleaning, resampling, harmonisation, and feature engineering.

Kp source note
--------------
fetch_kp_index() in ingest.py now returns kp_merged_clean.csv, which
combines GFZ historic Kp (1932-present) with the NOAA real-time feed.
The functions here are source-agnostic: they operate on whatever
3-hourly Kp DataFrame is passed in.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_sunspots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate gaps <= 3 days linearly; flag remaining NaN as sn_missing.

    Parameters
    ----------
    df : DataFrame with column 'sn'

    Returns
    -------
    DataFrame with 'sn' cleaned and 'sn_missing' bool column
    """
    df = df.copy()
    df["sn_missing"] = df["sn"].isna()
    df["sn"] = df["sn"].interpolate(method="time", limit=3)
    return df


def clean_kp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip Kp to valid physical range [0, 9] and flag missing values.

    Parameters
    ----------
    df : DataFrame with column 'Kp'

    Returns
    -------
    DataFrame with 'Kp' clipped and 'Kp_missing' bool column
    """
    df = df.copy()
    df["Kp_missing"] = df["Kp"].isna()
    df["Kp"] = df["Kp"].clip(0, 9)
    return df


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_kp_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 3-hourly Kp to daily resolution.

    New columns
    -----------
    Kp_mean        : daily mean Kp
    Kp_max         : daily max Kp
    Kp_storm_hours : count of 3-hour periods with Kp >= 5 (i.e. G1 storm or above)

    Parameters
    ----------
    df : DataFrame indexed by DatetimeIndex with column 'Kp'
    """
    daily = df["Kp"].resample("D").agg(
        Kp_mean="mean",
        Kp_max="max",
    )
    storm = (df["Kp"] >= 5).resample("D").sum().rename("Kp_storm_hours")
    daily = daily.join(storm)
    daily.index.name = "date"
    return daily


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_datasets(sunspots: pd.DataFrame, kp_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join daily sunspot and Kp DataFrames on a shared DatetimeIndex.

    Both datasets now have long histories:
      - Sunspots: 1818-present
      - Kp:       1932-present  (after GFZ historic import)
    The meaningful joint analysis window is 1932-present.

    Parameters
    ----------
    sunspots : DataFrame indexed by 'date'
    kp_daily : DataFrame indexed by 'date'

    Returns
    -------
    Combined DataFrame indexed by 'date'
    """
    combined = sunspots.join(kp_daily, how="outer")
    combined.index.name = "date"
    return combined.sort_index()


# ---------------------------------------------------------------------------
# Smoothing / baseline estimation
# ---------------------------------------------------------------------------

def add_smoothed_columns(
    df: pd.DataFrame,
    windows: tuple[int, int] = (27, 365),
) -> pd.DataFrame:
    """
    Add rolling mean and Savitzky-Golay cycle-baseline columns to df.

    New columns
    -----------
    sn_roll27d         : 27-day centred rolling mean of 'sn'
    sn_roll365d        : 365-day centred rolling mean of 'sn'
    sn_cycle_baseline  : Savitzky-Golay smoothed cycle baseline (requires >= 4 yrs data)

    Parameters
    ----------
    df      : DataFrame with 'sn' column, DatetimeIndex at daily frequency
    windows : tuple of rolling window sizes in days
    """
    df = df.copy()

    for w in windows:
        col = f"sn_roll{w}d"
        df[col] = (
            df["sn"]
            .rolling(window=w, center=True, min_periods=w // 2)
            .mean()
        )

    # Savitzky-Golay — needs >= 4 years of non-NaN data
    sn_vals = df["sn"].values.astype(float)
    n = np.sum(~np.isnan(sn_vals))
    if n >= 4 * 365:
        filled = pd.Series(sn_vals).interpolate(method="linear", limit_direction="both").values
        sg = savgol_filter(filled, window_length=1461, polyorder=3)
        df["sn_cycle_baseline"] = sg
    else:
        df["sn_cycle_baseline"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

def add_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling 365-day z-scores for 'sn' and 'Kp_max'.

    New columns
    -----------
    sn_zscore     : rolling z-score of 'sn'
    Kp_max_zscore : rolling z-score of 'Kp_max'
    """
    df = df.copy()

    for col, zcol in [("sn", "sn_zscore"), ("Kp_max", "Kp_max_zscore")]:
        if col not in df.columns:
            df[zcol] = np.nan
            continue
        roll = df[col].rolling(window=365, center=True, min_periods=182)
        mu = roll.mean()
        sigma = roll.std()
        df[zcol] = (df[col] - mu) / sigma.replace(0, np.nan)

    return df
