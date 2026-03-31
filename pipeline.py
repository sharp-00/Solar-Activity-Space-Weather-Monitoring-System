"""
pipeline.py
Data cleaning, resampling, harmonisation, and feature engineering.

Kp, Flare, and DST source note
------------------------------
The ingest module now provides a mix of deep historical flat files 
(Sunspots, Kp) and dynamically updated 7-day rolling continuous feeds 
(Solar Flare flux, Geospace DST). 

This pipeline normalizes all frequencies to a shared Daily resolution 
for downstream modeling and analysis.
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
    """
    df = df.copy()
    df["sn_missing"] = df["sn"].isna()
    df["sn"] = df["sn"].interpolate(method="time", limit=3)
    return df


def clean_kp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip Kp to valid physical range [0, 9] and flag missing values.
    """
    df = df.copy()
    df["Kp_missing"] = df["Kp"].isna()
    df["Kp"] = df["Kp"].clip(0, 9)
    return df


def clean_flares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolate the primary GOES X-ray band (0.1-0.8nm), drop invalid flux, 
    and flag missing.
    """
    df = df.copy()
    if "energy_band" in df.columns:
        # NOAA reports 0.05-0.4nm and 0.1-0.8nm; the latter dictates Flare Class
        df = df[df["energy_band"] == "0.1-0.8nm"]
    
    df["flux_missing"] = df["xray_flux"].isna()
    # Clip extreme anomalies or baseline sensor zeroes
    df["xray_flux"] = df["xray_flux"].clip(lower=1e-10) 
    return df


def clean_dst(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag missing DST values. Valid DST is typically between -2000 and +100 nT.
    """
    df = df.copy()
    df["dst_missing"] = df["dst"].isna()
    return df


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_kp_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 3-hourly Kp to daily resolution.
    New columns: Kp_mean, Kp_max, Kp_storm_hours (Kp >= 5)
    """
    daily = df["Kp"].resample("D").agg(
        Kp_mean="mean",
        Kp_max="max",
    )
    storm = (df["Kp"] >= 5).resample("D").sum().rename("Kp_storm_hours")
    daily = daily.join(storm)
    daily.index.name = "date"
    return daily


def resample_flares_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute GOES X-ray flux to daily resolution.
    New columns: flare_max_flux, flare_m_class_mins, flare_x_class_mins
    """
    # Max flux dictates the official daily flare classification
    daily = df.resample("D").agg(
        flare_max_flux=("xray_flux", "max")
    )
    
    # Count minutes spent above critical thresholds
    # M-Class: >= 1e-5 W/m^2
    # X-Class: >= 1e-4 W/m^2
    m_mins = (df["xray_flux"] >= 1e-5).resample("D").sum().rename("flare_m_class_mins")
    x_mins = (df["xray_flux"] >= 1e-4).resample("D").sum().rename("flare_x_class_mins")
    
    daily = daily.join(m_mins).join(x_mins)
    daily.index.name = "date"
    return daily


def resample_dst_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Geospace DST to daily resolution.
    New columns: dst_min, dst_mean
    """
    # For DST, the daily minimum is the most critical metric for geomagnetic storms
    daily = df.resample("D").agg(
        dst_min=("dst", "min"),
        dst_mean=("dst", "mean")
    )
    daily.index.name = "date"
    return daily


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_datasets(
    sunspots: pd.DataFrame, 
    kp_daily: pd.DataFrame, 
    flare_daily: pd.DataFrame = None, 
    dst_daily: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Outer-join all daily DataFrames on a shared DatetimeIndex.
    
    Returns
    -------
    Combined DataFrame indexed by 'date'. Missing historical data for 
    newer endpoints (Flares, DST) will safely resolve to NaN.
    """
    combined = sunspots.join(kp_daily, how="outer")
    
    if flare_daily is not None and not flare_daily.empty:
        combined = combined.join(flare_daily, how="outer")
        
    if dst_daily is not None and not dst_daily.empty:
        combined = combined.join(dst_daily, how="outer")
        
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
    Compute rolling 365-day z-scores for primary space weather indicators.
    """
    df = df.copy()

    # Track anomalies for Sunspots, Kp Max, and DST Min (if available)
    targets = [("sn", "sn_zscore"), ("Kp_max", "Kp_max_zscore"), ("dst_min", "dst_min_zscore")]

    for col, zcol in targets:
        if col not in df.columns:
            df[zcol] = np.nan
            continue
        roll = df[col].rolling(window=365, center=True, min_periods=182)
        mu = roll.mean()
        sigma = roll.std()
        df[zcol] = (df[col] - mu) / sigma.replace(0, np.nan)

    return df
