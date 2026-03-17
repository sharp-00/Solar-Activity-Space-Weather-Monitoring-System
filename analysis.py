"""
analysis.py
Correlation analysis, extreme event detection, cycle phase, and monthly statistics.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def cross_correlation(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 30,
) -> pd.DataFrame:
    """
    Compute Pearson r between x and y at integer lags from -max_lag to +max_lag.

    Convention: positive lag means x leads y (x is shifted forward in time).

    Parameters
    ----------
    x, y     : aligned Series (NaN rows dropped pairwise per lag)
    max_lag  : maximum absolute lag in days

    Returns
    -------
    DataFrame with columns: lag_days, pearson_r, n
    """
    records = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs = x.iloc[lag:].values
            ys = y.iloc[: len(x) - lag].values
        else:
            xs = x.iloc[: len(x) + lag].values
            ys = y.iloc[-lag:].values

        # Pairwise valid
        mask = ~np.isnan(xs) & ~np.isnan(ys)
        n = mask.sum()
        if n < 10:
            records.append({"lag_days": lag, "pearson_r": np.nan, "n": n})
            continue
        r, _ = pearsonr(xs[mask], ys[mask])
        records.append({"lag_days": lag, "pearson_r": r, "n": n})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Extreme events
# ---------------------------------------------------------------------------

def find_extreme_events(
    df: pd.DataFrame,
    col: str,
    threshold_sigma: float = 2.5,
) -> pd.DataFrame:
    """
    Return rows where col > mean + threshold_sigma * std.

    Parameters
    ----------
    df              : DataFrame containing col
    col             : column name to analyse
    threshold_sigma : z-score threshold

    Returns
    -------
    DataFrame with original col and 'z_score', sorted descending by z_score
    """
    series = df[col].dropna()
    mu = series.mean()
    sigma = series.std()
    z = (df[col] - mu) / sigma
    mask = z > threshold_sigma
    result = df.loc[mask, [col]].copy()
    result["z_score"] = z[mask]
    return result.sort_values("z_score", ascending=False)


# ---------------------------------------------------------------------------
# Solar cycle phase
# ---------------------------------------------------------------------------

def solar_cycle_phase(
    date_index: pd.DatetimeIndex,
    cycle_start: str = "1996-08-01",
    cycle_length_years: float = 11.0,
) -> pd.Series:
    """
    Compute fractional phase within a solar cycle [0, 1).

    Parameters
    ----------
    date_index         : DatetimeIndex of dates
    cycle_start        : ISO date string for cycle minimum
    cycle_length_years : nominal cycle length

    Returns
    -------
    Series of phase values indexed by date_index
    """
    t0 = pd.Timestamp(cycle_start)
    cycle_days = cycle_length_years * 365.25
    elapsed = (date_index - t0).total_seconds() / 86400.0
    phase = (elapsed % cycle_days) / cycle_days
    return pd.Series(phase, index=date_index, name="cycle_phase")


# ---------------------------------------------------------------------------
# Monthly statistics
# ---------------------------------------------------------------------------

def compute_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a daily DataFrame to monthly frequency.

    Output columns
    --------------
    sn_mean, sn_max, Kp_mean, Kp_max, storm_hours
    """
    agg: dict[str, tuple] = {}

    if "sn" in df.columns:
        agg["sn_mean"] = ("sn", "mean")
        agg["sn_max"] = ("sn", "max")

    if "Kp_mean" in df.columns:
        agg["Kp_mean"] = ("Kp_mean", "mean")

    if "Kp_max" in df.columns:
        agg["Kp_max"] = ("Kp_max", "max")

    if "Kp_storm_hours" in df.columns:
        agg["storm_hours"] = ("Kp_storm_hours", "sum")

    if not agg:
        return pd.DataFrame()

    monthly = df.resample("ME").agg(**agg)
    monthly.index.name = "date"
    return monthly
