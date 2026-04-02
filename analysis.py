"""
analysis.py — Core Statistical & Analytical Engine
====================================================
Provides reusable, UI-agnostic functions for solar-terrestrial analysis.

This module is consumed by ``dashboard.py`` for interactive visualisation
and can be used independently for batch processing or notebook exploration.

Functions
---------
cross_correlation        : Pearson r at integer daily lags.
find_extreme_events      : Z-score outlier detection.
solar_cycle_phase        : Fractional phase within the 11-year cycle.
compute_periodly_stats   : Resample daily data to W / 2W / M / Q / Y.
compute_monthly_stats    : Convenience wrapper for monthly resampling.
duty_cycle               : Data-reliability metric (actual vs expected obs).
analyze_periodicity      : FFT or Continuous Wavelet spectral analysis.
predictive_dominance     : Compare predictive power of solar drivers for Kp.
analyze_phase_locked_climatology : Bin Kp by solar-cycle phase.
analyze_hysteresis       : Rising vs falling phase scatter analysis.

Dependencies
------------
numpy, pandas, scipy, pywt, matplotlib
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import pywt

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def cross_correlation(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 30,
    min_obs: int = 10,
) -> pd.DataFrame:
    """Compute Pearson *r* between *x* and *y* at integer lags.

    Convention
    ----------
    A **positive lag** means *x* leads *y* — i.e. a solar driver (*x*)
    measured today is correlated with a geomagnetic response (*y*)
    ``lag`` days in the future.

    Parameters
    ----------
    x : pd.Series
        Driver time-series (e.g. SSN, F10.7).
    y : pd.Series
        Response time-series (e.g. Kp, Dst).
    max_lag : int, default 30
        Maximum absolute lag in days.  Automatically clamped to
        ``len(x) - min_obs`` so that the function never attempts to
        correlate arrays shorter than *min_obs*.
    min_obs : int, default 10
        Minimum number of pairwise-valid observations required to
        compute a correlation; otherwise ``NaN`` is recorded.

    Returns
    -------
    pd.DataFrame
        Columns: ``lag_days``, ``pearson_r``, ``n``.
    """
    n_total = min(len(x), len(y))
    # Clamp max_lag so sliced arrays are never empty
    safe_max_lag = min(max_lag, max(0, n_total - min_obs))

    records = []
    for lag in range(-safe_max_lag, safe_max_lag + 1):
        if lag >= 0:
            xs = x.iloc[lag:].values
            ys = y.iloc[: len(x) - lag].values
        else:
            xs = x.iloc[: len(x) + lag].values
            ys = y.iloc[-lag:].values

        # Pairwise valid
        mask = ~np.isnan(xs) & ~np.isnan(ys)
        n = int(mask.sum())
        if n < min_obs:
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
    """Detect statistical outlier days using Z-score analysis.

    An observation is flagged as *extreme* when its Z-score exceeds
    ``threshold_sigma`` standard deviations above the mean.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``col``.
    col : str
        Column name to analyse (e.g. ``'kp_daily_max'``).
    threshold_sigma : float, default 2.5
        Z-score cutoff.  Lower values flag more events.

    Returns
    -------
    pd.DataFrame
        Subset of ``df`` containing only extreme rows, with an added
        ``z_score`` column, sorted descending by Z-score.
    """
    series = df[col].dropna()
    mu = series.mean()
    sigma = series.std()
    if sigma == 0:
        return pd.DataFrame(columns=[col, "z_score"])
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
    """Compute the fractional phase within a nominal solar cycle.

    Phase values range from **0** (cycle minimum / start) to just
    below **1** (end of cycle, about to wrap).  A value of ~0.4
    conventionally marks the transition from *rising* to *falling*.

    Parameters
    ----------
    date_index : pd.DatetimeIndex
        Dates to compute the phase for.
    cycle_start : str, default '1996-08-01'
        ISO date string for a known cycle minimum (reference epoch).
    cycle_length_years : float, default 11.0
        Nominal length of one solar cycle in years.

    Returns
    -------
    pd.Series
        Phase values indexed by *date_index*, named ``'cycle_phase'``.
    """
    t0 = pd.Timestamp(cycle_start)
    cycle_days = cycle_length_years * 365.25
    elapsed = (date_index - t0).total_seconds() / 86400.0
    phase = (elapsed % cycle_days) / cycle_days
    return pd.Series(phase, index=date_index, name="cycle_phase")


# ---------------------------------------------------------------------------
# Periodic statistics (resampling)
# ---------------------------------------------------------------------------

def compute_periodly_stats(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Resample daily solar-weather data to a coarser temporal resolution.

    Computes mean and max aggregates for Sunspot Number (SSN) and
    Kp index columns, if present in the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Daily-resolution DataFrame with a ``DatetimeIndex``.
    period : str
        Pandas offset alias: ``'W'`` (weekly), ``'2W'`` (bi-weekly),
        ``'M'`` (monthly), ``'Q'`` (quarterly), or ``'Y'`` (yearly).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with columns such as ``ssn_mean``,
        ``ssn_max``, ``kp_daily_mean``, ``kp_daily_max``.

    Raises
    ------
    ValueError
        If *period* is not one of the accepted aliases.
    """
    valid_periods = ['W', '2W', 'M', 'Q', 'Y']
    if period.upper().replace('E', '') not in valid_periods:
        raise ValueError(f"Period must be one of {valid_periods}, got '{period}'")

    agg = {}
    if "ssn" in df.columns:
        agg.update({"ssn_mean": ("ssn", "mean"), "ssn_max": ("ssn", "max")})
    if "kp_daily_mean" in df.columns:
        agg["kp_daily_mean"] = ("kp_daily_mean", "mean")
    if "kp_daily_max" in df.columns:
        agg["kp_daily_max"] = ("kp_daily_max", "max")

    result = df.resample(period + "E").agg(**agg)
    result.index.name = "date"
    return result


def compute_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: resample daily data to monthly aggregates.

    Equivalent to ``compute_periodly_stats(df, 'M')``.

    Parameters
    ----------
    df : pd.DataFrame
        Daily-resolution DataFrame with a ``DatetimeIndex``.

    Returns
    -------
    pd.DataFrame
        Monthly-resampled statistics.
    """
    return compute_periodly_stats(df, "M")


# ---------------------------------------------------------------------------
# Duty Cycle (data reliability)
# ---------------------------------------------------------------------------

def duty_cycle(df: pd.DataFrame, col: str, window: str) -> pd.DataFrame:
    """Quantify data reliability by comparing actual vs expected observations.

    Measures the fraction of non-null values within fixed time windows,
    allowing downstream analysis to assess whether a particular epoch
    has sufficient coverage for statistical conclusions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` and at least the column *col*.
    col : str
        Column to evaluate (e.g. ``'ssn'``, ``'dst_daily_mean'``).
    window : str
        Resampling window — one of:

        - ``'11YE'`` — Full solar cycle (~4018 days expected).
        - ``'10YE'`` — Decade.
        - ``'YE'``   — Annual.
        - ``'27D'``  — One solar rotation (Carrington rotation).

    Returns
    -------
    pd.DataFrame
        Columns: ``expected_count``, ``actual_count``, ``duty_cycle``
        (ratio), and ``is_reliable`` (boolean flag based on a
        per-window threshold).

    Raises
    ------
    ValueError
        If *window* is not one of the accepted strings.
    """
    config = {"11YE": 0.95, "10YE": 0.95, "YE": 0.90, "27D": 0.99}

    if window not in config:
        raise ValueError(f"Invalid window '{window}'. Use: {list(config.keys())}")

    grouped = df[col].resample(window)
    result = pd.DataFrame({
        "expected_count": grouped.size(),
        "actual_count": grouped.count()
    })

    result["duty_cycle"] = result["actual_count"] / result["expected_count"]
    result["is_reliable"] = result["duty_cycle"] >= config[window]
    return result


# ---------------------------------------------------------------------------
# Periodicity analysis (FFT / Wavelet)
# ---------------------------------------------------------------------------

def analyze_periodicity(
    series: pd.Series,
    method: str = 'fft',
    fs: float = 1.0,
) -> pd.DataFrame:
    """Identify dominant periodic cycles in a time-series.

    Supports two complementary spectral methods:

    - **FFT** (Fast Fourier Transform): reveals globally dominant
      frequencies across the entire record.
    - **Wavelet** (Continuous Wavelet Transform with Morlet kernel):
      captures how the dominant period evolves over time (e.g.
      whether the 11-year cycle was shorter in the 1990s).

    Parameters
    ----------
    series : pd.Series
        Input time-series (daily resolution assumed).  NaN values are
        dropped before analysis.
    method : str, default 'fft'
        ``'fft'`` for global spectral analysis, ``'wv'`` for temporal
        wavelet analysis.
    fs : float, default 1.0
        Sampling frequency in samples per day.

    Returns
    -------
    pd.DataFrame
        For ``'fft'``: columns ``frequency``, ``period_years``,
        ``amplitude`` — sorted by amplitude descending.

        For ``'wv'``: columns ``dominant_period_yrs``, ``max_power``
        — indexed by the original date index.

    Raises
    ------
    ValueError
        If *method* is not ``'fft'`` or ``'wv'``.
    """
    clean_series = series.dropna()
    data = clean_series.values
    data = data - np.mean(data)  # Mean-detrend
    n = len(data)

    if method == 'fft':
        fft_values = np.fft.fft(data)
        freqs = np.fft.fftfreq(n, d=1 / fs)
        idx = np.where(freqs > 0)[0]

        results = pd.DataFrame({
            "frequency": freqs[idx],
            "period_years": (1 / freqs[idx]) / 365.25,
            "amplitude": np.abs(fft_values)[idx]
        })
        return results.sort_values("amplitude", ascending=False).reset_index(drop=True)

    elif method == 'wv':
        scales = np.arange(365, 365 * 20, 15)  # 1 to 20 year cycles
        coeffs, freqs = pywt.cwt(data, scales, 'cmor1.5-1.0', sampling_period=1 / fs)
        power_matrix = np.abs(coeffs) ** 2
        periods_yrs = (1 / freqs) / 365.25

        peak_indices = np.argmax(power_matrix, axis=0)
        return pd.DataFrame({
            "dominant_period_yrs": periods_yrs[peak_indices],
            "max_power": np.max(power_matrix, axis=0)
        }, index=clean_series.index)

    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'fft' or 'wv'.")


# ---------------------------------------------------------------------------
# Predictive Dominance
# ---------------------------------------------------------------------------

def predictive_dominance(
    df: pd.DataFrame,
    max_lag: int = 14,
    ssn_col: str = "ssn",
    flare_col: str = "flare_xray_total",
    kp_col: str = "kp_daily_mean",
) -> pd.DataFrame:
    """Compare the predictive power of Sunspots vs Flares for Kp.

    Computes lagged cross-correlations of both solar drivers against
    the Kp index and reports which one achieves the stronger peak
    correlation and at what lag.

    Parameters
    ----------
    df : pd.DataFrame
        Daily-resolution DataFrame with the required columns.
    max_lag : int, default 14
        Maximum lag in days to search.
    ssn_col : str, default 'ssn'
        Column name for sunspot number.
    flare_col : str, default 'flare_xray_total'
        Column name for total X-ray flare count.
    kp_col : str, default 'kp_daily_mean'
        Column name for the Kp index.

    Returns
    -------
    pd.DataFrame
        Comparison table with rows for Max Correlation and Optimal Lag,
        and columns for Sunspots and Flares.
    """
    sn_corr = cross_correlation(df[ssn_col], df[kp_col], max_lag)
    flare_corr = cross_correlation(df[flare_col], df[kp_col], max_lag)

    max_r_sn = sn_corr['pearson_r'].max()
    max_r_flare = flare_corr['pearson_r'].max()

    best_driver = "Sunspots" if max_r_sn > max_r_flare else "Flares"

    comparison = pd.DataFrame({
        "Metric": ["Max Correlation (r)", "Optimal Lag (Days)"],
        "Sunspots": [max_r_sn, sn_corr.loc[sn_corr['pearson_r'].idxmax(), 'lag_days']],
        "Flares": [max_r_flare, flare_corr.loc[flare_corr['pearson_r'].idxmax(), 'lag_days']]
    })

    log.info("Dominant Driver: %s", best_driver)
    return comparison


# ---------------------------------------------------------------------------
# Phase-Locked Climatology
# ---------------------------------------------------------------------------

def analyze_phase_locked_climatology(
    df: pd.DataFrame,
    kp_col: str = "kp_daily_mean",
    storm_threshold: float = 6.0,
    cycle_start: str = "1996-08-01",
    cycle_length_years: float = 11.0,
    bins: int = 20,
) -> pd.DataFrame:
    """Bin Kp observations by solar-cycle phase to find storm-prone zones.

    Divides the 11-year cycle into ``bins`` equal-width phase intervals
    and computes per-bin statistics (mean, max, storm probability).
    This reveals whether certain phases are inherently stormier than
    others, independent of absolute SSN magnitude.

    Parameters
    ----------
    df : pd.DataFrame
        Daily-resolution DataFrame with a ``DatetimeIndex``.
    kp_col : str, default 'kp_daily_mean'
        Name of the Kp column to analyse.
    storm_threshold : float, default 6.0
        Kp value above which an observation counts as a storm.
    cycle_start : str, default '1996-08-01'
        ISO date for the reference cycle minimum.
    cycle_length_years : float, default 11.0
        Nominal solar-cycle length.
    bins : int, default 20
        Number of phase bins (20 → 5 % intervals).

    Returns
    -------
    pd.DataFrame
        Indexed by ``phase_bin`` (float 0 → 1), with columns
        ``kp_mean``, ``kp_max``, ``n_days``, ``storm_prob_pct``.
    """
    t0 = pd.Timestamp(cycle_start)
    cycle_days = cycle_length_years * 365.25

    elapsed_days = (df.index - t0).total_seconds() / 86400.0
    phase = (elapsed_days % cycle_days) / cycle_days

    df_temp = df.copy()
    df_temp['cycle_phase'] = phase

    bin_edges = np.linspace(0, 1, bins + 1)
    df_temp['phase_bin'] = pd.cut(
        df_temp['cycle_phase'], bins=bin_edges,
        include_lowest=True, labels=bin_edges[:-1]
    )

    grouped = df_temp.groupby('phase_bin', observed=True)

    climatology = pd.DataFrame({
        "kp_mean": grouped[kp_col].mean(),
        "kp_max": grouped[kp_col].max(),
        "n_days": grouped[kp_col].count()
    })

    storm_counts = grouped[kp_col].apply(lambda x: (x >= storm_threshold).sum())
    climatology["storm_prob_pct"] = (storm_counts / climatology["n_days"]) * 100

    return climatology.sort_index()


# ---------------------------------------------------------------------------
# Hysteresis Analysis
# ---------------------------------------------------------------------------

def analyze_hysteresis(
    df: pd.DataFrame,
    sn_col: str = "ssn",
    kp_col: str = "kp_daily_mean",
    cycle_start: str = "1996-08-01",
    cycle_length_years: float = 11.0,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Classify daily observations as *rising* or *falling* solar phase.

    Optionally saves a matplotlib scatter plot to *save_path* for
    standalone use.  The interactive dashboard ignores *save_path*
    and renders its own Plotly visualisation from the returned DataFrame.

    A phase value < 0.4 is classified as **Rising** (the Sun is
    building toward maximum), and ≥ 0.4 as **Falling** (declining
    toward the next minimum).

    Parameters
    ----------
    df : pd.DataFrame
        Daily-resolution DataFrame with a ``DatetimeIndex``.
    sn_col : str, default 'ssn'
        Column for sunspot number (x-axis of the scatter).
    kp_col : str, default 'kp_daily_mean'
        Column for Kp index (y-axis of the scatter).
    cycle_start : str, default '1996-08-01'
        Reference cycle minimum date.
    cycle_length_years : float, default 11.0
        Nominal cycle length.
    save_path : str or None, default None
        If provided, save a static matplotlib PNG to this path.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with added columns ``cycle_phase`` (float 0–1)
        and ``phase_type`` (``'Rising'`` or ``'Falling'``).
    """
    t0 = pd.Timestamp(cycle_start)
    cycle_days = cycle_length_years * 365.25
    elapsed_days = (df.index - t0).total_seconds() / 86400.0
    phase = (elapsed_days % cycle_days) / cycle_days

    df_plot = df.copy()
    df_plot['cycle_phase'] = phase
    df_plot['phase_type'] = np.where(
        df_plot['cycle_phase'] < 0.4, 'Rising', 'Falling'
    )

    # Optional static plot for non-dashboard use
    if save_path is not None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        for p_type, color in zip(['Falling', 'Rising'], ['#d62728', '#1f77b4']):
            subset = df_plot[df_plot['phase_type'] == p_type]
            plt.scatter(
                subset[sn_col], subset[kp_col],
                alpha=0.3, label=f"{p_type} Phase",
                c=color, edgecolors='none', s=10,
            )
        plt.title(f"Solar Hysteresis: {sn_col} vs {kp_col}")
        plt.xlabel("Sunspot Number")
        plt.ylabel("Geomagnetic Index (Kp)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return df_plot
