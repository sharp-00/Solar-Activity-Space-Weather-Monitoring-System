"""
analysis.py
Correlation analysis, extreme event detection, cycle phase, and monthly statistics.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import pywt
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    """
    Compute Pearson r between x and y at integer lags from -max_lag to +max_lag.
    Convention: positive lag means x leads y.
    """
    records = []
    for lag in range(-max_lag, max_lag + 1):
        # Shift y backwards/forwards in time
        y_shifted = y.shift(-lag)
        
        # Calculate valid pairs (n)
        valid_mask = x.notna() & y_shifted.notna()
        n = valid_mask.sum()
        
        if n < 10:
            records.append({"lag_days": lag, "pearson_r": np.nan, "n": n})
        else:
            # Pandas native correlation handles the NaN dropping automatically
            r = x.corr(y_shifted)
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
# Periodly statistics
# ---------------------------------------------------------------------------

def compute_periodly_stats(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Resamples daily data into W, 2W, M, or Y periods with peak tracking.
    """
    valid_periods = ['W', '2W', 'M', 'Y']
    if period.upper().replace('E', '') not in valid_periods:
        raise ValueError(f"Period must be in {valid_periods}")

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
    """Resample daily data into monthly aggregates with peak tracking."""
    return compute_periodly_stats(df, "M")

# ---------------------------------------------------------------------------
# Duty Cycle
# ---------------------------------------------------------------------------

def duty_cycle(df: pd.DataFrame, col: str, window: str) -> pd.DataFrame:
    """
    Quantifies reliability based on valid observations vs. expected time-steps.
    Windows: '11YE' (Cycle), '10YE' (Decade), 'YE' (Annual), '27D' (Rotation)
    """
    config = {"11YE": 0.95, "10YE": 0.95, "YE": 0.90, "27D": 0.99}
    
    if window not in config:
        raise ValueError(f"Invalid window. Use: {list(config.keys())}")

    grouped = df[col].resample(window)
    result = pd.DataFrame({
        "expected_count": grouped.size(),
        "actual_count": grouped.count()
    })

    result["duty_cycle"] = result["actual_count"] / result["expected_count"]
    result["is_reliable"] = result["duty_cycle"] >= config[window]
    return result

# ----------------------------------------------------------------------------
# Analyse Periodicity
# ----------------------------------------------------------------------------

def analyze_periodicity(series: pd.Series, method: str = 'fft', fs: float = 1.0) -> pd.DataFrame:
    """
    Identifies dominant cycles using FFT (Global) or CWT (Temporal evolution).
    """
    clean_series = series.dropna()
    data = clean_series.values
    data = data - np.mean(data) # Detrending
    n = len(data)

    if method == 'fft':
        fft_values = np.fft.fft(data)
        freqs = np.fft.fftfreq(n, d=1/fs)
        idx = np.where(freqs > 0)[0]

        results = pd.DataFrame({
            "frequency": freqs[idx],
            "period_years": (1 / freqs[idx]) / 365.25,
            "amplitude": np.abs(fft_values)[idx]
        })
        return results.sort_values("amplitude", ascending=False).reset_index(drop=True)

    elif method == 'wv':
        scales = np.arange(365, 365 * 20, 15) # 1 to 20 year cycles
        coeffs, freqs = pywt.cwt(data, scales, 'cmor1.5-1.0', sampling_period=1/fs)
        power_matrix = np.abs(coeffs)**2
        periods_yrs = (1 / freqs) / 365.25

        peak_indices = np.argmax(power_matrix, axis=0)
        return pd.DataFrame({
            "dominant_period_yrs": periods_yrs[peak_indices],
            "max_power": np.max(power_matrix, axis=0)
        }, index=clean_series.index)

# --------------------------------------------------------------------------------
# Predictive Dominance
# --------------------------------------------------------------------------------

def predictive_dominance(df: pd.DataFrame, max_lag: int = 14):
    """
    Compares the predictive power of Sunspots vs Flares for Kp.
    """
    
    sn_corr = cross_correlation(df['sn'], df['Kp_mean'], max_lag)
    flare_corr = cross_correlation(df['flare_index'], df['Kp_mean'], max_lag)

    max_r_sn = sn_corr['pearson_r'].max()
    max_r_flare = flare_corr['pearson_r'].max()

    best_driver = "Sunspots" if max_r_sn > max_r_flare else "Flares"

    comparison = pd.DataFrame({
        "Metric": ["Max Correlation (r)", "Optimal Lag (Days)"],
        "Sunspots": [max_r_sn, sn_corr.loc[sn_corr['pearson_r'].idxmax(), 'lag_days']],
        "Flares": [max_r_flare, flare_corr.loc[flare_corr['pearson_r'].idxmax(), 'lag_days']]
    })

    print(f"Dominant Driver: {best_driver}")
    return comparison

# ----------------------------------------------------------------------------------
# Phase Locked Analysis
# ----------------------------------------------------------------------------------

def analyze_phase_locked_climatology(
    df: pd.DataFrame,
    kp_col: str = "Kp_mean",
    storm_threshold: float = 6.0,
    cycle_start: str = "1996-08-01",
    cycle_length_years: float = 11.0,
    bins: int = 20
) -> pd.DataFrame:
    """
    Groups Kp observations by their fractional solar phase to identify 
    'danger zones' in the 11-year cycle.

    Parameters
    ----------
    df                 : DataFrame with DatetimeIndex
    kp_col             : Name of the Kp column to analyze
    storm_threshold    : Kp value to define an 'extreme event'
    cycle_start        : ISO date string for reference cycle minimum
    cycle_length_years : Nominal cycle length for the modulo calculation
    bins               : Number of phase bins (default 20 = 5% phase steps)

    Returns
    -------
    pd.DataFrame indexed by phase_bin (0 to 1)
    Columns: kp_mean, kp_max, storm_prob_pct, n_days
    """
    
    t0 = pd.Timestamp(cycle_start)
    cycle_days = cycle_length_years * 365.25
    
    elapsed_days = (df.index - t0).total_seconds() / 86400.0
    
    phase = (elapsed_days % cycle_days) / cycle_days
    df_temp = df.copy()
    df_temp['cycle_phase'] = phase

    bin_edges = np.linspace(0, 1, bins + 1)
    df_temp['phase_bin'] = pd.cut(df_temp['cycle_phase'], bins=bin_edges, include_lowest=True, labels=bin_edges[:-1])

    grouped = df_temp.groupby('phase_bin', observed=True)
    
    climatology = pd.DataFrame({
        "kp_mean": grouped[kp_col].mean(),
        "kp_max": grouped[kp_col].max(),
        "n_days": grouped[kp_col].count()
    })

    storm_counts = grouped[kp_col].apply(lambda x: (x >= storm_threshold).sum())
    climatology["storm_prob_pct"] = (storm_counts / climatology["n_days"]) * 100

    return climatology.sort_index()

# -----------------------------------------------------------------------------
# Analyse Hysteresis
# -----------------------------------------------------------------------------

def analyze_hysteresis(
    df: pd.DataFrame,
    sn_col: str = "sn",
    kp_col: str = "Kp_mean",
    cycle_start: str = "1996-08-01",
    cycle_length_years: float = 11.0,
    save_path: str = "hysteresis_plot.png"
) -> pd.DataFrame:
    """
    Analyzes and saves a plot of the Kp response vs. Sunspot Number,
    categorized by the Rising and Falling phases of the solar cycle.

    Parameters
    ----------
    df                 : DataFrame with DatetimeIndex
    sn_col             : Sunspot number column
    kp_col             : Kp index column
    cycle_start        : ISO date string for reference cycle minimum
    cycle_length_years : Nominal cycle length
    save_path          : File path to save the generated scatter plot

    Returns
    -------
    pd.DataFrame with added 'cycle_phase' and 'phase_type' columns
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

    plt.figure(figsize=(10, 6))

    for p_type, color in zip(['Falling', 'Rising'], ['#d62728', '#1f77b4']):
        subset = df_plot[df_plot['phase_type'] == p_type]
        plt.scatter(
            subset[sn_col],
            subset[kp_col],
            alpha=0.3,
            label=f"{p_type} Phase",
            c=color,
            edgecolors='none',
            s=10
        )

    plt.title(f"Solar Hysteresis: {sn_col} vs {kp_col} (90-Year Window)")
    plt.xlabel("Sunspot Number (SN)")
    plt.ylabel("Geomagnetic Index (Kp)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

    return df_plot
