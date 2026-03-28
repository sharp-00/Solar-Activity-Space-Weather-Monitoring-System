"""
analysis.py
Correlation analysis, extreme event detection, cycle phase, and monthly statistics.
"""

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
    if "sn" in df.columns:
        agg.update({"sn_mean": ("sn", "mean"), "sn_max": ("sn", "max")})
    if "Kp_mean" in df.columns:
        agg["Kp_mean"] = ("Kp_mean", "mean")
    if "Kp_max" in df.columns:
        agg["Kp_max"] = ("Kp_max", "max")
    if "Kp_storm_hours" in df.columns:
        agg["storm_hours"] = ("Kp_storm_hours", "sum")

    result = df.resample(period + "E").agg(**agg)
    result.index.name = "date"
    return result

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

