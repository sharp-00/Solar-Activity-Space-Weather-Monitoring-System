"""analysis.py — Statistical analysis functions for the solar weather pipeline."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

try:
    import pywt
    _PYWT_OK = True
except ImportError:
    _PYWT_OK = False

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
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — safe in Streamlit
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

# Pandas >= 2.2 renamed resample aliases: "M" → "ME", "Y" → "YE", "Q" → "QE"
def _resample_alias(freq: str) -> str:
    """Return the correct resample frequency string for the installed pandas."""
    _MAP = {"M": "ME", "Y": "YE", "Q": "QE", "A": "YE"}
    major = int(pd.__version__.split(".")[0])
    minor = int(pd.__version__.split(".")[1])
    if (major, minor) >= (2, 2):
        return _MAP.get(freq, freq)
    return freq


# ── Cross-correlation ─────────────────────────────────────────────────────────

def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    """Pearson r between x and y at integer lags from -max_lag to +max_lag."""
    records = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs, ys = x.iloc[lag:].values, y.iloc[:len(x)-lag].values
        else:
            xs, ys = x.iloc[:len(x)+lag].values, y.iloc[-lag:].values
        mask = ~np.isnan(xs) & ~np.isnan(ys)
        n = mask.sum()
        if n < 10:
            records.append({"lag_days": lag, "pearson_r": np.nan, "n": n})
            continue
        r, _ = pearsonr(xs[mask], ys[mask])
        records.append({"lag_days": lag, "pearson_r": r, "n": n})
    return pd.DataFrame(records)


# ── Extreme events ────────────────────────────────────────────────────────────
>>>>>>> 13cb7cd93d84e80731ff958e001d47d175a08c8c

def find_extreme_events(df: pd.DataFrame, col: str,
                         threshold_sigma: float = 2.5) -> pd.DataFrame:
    """Return rows where col is more than threshold_sigma std devs above the mean."""
    series = df[col].dropna()
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        return pd.DataFrame()
    z = (df[col] - mu) / sigma
    mask = z.abs() > threshold_sigma
    result = df.loc[mask, [col]].copy()
    result["z_score"] = z[mask]
    return result.sort_values("z_score", ascending=False)


# ── Solar cycle phase ─────────────────────────────────────────────────────────

def solar_cycle_phase(date_index: pd.DatetimeIndex,
                      cycle_start: str = "1996-08-01",
                      cycle_length_years: float = 11.0) -> pd.Series:
    t0 = pd.Timestamp(cycle_start)
    cycle_days = cycle_length_years * 365.25
    elapsed = (date_index - t0).total_seconds() / 86400.0
    phase = (elapsed % cycle_days) / cycle_days
    return pd.Series(phase, index=date_index, name="cycle_phase")


# ── Monthly / periodic statistics ─────────────────────────────────────────────

def compute_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to monthly aggregates."""
    alias = _resample_alias("M")
    agg = {}
    for col, agg_fn, out in [
        ("ssn",          "mean", "ssn_mean"),
        ("ssn",          "max",  "ssn_max"),
        ("kp_daily_mean","mean", "kp_daily_mean"),
        ("kp_daily_max", "max",  "kp_daily_max"),
        ("dst_daily_min","min",  "dst_daily_min"),
        ("f107",         "mean", "f107_mean"),
    ]:
        if col in df.columns and df[col].notna().any():
            agg[out] = (col, agg_fn)
    if not agg:
        return pd.DataFrame()
    result = df.resample(alias).agg(**agg)
    # Convert PeriodIndex → DatetimeIndex so Plotly renders x-axis as dates
    if hasattr(result.index, "to_timestamp"):
        result.index = result.index.to_timestamp()
    result.index.name = "date"
    return result


# ── Periodicity ───────────────────────────────────────────────────────────────

def analyze_periodicity(series: pd.Series, method: str = "fft",
                         fs: float = 1.0) -> pd.DataFrame:
    """FFT or CWT (wavelet) periodicity analysis."""
    clean = series.dropna()
    if len(clean) < 100:
        raise ValueError("Need at least 100 data points for periodicity analysis.")
    data = clean.values - np.mean(clean.values)
    n = len(data)

    if method == "fft":
        fft_vals = np.fft.fft(data)
        freqs    = np.fft.fftfreq(n, d=1.0 / fs)
        idx      = np.where(freqs > 0)[0]
        with np.errstate(divide="ignore"):
            period_years = np.where(freqs[idx] > 0, (1.0 / freqs[idx]) / 365.25, np.nan)
        return pd.DataFrame({
            "frequency":    freqs[idx],
            "period_years": period_years,
            "amplitude":    np.abs(fft_vals)[idx],
        }).sort_values("amplitude", ascending=False).reset_index(drop=True)

    elif method == "wv":
        if not _PYWT_OK:
            raise ImportError(
                "pywavelets is not installed. "
                "Run: pip install pywavelets  then restart the app."
            )
        scales = np.arange(365, 365 * 20, 15)
        coeffs, freqs = pywt.cwt(data, scales, "cmor1.5-1.0", sampling_period=1.0 / fs)
        power  = np.abs(coeffs) ** 2
        p_yrs  = np.where(freqs > 0, (1.0 / freqs) / 365.25, np.nan)
        return pd.DataFrame({
            "dominant_period_yrs": p_yrs[np.argmax(power, axis=0)],
            "max_power":           np.max(power, axis=0),
        }, index=clean.index)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'fft' or 'wv'.")


# ── Phase-locked climatology ───────────────────────────────────────────────────

def analyze_phase_locked_climatology(df: pd.DataFrame,
                                      kp_col: str = "kp_daily_max",
                                      storm_threshold: float = 6.0,
                                      cycle_start: str = "1996-08-01",
                                      cycle_length_years: float = 11.0,
                                      bins: int = 20) -> pd.DataFrame:
    """Group Kp by fractional solar cycle phase to find storm danger zones."""
    if kp_col not in df.columns or df[kp_col].dropna().empty:
        raise ValueError(f"Column '{kp_col}' missing or all-NaN.")

    t0 = pd.Timestamp(cycle_start)
    cycle_days = cycle_length_years * 365.25
    elapsed = (df.index - t0).total_seconds() / 86400.0
    phase   = (elapsed % cycle_days) / cycle_days

    tmp = df[[kp_col]].copy()
    tmp["cycle_phase"] = phase
    bin_edges = np.linspace(0, 1, bins + 1)
    tmp["phase_bin"] = pd.cut(tmp["cycle_phase"], bins=bin_edges,
                               include_lowest=True, labels=bin_edges[:-1])
    g = tmp.groupby("phase_bin", observed=True)
    clim = pd.DataFrame({
        "kp_mean":       g[kp_col].mean(),
        "kp_max":        g[kp_col].max(),
        "n_days":        g[kp_col].count(),
        "storm_prob_pct": g[kp_col].apply(lambda x: (x >= storm_threshold).sum())
                          / g[kp_col].count() * 100,
    })
    return clim.sort_index()


# ── Hysteresis ────────────────────────────────────────────────────────────────

def analyze_hysteresis(df: pd.DataFrame,
                        sn_col: str = "ssn",
                        kp_col: str = "kp_daily_max",
                        cycle_start: str = "1996-08-01",
                        cycle_length_years: float = 11.0,
                        save_path: str = None) -> pd.DataFrame:
    """Label each day as Rising or Falling phase of the solar cycle."""
    for col in [sn_col, kp_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe.")

    t0 = pd.Timestamp(cycle_start)
    cycle_days = cycle_length_years * 365.25
    elapsed = (df.index - t0).total_seconds() / 86400.0
    phase   = (elapsed % cycle_days) / cycle_days

    out = df[[sn_col, kp_col]].copy()
    out["cycle_phase"] = phase
    out["phase_type"]  = np.where(out["cycle_phase"] < 0.5, "Rising", "Falling")

    # Save matplotlib scatter only if requested and matplotlib is available
    if save_path and _MPL_OK:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            for p_type, color in [("Falling", "#d62728"), ("Rising", "#1f77b4")]:
                sub = out[out["phase_type"] == p_type]
                ax.scatter(sub[sn_col], sub[kp_col], alpha=0.3,
                           label=f"{p_type} Phase", c=color,
                           edgecolors="none", s=10)
            ax.set_xlabel("Sunspot Number")
            ax.set_ylabel("Kp Index")
            ax.set_title("Solar Hysteresis")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass   # Non-critical — plotting failure shouldn't break analysis

    return out


# ── Predictive dominance (legacy, kept for compatibility) ──────────────────────

def predictive_dominance(df: pd.DataFrame, max_lag: int = 14):
    """Compare predictive power of SSN vs flares for Kp."""
    sn_col    = next((c for c in ["ssn","sn"] if c in df.columns), None)
    kp_col    = next((c for c in ["kp_daily_mean","kp_daily_max"] if c in df.columns), None)
    flare_col = next((c for c in ["flare_xray_total","flare_events_total"] if c in df.columns), None)

    if not all([sn_col, kp_col, flare_col]):
        return pd.DataFrame()

    sn_corr    = cross_correlation(df[sn_col].dropna(),    df[kp_col].dropna(), max_lag)
    flare_corr = cross_correlation(df[flare_col].dropna(), df[kp_col].dropna(), max_lag)

    return pd.DataFrame({
        "Metric":    ["Max Correlation (r)", "Optimal Lag (Days)"],
        "Sunspots":  [sn_corr["pearson_r"].max(),
                      sn_corr.loc[sn_corr["pearson_r"].abs().idxmax(), "lag_days"]],
        "Flares":    [flare_corr["pearson_r"].max(),
                      flare_corr.loc[flare_corr["pearson_r"].abs().idxmax(), "lag_days"]],
    })
