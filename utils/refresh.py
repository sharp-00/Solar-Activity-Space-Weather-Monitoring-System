"""utils/refresh.py — 6-hour auto-refresh + manual sidebar button."""
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_loader import run_pipeline

REFRESH_INTERVAL = 6 * 3600  # seconds


def _last() -> float:
    return st.session_state.get("last_fetch_epoch", 0.0)


def _set_last():
    st.session_state["last_fetch_epoch"] = time.time()


def _data_exists() -> bool:
    """Return True if any recognised data file is present on disk."""
    candidates = [
        Path("data/clean/solar_weather_daily.parquet"),
        Path("data/clean/solar_weather_daily.csv"),
        Path("data/solar_weather_daily.csv"),
    ]
    return any(p.exists() for p in candidates)


def auto_refresh_check():
    """
    Called at the top of every page.
    · No data on disk            → run full pipeline (first-time setup)
    · Data exists, no session fetch yet → load from disk, don't re-download
    · Data exists, fetched this session → re-fetch after 6 h
    """
    if not _data_exists():
        _trigger(auto=True)
        return
    # _last() == 0.0 means the parquet came from disk (e.g. cloned repo).
    # Only start the 6h countdown after the user's first manual or auto fetch.
    if _last() > 0 and (time.time() - _last()) > REFRESH_INTERVAL:
        _trigger(auto=True)


def _trigger(auto: bool = False):
    label = "🔄 Auto-updating data…" if auto else "⚡ Fetching latest data…"
    with st.sidebar:
        with st.status(label, expanded=True) as s:
            try:
                run_pipeline(st.write)
                _set_last()
                s.update(label="✅ Data up to date!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                s.update(label="❌ Refresh failed", state="error", expanded=True)
                st.error(str(e))


def render_refresh_sidebar():
    """Sidebar data-control block — call on every page."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔄 Data Controls")

        last = _last()
        if last:
            dt = datetime.fromtimestamp(last, tz=timezone.utc)
            st.caption(f"Updated: {dt.strftime('%Y-%m-%d %H:%M UTC')}")
        else:
            st.caption("Loaded from disk. Click below to fetch latest.")

        if st.button("⚡ Fetch Latest Data", use_container_width=True, type="primary"):
            _trigger(auto=False)

        st.caption("Auto-refreshes every 6 hours after first fetch.")


def render_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar date slider — persists across page navigations via session_state."""
    if df.empty:
        return df

    min_d = df.index.min().date()
    max_d = df.index.max().date()

    prev = st.session_state.get("date_range", (min_d, max_d))
    prev = (max(prev[0], min_d), min(prev[1], max_d))

    with st.sidebar:
        st.markdown("### 📅 Observation Window")
        date_range = st.slider(
            "Date range",
            min_value=min_d, max_value=max_d, value=prev,
            format="YYYY-MM-DD", label_visibility="collapsed",
        )

    st.session_state["date_range"] = date_range
    return df.loc[pd.Timestamp(date_range[0]):pd.Timestamp(date_range[1])]
